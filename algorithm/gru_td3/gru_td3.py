import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import Actor, Critic, VisualEncoder, GRUEncoder
from .buffer import SequenceReplayBuffer


class GRUTD3Agent:
    """
    TD3 with GRU for POMDP.
    State: Sequence of k steps.
    Each step: Base State + Visual Features (Depth).
    """

    def __init__(self, base_dim, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"TD3-GRU Agent using device: {self.device}")

        self.base_dim = base_dim
        self.depth_shape = depth_shape # (1, H, W)
        self.seq_len = args.seq_len    # k
        
        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.max_action_tensor = torch.from_numpy(self.max_action).float().to(self.device)
        self.min_action_tensor = torch.from_numpy(self.min_action).float().to(self.device)

        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0
        self.action_scale_tensor = torch.from_numpy(self.action_scale).float().to(self.device)
        self.action_bias_tensor = torch.from_numpy(self.action_bias).float().to(self.device)

        # Encoders
        C, h, w = depth_shape
        feature_dim = args.feature_dim

        # CRITIC Encoders
        self.critic_visual_encoder = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_visual_encoder_target = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_visual_encoder_target.load_state_dict(self.critic_visual_encoder.state_dict())
        visual_feature_dim = self.critic_visual_encoder.repr_dim
        
        # GRU Encoder (Critic)
        self.gru_hidden_dim = args.gru_hidden_dim
        gru_layers = getattr(args, 'gru_num_layers', 1)
        self.critic_gru = GRUEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.gru_hidden_dim,
            num_layers=gru_layers
        ).to(self.device)
        self.critic_gru_target = GRUEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.gru_hidden_dim,
            num_layers=gru_layers
        ).to(self.device)
        self.critic_gru_target.load_state_dict(self.critic_gru.state_dict())
        
        # ACTOR Encoders
        self.actor_visual_encoder = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_visual_encoder_target = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_visual_encoder_target.load_state_dict(self.actor_visual_encoder.state_dict())

        self.actor_gru = GRUEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.gru_hidden_dim,
            num_layers=gru_layers
        ).to(self.device)
        self.actor_gru_target = GRUEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.gru_hidden_dim,
            num_layers=gru_layers
        ).to(self.device)
        self.actor_gru_target.load_state_dict(self.actor_gru.state_dict())
        
        # State dim for Actor/Critic is GRU output + current base state
        self.state_dim = self.gru_hidden_dim + self.base_dim

        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_params = list(self.actor.parameters()) + list(self.actor_visual_encoder.parameters()) + list(self.actor_gru.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        self.critic_params = list(self.critic.parameters()) + list(self.critic_visual_encoder.parameters()) + list(self.critic_gru.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = SequenceReplayBuffer(args.buffer_size, self.seq_len)
        
        self.discount = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        self.grad_clip = getattr(args, "grad_clip", 1.0)
        
        self.exploration_noise = args.exploration_noise


        self.total_it = 0

    def _process_sequence(self, base, depth, visual_encoder, gru_encoder, detach_encoder=False):
        """
        Process sequence with frame-by-frame GRU.
        GRU only processes visual features, base state is concatenated at the end.
        Args:
            base: (B, K, D_base)
            depth: (B, K, 1, H, W)
        Returns:
            state: (B, gru_hidden_dim + base_dim) - GRU output concatenated with current base state
        """
        B, K, _, H, W = depth.shape
        
        # Visual Encoding: Batch process for efficiency
        # Note: CNN is applied independently to each frame, so
        # batch processing (B*K frames at once) is mathematically
        # equivalent to frame-by-frame processing, but much faster
        depth_flat = depth.view(B * K, 1, H, W)
        visual_feat = visual_encoder(depth_flat)  # (B*K, feature_dim)
        visual_feat = visual_feat.view(B, K, -1)  # (B, K, feature_dim)
        
        if detach_encoder:
            visual_feat = visual_feat.detach()
            
        # GRU Processing: Only visual features
        gru_state = gru_encoder(visual_feat)  # (B, gru_hidden_dim)
        
        # Get current base state (last timestep)
        current_base = base[:, -1, :]  # (B, base_dim)
        
        # Concatenate GRU output with current base state
        state = torch.cat([gru_state, current_base], dim=1)  # (B, gru_hidden_dim + base_dim)
        
        return state

    def select_action(self, base_seq, depth_seq, noise=True):
        # Inputs are numpy arrays of shape (K, ...)
        # Add batch dimension
        base = torch.as_tensor(base_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        depth = torch.as_tensor(depth_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            state = self._process_sequence(base, depth, self.actor_visual_encoder, self.actor_gru)
            action = self.actor(state).cpu().numpy().flatten()

        if noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
        
        action = np.clip(action, -1.0, 1.0)
        return action * self.action_scale + self.action_bias

    def train(self, progress_ratio=0.0):
        self.total_it += 1
        if self.replay_buffer.size < self.batch_size:
            return

        # Sample sequences
        (base, depth, action, reward,next_base, next_depth, done) = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action[:, -1, :], dtype=torch.float32, device=self.device) # Action at last step
        # Reward and Done are already (B, 1) from buffer slicing, no need to unsqueeze
        reward = torch.as_tensor(reward[:, -1], dtype=torch.float32, device=self.device) # Reward at last step
        done = torch.as_tensor(done[:, -1], dtype=torch.float32, device=self.device)     # Done at last step

        next_base = torch.as_tensor(next_base, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)

        # Normalize action from buffer to [-1, 1]
        action = (action - self.action_bias_tensor) / self.action_scale_tensor

        # Forward pass for current state (with gradients, using Critic encoders)
        state = self._process_sequence(base, depth, self.critic_visual_encoder, self.critic_gru, detach_encoder=False)

        with torch.no_grad():
            # Forward pass for next state (no gradients)
            next_state = self._process_sequence(next_base, next_depth, self.critic_visual_encoder_target, self.critic_gru_target, detach_encoder=True)

            # TD3 Target
            next_state_actor = self._process_sequence(next_base, next_depth, self.actor_visual_encoder_target, self.actor_gru_target, detach_encoder=True)
            next_action = self.actor_target(next_state_actor)

            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Critic Update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_visual_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_gru.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Actor Update
        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            # 1. State from Actor (gradients flow to actor encoders)
            state_actor = self._process_sequence(base, depth, self.actor_visual_encoder, self.actor_gru, detach_encoder=False)
            
            # 2. State for Critic (detached/no_grad)
            with torch.no_grad():
                state_critic = self._process_sequence(base, depth, self.critic_visual_encoder, self.critic_gru, detach_encoder=False)

            q1, _ = self.critic(state_critic, self.actor(state_actor))
            actor_loss = -q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_visual_encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_gru.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Soft Updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_visual_encoder.parameters(), self.critic_visual_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_gru.parameters(), self.critic_gru_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_visual_encoder.parameters(), self.actor_visual_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_gru.parameters(), self.actor_gru_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            'critic_loss': critic_loss.item(),
        }

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            
            'actor_visual_encoder': self.actor_visual_encoder.state_dict(),
            'actor_visual_encoder_target': self.actor_visual_encoder_target.state_dict(),
            'actor_gru': self.actor_gru.state_dict(),
            'actor_gru_target': self.actor_gru_target.state_dict(),
            'critic_visual_encoder': self.critic_visual_encoder.state_dict(),
            'critic_visual_encoder_target': self.critic_visual_encoder_target.state_dict(),
            'critic_gru': self.critic_gru.state_dict(),
            'critic_gru_target': self.critic_gru_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            
            'total_it': self.total_it
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        if 'actor_target' in checkpoint:
            self.actor_target.load_state_dict(checkpoint['actor_target'])
        
        self.critic.load_state_dict(checkpoint['critic'])
        if 'critic_target' in checkpoint:
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        if 'actor_visual_encoder' in checkpoint:
            self.actor_visual_encoder.load_state_dict(checkpoint['actor_visual_encoder'])
            self.actor_visual_encoder_target.load_state_dict(checkpoint['actor_visual_encoder_target'])
            self.actor_gru.load_state_dict(checkpoint['actor_gru'])
            self.actor_gru_target.load_state_dict(checkpoint['actor_gru_target'])
            self.critic_visual_encoder.load_state_dict(checkpoint['critic_visual_encoder'])
            self.critic_visual_encoder_target.load_state_dict(checkpoint['critic_visual_encoder_target'])
            self.critic_gru.load_state_dict(checkpoint['critic_gru'])
            self.critic_gru_target.load_state_dict(checkpoint['critic_gru_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        else:
            self.actor_visual_encoder.load_state_dict(checkpoint['visual_encoder'])
            self.actor_visual_encoder_target.load_state_dict(checkpoint['visual_encoder'])
            self.critic_visual_encoder.load_state_dict(checkpoint['visual_encoder'])
            self.critic_visual_encoder_target.load_state_dict(checkpoint['visual_encoder'])
            
            self.actor_gru.load_state_dict(checkpoint['gru'])
            self.actor_gru_target.load_state_dict(checkpoint['gru'])
            self.critic_gru.load_state_dict(checkpoint['gru'])
            self.critic_gru_target.load_state_dict(checkpoint['gru'])
            
        self.total_it = checkpoint['total_it']

def make_agent(env, initial_obs, args, device=None):
    # Assuming initial_obs has structure for shapes
    # But we need shapes from args or env usually
    # Here we infer from initial_obs if available
    base = initial_obs["base"]
    depth = initial_obs["depth"]
    
    return GRUTD3Agent(
        base_dim=base.shape[-1],
        depth_shape=depth.shape[-3:], # (1, H, W)
        action_space=env.action_space,
        args=args,
        device=device
    )
