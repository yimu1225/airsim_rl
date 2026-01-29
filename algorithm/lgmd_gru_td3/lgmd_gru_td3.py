import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import Actor, Critic, DepthEncoder, MotionEncoder, GRUEncoder, D_LGMD, BaseStateExpander
from .buffer import SequenceReplayBuffer
from ..ou_noise import OUNoise


class D_LGMDGRUTD3Agent:
    """
    TD3 with GRU and D-LGMD for POMDP.
    State: Sequence of k steps.
    Each step: Base State + Visual Features (Depth + Motion from D-LGMD).
    Motion features are processed separately and concatenated with GRU output.
    """

    def __init__(self, base_dim, depth_shape, gray_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"TD3-GRU-D-LGMD Agent using device: {self.device}")

        self.base_dim = base_dim
        self.depth_shape = depth_shape # (1, H, W)
        self.gray_shape = gray_shape   # (M, H, W)
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
        _, h, w = depth_shape
        m_channels = gray_shape[0]
        feature_dim = args.feature_dim
        
        # D-LGMD for motion detection
        self.d_lgmd = D_LGMD(input_height=h, input_width=w, device=self.device)
        
        # Separate encoders for depth and motion
        self.depth_encoder = DepthEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.motion_encoder = MotionEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        
        # Base State Expanders (separate for actor/critic)
        expanded_base_dim = 32
        self.actor_base_expander = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.actor_base_expander_target = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.actor_base_expander_target.load_state_dict(self.actor_base_expander.state_dict())

        self.critic_base_expander = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.critic_base_expander_target = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.critic_base_expander_target.load_state_dict(self.critic_base_expander.state_dict())
        
        # GRU Encoder - only processes depth features
        self.gru_hidden_dim = args.gru_hidden_dim
        gru_layers = getattr(args, 'gru_num_layers', 1)
        
        # Actor GRU
        self.actor_gru = GRUEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.depth_encoder.repr_dim, hidden_dim=self.gru_hidden_dim, num_layers=gru_layers).to(self.device)
        self.actor_gru_target = GRUEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.depth_encoder.repr_dim, hidden_dim=self.gru_hidden_dim, num_layers=gru_layers).to(self.device)
        self.actor_gru_target.load_state_dict(self.actor_gru.state_dict())
        
        # Critic GRU
        self.critic_gru = GRUEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.depth_encoder.repr_dim, hidden_dim=self.gru_hidden_dim, num_layers=gru_layers).to(self.device)
        self.critic_gru_target = GRUEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.depth_encoder.repr_dim, hidden_dim=self.gru_hidden_dim, num_layers=gru_layers).to(self.device)
        self.critic_gru_target.load_state_dict(self.critic_gru.state_dict())
        
        # State dim for Actor/Critic is GRU output + motion features
        self.state_dim = self.gru_hidden_dim + self.motion_encoder.repr_dim

        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers (D-LGMD has fixed weights, no optimizer needed)
        self.actor_optimizer = Adam(
            list(self.actor.parameters()) + list(self.depth_encoder.parameters()) + list(self.motion_encoder.parameters()) + list(self.actor_base_expander.parameters()) + list(self.actor_gru.parameters()), 
            lr=args.actor_lr
        )
        self.critic_optimizer = Adam(
            list(self.critic.parameters()) + list(self.depth_encoder.parameters()) + list(self.motion_encoder.parameters()) + list(self.critic_base_expander.parameters()) + list(self.critic_gru.parameters()), 
            lr=args.critic_lr
        )

        self.replay_buffer = SequenceReplayBuffer(args.buffer_size, self.seq_len)
        
        self.discount = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        self.grad_clip = getattr(args, "grad_clip", 1.0)

        # OU Noise initialization
        self.ou_noise = OUNoise(
            size=self.action_dim,
            mu=0.0,
            theta=getattr(args, 'ou_theta', 0.15),
            sigma=getattr(args, 'ou_sigma', 0.2),
            sigma_min=getattr(args, 'ou_sigma_min', 0.01),
            dt=getattr(args, 'ou_dt', 1.0)
        )
        self.ou_noise.reset()

        self.total_it = 0

    def _process_sequence(self, base, depth, gray, depth_encoder, motion_encoder, gru_encoder, base_expander, detach_encoder=False):
        """
        Process sequence with D-LGMD and GRU.
        Args:
            base: (B, K, D_base)
            depth: (B, K, 1, H, W)
            gray: (B, K, M, H, W) - for D-LGMD motion processing
        Returns:
            state: (B, gru_hidden_dim + motion_repr_dim)
        """
        B, K, _, H, W = depth.shape
        M = gray.shape[2]
        
        # Process D-LGMD on the entire gray sequence
        # D-LGMD expects (B, T, C, H, W) or (B, T, H, W)
        # Convert to proper format: (B, K, M, H, W) -> (B, K*M, 1, H, W) if M>1, or (B, K, 1, H, W)
        if M == 1:
            gray_for_lgmd = gray  # (B, K, 1, H, W)
        else:
            # Stack multiple gray channels as time sequence
            gray_for_lgmd = gray  # D-LGMD can handle (B, T, H, W) format
        
        # D-LGMD processing - returns (B, K-3, 1, H, W) for valid time steps
        motion_sequence = self.d_lgmd(gray_for_lgmd)  # (B, K-3, 1, H, W)
        
        # For simplicity, we'll use the last motion frame or average motion
        # Here we use the last motion frame
        motion_last = motion_sequence[:, -1:, :, :, :]  # (B, 1, 1, H, W)
        motion_last_flat = motion_last_flat = motion_last.view(B * 1, 1, H, W)
        
        # Motion feature extraction
        motion_feat = motion_encoder(motion_last_flat)  # (B, motion_repr_dim)
        
        # Process depth for temporal module
        depth_flat = depth.contiguous().view(B * K, 1, H, W)
        depth_feat = depth_encoder(depth_flat)  # (B*K, depth_repr_dim)
        
        # Reshape back to sequence
        depth_feat = depth_feat.view(B, K, -1)  # (B, K, depth_repr_dim)
        
        if detach_encoder:
            depth_feat = depth_feat.detach()
        
        # Expand base state
        base_expanded = base_expander(base)  # (B, K, expanded_base_dim)
            
        # GRU Processing - only processes depth features
        gru_state = gru_encoder(base_expanded, depth_feat)  # (B, gru_hidden_dim)
        
        # Concatenate GRU output with motion features
        state = torch.cat([gru_state, motion_feat], dim=1)  # (B, gru_hidden_dim + motion_repr_dim)
        
        return state

    def select_action(self, base_seq, depth_seq, gray_seq, noise=True):
        # Inputs are numpy arrays of shape (K, ...)
        # Add batch dimension
        base = torch.as_tensor(base_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        depth = torch.as_tensor(depth_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        gray = torch.as_tensor(gray_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            state = self._process_sequence(base, depth, gray, self.depth_encoder, self.motion_encoder, self.actor_gru, self.actor_base_expander)
            action = self.actor(state).cpu().numpy().flatten()

        if noise:
            action = action + self.ou_noise.sample()
        
        action = np.clip(action, -1.0, 1.0)
        return action * self.action_scale + self.action_bias

    def train(self, progress_ratio=0.0):
        self.total_it += 1
        
        # 应用OU噪声衰减
        self.ou_noise.scale_sigma(progress_ratio)
        
        if self.replay_buffer.size < self.batch_size:
            return

        # Sample sequences
        (base, depth, gray, action, reward,
         next_base, next_depth, next_gray, done) = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        # Gray is float16 from buffer, move to GPU then convert to float
        gray = torch.as_tensor(gray, device=self.device).float()
        action = torch.as_tensor(action[:, -1, :], dtype=torch.float32, device=self.device) # Action at last step
        # Reward and Done are already (B, 1) from buffer slicing, no need to unsqueeze
        reward = torch.as_tensor(reward[:, -1], dtype=torch.float32, device=self.device) # Reward at last step
        done = torch.as_tensor(done[:, -1], dtype=torch.float32, device=self.device)     # Done at last step

        next_base = torch.as_tensor(next_base, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)
        # Gray is float16 from buffer
        next_gray = torch.as_tensor(next_gray, device=self.device).float()

        # Normalize action from buffer to [-1, 1]
        action = (action - self.action_bias_tensor) / self.action_scale_tensor

        # Critic Feature Extraction
        state = self._process_sequence(base, depth, gray, self.depth_encoder, self.motion_encoder, self.critic_gru, self.critic_base_expander, detach_encoder=False)

        with torch.no_grad():
            # Critic Target Features
            next_state = self._process_sequence(next_base, next_depth, next_gray, self.depth_encoder, self.motion_encoder, self.critic_gru_target, self.critic_base_expander_target, detach_encoder=True)
            
            # Actor Target Features
            next_state_actor = self._process_sequence(next_base, next_depth, next_gray, self.depth_encoder, self.motion_encoder, self.actor_gru_target, self.actor_base_expander_target, detach_encoder=True)
            
            # TD3 Target
            next_action = self.actor_target(next_state_actor)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.motion_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_base_expander.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_gru.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            # Actor Feature Extraction
            state_actor = self._process_sequence(base, depth, gray, self.depth_encoder, self.motion_encoder, self.actor_gru, self.actor_base_expander, detach_encoder=False)
            
            with torch.no_grad():
                state_critic = self._process_sequence(base, depth, gray, self.depth_encoder, self.motion_encoder, self.critic_gru, self.critic_base_expander, detach_encoder=False)
            
            q1, _ = self.critic(state_critic, self.actor(state_actor))
            actor_loss = -q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.motion_encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_base_expander.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_gru.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.depth_encoder.parameters(), self.depth_encoder_target.parameters() if hasattr(self, 'depth_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.motion_encoder.parameters(), self.motion_encoder_target.parameters() if hasattr(self, 'motion_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_base_expander.parameters(), self.critic_base_expander_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_gru.parameters(), self.critic_gru_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.depth_encoder.parameters(), self.depth_encoder_target.parameters() if hasattr(self, 'depth_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.motion_encoder.parameters(), self.motion_encoder_target.parameters() if hasattr(self, 'motion_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_base_expander.parameters(), self.actor_base_expander_target.parameters()):
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
            
            'depth_encoder': self.depth_encoder.state_dict(),
            'motion_encoder': self.motion_encoder.state_dict(),
            'actor_gru': self.actor_gru.state_dict(),
            'actor_gru_target': self.actor_gru_target.state_dict(),
            'actor_base_expander': self.actor_base_expander.state_dict(),
            'actor_base_expander_target': self.actor_base_expander_target.state_dict(),
            
            'critic_gru': self.critic_gru.state_dict(),
            'critic_gru_target': self.critic_gru_target.state_dict(),
            'critic_base_expander': self.critic_base_expander.state_dict(),
            'critic_base_expander_target': self.critic_base_expander_target.state_dict(),
            
            'd_lgmd': self.d_lgmd.state_dict(),
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
        
        self.d_lgmd.load_state_dict(checkpoint['d_lgmd'])
        
        if 'depth_encoder' in checkpoint:
            self.depth_encoder.load_state_dict(checkpoint['depth_encoder'])
            self.motion_encoder.load_state_dict(checkpoint['motion_encoder'])
            self.actor_gru.load_state_dict(checkpoint['actor_gru'])
            self.actor_gru_target.load_state_dict(checkpoint['actor_gru_target'])
            if 'actor_base_expander' in checkpoint:
                self.actor_base_expander.load_state_dict(checkpoint['actor_base_expander'])
                self.actor_base_expander_target.load_state_dict(checkpoint['actor_base_expander_target'])
            
            self.critic_gru.load_state_dict(checkpoint['critic_gru'])
            self.critic_gru_target.load_state_dict(checkpoint['critic_gru_target'])
            if 'critic_base_expander' in checkpoint:
                self.critic_base_expander.load_state_dict(checkpoint['critic_base_expander'])
                self.critic_base_expander_target.load_state_dict(checkpoint['critic_base_expander_target'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
        self.total_it = checkpoint['total_it']

def make_agent(env, initial_obs, args, device=None):
    # Assuming initial_obs has structure for shapes
    # But we need shapes from args or env usually
    # Here we infer from initial_obs if available
    base = initial_obs["base"]
    depth = initial_obs["depth"]
    gray = initial_obs["gray"] # Assuming this exists in obs
    
    return D_LGMDGRUTD3Agent(
        base_dim=base.shape[-1],
        depth_shape=depth.shape[-3:], # (1, H, W)
        gray_shape=gray.shape[-3:],   # (M, H, W)
        action_space=env.action_space,
        args=args,
        device=device
    )
