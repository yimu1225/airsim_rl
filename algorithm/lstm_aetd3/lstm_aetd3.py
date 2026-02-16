import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import Actor, Critic, VisualEncoder, LSTMEncoder, MetaNet
from .buffer import SequenceReplayBuffer


class LSTMAETD3Agent:
    """
    AETD3 with LSTM for POMDP.
    State: Sequence of k steps.
    Each step: Base State + Visual Features (Depth + Motion).
    """

    def __init__(self, base_dim, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"AETD3-LSTM Agent using device: {self.device}")  # 调试：确认设备

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
        
        # LSTM Encoder (Critic)
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.critic_lstm = LSTMEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.lstm_hidden_dim
        ).to(self.device)
        self.critic_lstm_target = LSTMEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.lstm_hidden_dim
        ).to(self.device)
        self.critic_lstm_target.load_state_dict(self.critic_lstm.state_dict())
        
        # ACTOR Encoders
        self.actor_visual_encoder = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_visual_encoder_target = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_visual_encoder_target.load_state_dict(self.actor_visual_encoder.state_dict())
        
        self.actor_lstm = LSTMEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.lstm_hidden_dim
        ).to(self.device)
        self.actor_lstm_target = LSTMEncoder(
            visual_feature_dim=visual_feature_dim,
            hidden_dim=self.lstm_hidden_dim
        ).to(self.device)
        self.actor_lstm_target.load_state_dict(self.actor_lstm.state_dict())
        
        # State dim for Actor/Critic is LSTM output + current base state
        self.state_dim = self.lstm_hidden_dim + self.base_dim

        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Adaptive Ensemble
        self.adaptive_k = int(args.adaptive_k)
        self.adaptive_reg = args.adaptive_reg
        meta_input_dim = self.state_dim + 2
        self.meta_net = MetaNet(meta_input_dim, self.adaptive_k).to(self.device)

        # Optimizers
        self.actor_params = list(self.actor.parameters()) + list(self.actor_visual_encoder.parameters()) + list(self.actor_lstm.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        self.critic_params = list(self.critic.parameters()) + list(self.critic_visual_encoder.parameters()) + list(self.critic_lstm.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)
        
        self.meta_optimizer = Adam(self.meta_net.parameters(), lr=args.adaptive_meta_lr)

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

    def _process_sequence(self, base, depth, visual_encoder, lstm_encoder, detach_encoder=False):
        """
        Process sequence with frame-by-frame LSTM.
        LSTM only processes visual features, base state is concatenated at the end.
        Args:
            base: (B, K, D_base)
            depth: (B, K, 1, H, W)
        Returns:
            state: (B, lstm_hidden_dim + base_dim) - LSTM output concatenated with current base state
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
            
        # LSTM Processing: Only visual features
        lstm_state = lstm_encoder(visual_feat)  # (B, lstm_hidden_dim)
        
        if base.dim() == 3:
            current_base = base[:, -1, :]  # (B, base_dim)
        else:
            current_base = base
        
        # Concatenate LSTM output with current base state
        state = torch.cat([lstm_state, current_base], dim=1)  # (B, lstm_hidden_dim + base_dim)
        
        return state

    def select_action(self, base_seq, depth_seq, noise=True):
        base = torch.as_tensor(base_seq, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(depth_seq, dtype=torch.float32, device=self.device)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() == 4:
            depth = depth.unsqueeze(0)
        if base.dim() == 1:
            base = base.unsqueeze(0)
        elif base.dim() == 2:
            base = base[-1:, :]
        else:
            base = base[:, -1, :]
        
        with torch.no_grad():
            state = self._process_sequence(base, depth, self.actor_visual_encoder, self.actor_lstm)
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
        (base, depth, action, reward,
         next_base, next_depth, done) = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        reward = reward.view(-1, 1)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        done = done.view(-1, 1)

        next_base = torch.as_tensor(next_base, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)

        # Normalize action from buffer to [-1, 1]
        action = (action - self.action_bias_tensor) / self.action_scale_tensor

        # Forward pass for current state (with gradients, using Critic encoders)
        state = self._process_sequence(base, depth, self.critic_visual_encoder, self.critic_lstm, detach_encoder=False)

        with torch.no_grad():
            # Forward pass for next state (no gradients, using Critic Target encoders)
            next_state = self._process_sequence(next_base, next_depth, self.critic_visual_encoder_target, self.critic_lstm_target, detach_encoder=True)

            # Actor Target logic needs next state from Actor Target encoders
            next_state_actor = self._process_sequence(next_base, next_depth, self.actor_visual_encoder_target, self.actor_lstm_target, detach_encoder=True)

            # AE-TD3 Target
            base_next_action = self.actor_target(next_state_actor)
            q_samples = []
            for _ in range(self.adaptive_k):
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                noisy_action = (base_next_action + noise).clamp(-1.0, 1.0)
                q1, q2 = self.critic_target(next_state, noisy_action)
                q_min = torch.min(q1, q2)
                q_samples.append(q_min)
            q_samples = torch.stack(q_samples, dim=1)

        q_mean = q_samples.mean(dim=1)
        q_std = q_samples.std(dim=1)
        meta_input = torch.cat([next_state, q_mean, q_std], dim=1)
        weights = F.softmax(self.meta_net(meta_input), dim=1)
        weighted_q = torch.sum(weights.unsqueeze(-1) * q_samples, dim=1)
        target_Q = reward + (1 - done) * self.discount * weighted_q

        # Critic Update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        weight_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        adaptive_loss = -self.adaptive_reg * weight_entropy
        total_loss = critic_loss + adaptive_loss

        self.critic_optimizer.zero_grad()
        self.meta_optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_visual_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_lstm.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), self.grad_clip)
                
        self.critic_optimizer.step()
        self.meta_optimizer.step()

        # Actor Update
        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            # 1. State from Actor (gradients flow to actor encoders)
            state_actor = self._process_sequence(base, depth, self.actor_visual_encoder, self.actor_lstm, detach_encoder=False)
            
            # 2. State for Critic (detached/no_grad, consistent with critic update input)
            with torch.no_grad():
                state_critic = self._process_sequence(base, depth, self.critic_visual_encoder, self.critic_lstm, detach_encoder=False)

            q1, _ = self.critic(state_critic, self.actor(state_actor))
            actor_loss = -q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_visual_encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_lstm.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Soft Updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_visual_encoder.parameters(), self.critic_visual_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_lstm.parameters(), self.critic_lstm_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_visual_encoder.parameters(), self.actor_visual_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_lstm.parameters(), self.actor_lstm_target.parameters()):
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
            'actor_lstm': self.actor_lstm.state_dict(),
            'actor_lstm_target': self.actor_lstm_target.state_dict(),
            'critic_visual_encoder': self.critic_visual_encoder.state_dict(),
            'critic_visual_encoder_target': self.critic_visual_encoder_target.state_dict(),
            'critic_lstm': self.critic_lstm.state_dict(),
            'critic_lstm_target': self.critic_lstm_target.state_dict(),
            'meta_net': self.meta_net.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
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
            self.actor_lstm.load_state_dict(checkpoint['actor_lstm'])
            self.actor_lstm_target.load_state_dict(checkpoint['actor_lstm_target'])
            self.critic_visual_encoder.load_state_dict(checkpoint['critic_visual_encoder'])
            self.critic_visual_encoder_target.load_state_dict(checkpoint['critic_visual_encoder_target'])
            self.critic_lstm.load_state_dict(checkpoint['critic_lstm'])
            self.critic_lstm_target.load_state_dict(checkpoint['critic_lstm_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        else:
            self.actor_visual_encoder.load_state_dict(checkpoint['visual_encoder'])
            self.actor_visual_encoder_target.load_state_dict(checkpoint['visual_encoder'])
            self.critic_visual_encoder.load_state_dict(checkpoint['visual_encoder'])
            self.critic_visual_encoder_target.load_state_dict(checkpoint['visual_encoder'])
            
            self.actor_lstm.load_state_dict(checkpoint['lstm'])
            self.actor_lstm_target.load_state_dict(checkpoint['lstm'])
            self.critic_lstm.load_state_dict(checkpoint['lstm'])
            self.critic_lstm_target.load_state_dict(checkpoint['lstm'])
            
        self.meta_net.load_state_dict(checkpoint['meta_net'])
        # meta_optimizer load...
        if 'meta_optimizer' in checkpoint:
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        
        self.total_it = checkpoint['total_it']

def make_agent(env, initial_obs, args, device=None):
    # Assuming initial_obs has structure for shapes
    # But we need shapes from args or env usually
    # Here we infer from initial_obs if available
    base = initial_obs["base"]
    depth = initial_obs["depth"]
    
    return LSTMAETD3Agent(
        base_dim=base.shape[-1],
        depth_shape=depth.shape[-3:], # (1, H, W)
        action_space=env.action_space,
        args=args,
        device=device
    )
