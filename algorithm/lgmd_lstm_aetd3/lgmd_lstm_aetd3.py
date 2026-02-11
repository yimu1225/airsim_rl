import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import Actor, Critic, DepthEncoder, MotionEncoder, LSTMEncoder, MetaNet, D_LGMD,  BaseStateExpander
from .buffer import SequenceReplayBuffer


class D_LGMDLSTMAETD3Agent:
    """
    AETD3 with LSTM and D-LGMD for POMDP.
    State: Sequence of k steps.
    Each step: Base State + Visual Features (Depth + Motion from D-LGMD).
    Motion features are processed separately and concatenated with LSTM output.
    """

    def __init__(self, base_dim, depth_shape, gray_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"AETD3-LSTM-D-LGMD Agent using device: {self.device}")

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
        
        # Actor Encoders
        self.actor_depth_encoder = DepthEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.actor_depth_encoder_target = DepthEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.actor_depth_encoder_target.load_state_dict(self.actor_depth_encoder.state_dict())
        
        self.actor_motion_encoder = MotionEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.actor_motion_encoder_target = MotionEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.actor_motion_encoder_target.load_state_dict(self.actor_motion_encoder.state_dict())
        
        # Base State Expanders (separate for actor/critic)
        expanded_base_dim = 32
        self.actor_base_expander = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.actor_base_expander_target = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.actor_base_expander_target.load_state_dict(self.actor_base_expander.state_dict())

        self.critic_base_expander = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.critic_base_expander_target = BaseStateExpander(self.base_dim, expanded_dim=expanded_base_dim).to(self.device)
        self.critic_base_expander_target.load_state_dict(self.critic_base_expander.state_dict())
        
        # Critic Encoders
        self.critic_depth_encoder = DepthEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.critic_depth_encoder_target = DepthEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.critic_depth_encoder_target.load_state_dict(self.critic_depth_encoder.state_dict())
        
        self.critic_motion_encoder = MotionEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.critic_motion_encoder_target = MotionEncoder(input_height=h, input_width=w, feature_dim=feature_dim).to(self.device)
        self.critic_motion_encoder_target.load_state_dict(self.critic_motion_encoder.state_dict())
        
        # Feature dim per step
        self.step_feature_dim = self.base_dim + self.actor_depth_encoder.repr_dim + self.actor_motion_encoder.repr_dim
        
        # LSTM Encoder
        self.lstm_hidden_dim = args.lstm_hidden_dim
        lstm_layers = getattr(args, 'lstm_num_layers', 1)
        
        # Actor LSTM
        self.actor_lstm = LSTMEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.actor_depth_encoder.repr_dim, hidden_dim=self.lstm_hidden_dim, num_layers=lstm_layers).to(self.device)
        self.actor_lstm_target = LSTMEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.actor_depth_encoder.repr_dim, hidden_dim=self.lstm_hidden_dim, num_layers=lstm_layers).to(self.device)
        self.actor_lstm_target.load_state_dict(self.actor_lstm.state_dict())
        
        # Critic LSTM
        self.critic_lstm = LSTMEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.critic_depth_encoder.repr_dim, hidden_dim=self.lstm_hidden_dim, num_layers=lstm_layers).to(self.device)
        self.critic_lstm_target = LSTMEncoder(expanded_base_dim=expanded_base_dim, depth_feature_dim=self.critic_depth_encoder.repr_dim, hidden_dim=self.lstm_hidden_dim, num_layers=lstm_layers).to(self.device)
        self.critic_lstm_target.load_state_dict(self.critic_lstm.state_dict())
        
        # State dim for Actor/Critic is LSTM output + motion features
        self.state_dim = self.lstm_hidden_dim + self.actor_motion_encoder.repr_dim

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

        # Optimizers (D-LGMD has fixed weights, no optimizer needed)
        self.actor_optimizer = Adam(
            list(self.actor.parameters()) + list(self.actor_depth_encoder.parameters()) + list(self.actor_motion_encoder.parameters()) + list(self.actor_base_expander.parameters()) + list(self.actor_lstm.parameters()), 
            lr=args.actor_lr
        )
        self.critic_optimizer = Adam(
            list(self.critic.parameters()) + list(self.critic_depth_encoder.parameters()) + list(self.critic_motion_encoder.parameters()) + list(self.critic_base_expander.parameters()) + list(self.critic_lstm.parameters()), 
            lr=args.critic_lr
        )
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

    def _process_sequence(self, base, depth, gray, depth_encoder, motion_encoder, lstm_encoder, base_expander, detach_encoder=False):
        """
        Process sequence with D-LGMD and LSTM.
        Args:
            base: (B, K, D_base)
            depth: (B, K, 1, H, W)
            gray: (B, K, M, H, W) - for D-LGMD motion processing
        Returns:
            state: (B, lstm_hidden_dim + motion_repr_dim)
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
        motion_last_flat = motion_last.view(B * 1, 1, H, W)
        
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
            
        # LSTM Processing - only processes depth features
        lstm_state = lstm_encoder(base_expanded, depth_feat)  # (B, lstm_hidden_dim)
        
        # Concatenate LSTM output with motion features
        state = torch.cat([lstm_state, motion_feat], dim=1)  # (B, lstm_hidden_dim + motion_repr_dim)
        
        return state

    def select_action(self, base_seq, depth_seq, gray_seq, noise=True):
        # Inputs are numpy arrays of shape (K, ...)
        # Add batch dimension
        base = torch.as_tensor(base_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        depth = torch.as_tensor(depth_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        gray = torch.as_tensor(gray_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            state = self._process_sequence(base, depth, gray, self.actor_depth_encoder, self.actor_motion_encoder, self.actor_lstm, self.actor_base_expander)
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
        state = self._process_sequence(base, depth, gray, self.critic_depth_encoder, self.critic_motion_encoder, self.critic_lstm, self.critic_base_expander, detach_encoder=False)

        with torch.no_grad():
            # Critic Target Features
            next_state = self._process_sequence(next_base, next_depth, next_gray, self.critic_depth_encoder_target, self.critic_motion_encoder_target, self.critic_lstm_target, self.critic_base_expander_target, detach_encoder=True)
            
            # Actor Target Features
            next_state_actor = self._process_sequence(next_base, next_depth, next_gray, self.actor_depth_encoder_target, self.actor_motion_encoder_target, self.actor_lstm_target, self.actor_base_expander_target, detach_encoder=True)
            
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
        torch.nn.utils.clip_grad_norm_(self.critic_depth_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_motion_encoder.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_base_expander.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_lstm.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), self.grad_clip)
                
        self.critic_optimizer.step()
        self.meta_optimizer.step()

        # Actor Update
        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            # Actor Feature Extraction
            state_actor = self._process_sequence(base, depth, gray, self.actor_depth_encoder, self.actor_motion_encoder, self.actor_lstm, self.actor_base_expander, detach_encoder=False)
            
            with torch.no_grad():
                state_critic = self._process_sequence(base, depth, gray, self.critic_depth_encoder, self.critic_motion_encoder, self.critic_lstm, self.critic_base_expander, detach_encoder=False)

            q1, _ = self.critic(state_critic, self.actor(state_actor))
            actor_loss = -q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_depth_encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_motion_encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_base_expander.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_lstm.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_depth_encoder.parameters(), self.critic_depth_encoder_target.parameters() if hasattr(self, 'critic_depth_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_motion_encoder.parameters(), self.critic_motion_encoder_target.parameters() if hasattr(self, 'critic_motion_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_base_expander.parameters(), self.critic_base_expander_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_lstm.parameters(), self.critic_lstm_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_depth_encoder.parameters(), self.actor_depth_encoder_target.parameters() if hasattr(self, 'actor_depth_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_motion_encoder.parameters(), self.actor_motion_encoder_target.parameters() if hasattr(self, 'actor_motion_encoder_target') else []):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_base_expander.parameters(), self.actor_base_expander_target.parameters()):
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
            
            'actor_depth_encoder': self.actor_depth_encoder.state_dict(),
            'actor_depth_encoder_target': self.actor_depth_encoder_target.state_dict(),
            'actor_motion_encoder': self.actor_motion_encoder.state_dict(),
            'actor_motion_encoder_target': self.actor_motion_encoder_target.state_dict(),
            'actor_base_expander': self.actor_base_expander.state_dict(),
            'actor_base_expander_target': self.actor_base_expander_target.state_dict(),
            'actor_lstm': self.actor_lstm.state_dict(),
            'actor_lstm_target': self.actor_lstm_target.state_dict(),
            
            'critic_depth_encoder': self.critic_depth_encoder.state_dict(),
            'critic_depth_encoder_target': self.critic_depth_encoder_target.state_dict(),
            'critic_motion_encoder': self.critic_motion_encoder.state_dict(),
            'critic_motion_encoder_target': self.critic_motion_encoder_target.state_dict(),
            'critic_base_expander': self.critic_base_expander.state_dict(),
            'critic_base_expander_target': self.critic_base_expander_target.state_dict(),
            'critic_lstm': self.critic_lstm.state_dict(),
            'critic_lstm_target': self.critic_lstm_target.state_dict(),
            
            'd_lgmd': self.d_lgmd.state_dict(),
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
        
        self.d_lgmd.load_state_dict(checkpoint['d_lgmd'])
        self.meta_net.load_state_dict(checkpoint['meta_net'])
        
        if 'actor_depth_encoder' in checkpoint:
            self.actor_depth_encoder.load_state_dict(checkpoint['actor_depth_encoder'])
            self.actor_depth_encoder_target.load_state_dict(checkpoint['actor_depth_encoder_target'])
            self.actor_motion_encoder.load_state_dict(checkpoint['actor_motion_encoder'])
            self.actor_motion_encoder_target.load_state_dict(checkpoint['actor_motion_encoder_target'])
            self.actor_base_expander.load_state_dict(checkpoint['actor_base_expander'])
            self.actor_base_expander_target.load_state_dict(checkpoint['actor_base_expander_target'])
            self.actor_lstm.load_state_dict(checkpoint['actor_lstm'])
            self.actor_lstm_target.load_state_dict(checkpoint['actor_lstm_target'])
            
            self.critic_depth_encoder.load_state_dict(checkpoint['critic_depth_encoder'])
            self.critic_depth_encoder_target.load_state_dict(checkpoint['critic_depth_encoder_target'])
            self.critic_motion_encoder.load_state_dict(checkpoint['critic_motion_encoder'])
            self.critic_motion_encoder_target.load_state_dict(checkpoint['critic_motion_encoder_target'])
            self.critic_base_expander.load_state_dict(checkpoint['critic_base_expander'])
            self.critic_base_expander_target.load_state_dict(checkpoint['critic_base_expander_target'])
            self.critic_lstm.load_state_dict(checkpoint['critic_lstm'])
            self.critic_lstm_target.load_state_dict(checkpoint['critic_lstm_target'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            if 'meta_optimizer' in checkpoint:
                self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
            
        self.total_it = checkpoint['total_it']

def make_agent(env, initial_obs, args, device=None):
    # Assuming initial_obs has structure for shapes
    # But we need shapes from args or env usually
    # Here we infer from initial_obs if available
    base = initial_obs["base"]
    depth = initial_obs["depth"]
    gray = initial_obs["gray"] # Assuming this exists in obs
    
    return D_LGMDLSTMAETD3Agent(
        base_dim=base.shape[-1],
        depth_shape=depth.shape[-3:], # (1, H, W)
        gray_shape=gray.shape[-3:],   # (M, H, W)
        action_space=env.action_space,
        args=args,
        device=device
    )
