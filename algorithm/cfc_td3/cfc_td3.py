import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from ncps.wirings import AutoNCP

from .networks import Actor, Critic, VisualEncoder, CFCEncoder
from .buffer import SequenceReplayBuffer


class CFCTD3Agent:
    """
    TD3 with CfC (Closed-form Continuous-time Neural Network) for POMDP.
    State: Sequence of k steps.
    Each step: Base State + Visual Features (Depth).
    """

    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"TD3-CFC Agent using device: {self.device}")
        self.rng = np.random.default_rng(seed)

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

        # NCPs Wiring definition
        # cfc_units: total neurons, cfc_motor_units: output neurons
        self.cfc_units = getattr(args, 'cfc_units', 32)
        self.cfc_motor_units = getattr(args, 'cfc_motor_units', 8)
        
        # State dim for Actor/Critic is the number of NCP motor neurons
        self.state_dim = self.cfc_motor_units

        # CRITIC Encoders
        self.critic_visual_encoder = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_visual_encoder_target = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_visual_encoder_target.load_state_dict(self.critic_visual_encoder.state_dict())
        visual_feature_dim = self.critic_visual_encoder.repr_dim

        # CFC Encoder with NCP Wiring
        self.critic_wiring = AutoNCP(self.cfc_units, self.cfc_motor_units)
        self.critic_cfc = CFCEncoder(base_dim=self.base_dim, visual_feature_dim=visual_feature_dim, wiring=self.critic_wiring).to(self.device)
        self.critic_cfc_target = CFCEncoder(base_dim=self.base_dim, visual_feature_dim=visual_feature_dim, wiring=self.critic_wiring).to(self.device)
        self.critic_cfc_target.load_state_dict(self.critic_cfc.state_dict())
        
        # ACTOR Encoders
        self.actor_visual_encoder = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_visual_encoder_target = VisualEncoder(input_height=h, input_width=w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_visual_encoder_target.load_state_dict(self.actor_visual_encoder.state_dict())

        self.actor_wiring = AutoNCP(self.cfc_units, self.cfc_motor_units)
        self.actor_cfc = CFCEncoder(base_dim=self.base_dim, visual_feature_dim=visual_feature_dim, wiring=self.actor_wiring).to(self.device)
        self.actor_cfc_target = CFCEncoder(base_dim=self.base_dim, visual_feature_dim=visual_feature_dim, wiring=self.actor_wiring).to(self.device)
        self.actor_cfc_target.load_state_dict(self.actor_cfc.state_dict())

        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers with separate LR for CFC temporal module
        # CFC usually benefits from a larger learning rate (e.g., 1e-3)
        self.cfc_lr = getattr(args, 'cfc_lr', 1e-3) 
        
        self.actor_optimizer = Adam([
            {'params': self.actor.parameters(), 'lr': args.actor_lr},
            {'params': self.actor_visual_encoder.parameters(), 'lr': args.actor_lr},
            {'params': self.actor_cfc.parameters(), 'lr': self.cfc_lr}
        ])
        
        self.critic_optimizer = Adam([
            {'params': self.critic.parameters(), 'lr': args.critic_lr},
            {'params': self.critic_visual_encoder.parameters(), 'lr': args.critic_lr},
            {'params': self.critic_cfc.parameters(), 'lr': self.cfc_lr}
        ])

        # Collecting all parameters for gradient clipping
        self.actor_params = list(self.actor.parameters()) + list(self.actor_visual_encoder.parameters()) + list(self.actor_cfc.parameters())
        self.critic_params = list(self.critic.parameters()) + list(self.critic_visual_encoder.parameters()) + list(self.critic_cfc.parameters())

        self.replay_buffer = SequenceReplayBuffer(args.buffer_size, self.seq_len, seed=seed)
        
        self.discount = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        self.grad_clip = getattr(args, "grad_clip", 1.0)
        
        self.exploration_noise = args.exploration_noise
        self.exploration_noise_final = getattr(args, "exploration_noise_final", 0.05)


        self.total_it = 0

    def _get_current_noise(self, progress_ratio: float) -> float:
        current_noise = self.exploration_noise * (1 - progress_ratio) + self.exploration_noise_final * progress_ratio
        return current_noise

    def select_action(self, base_seq, depth_seq, noise=True, progress_ratio: float = 0.0):
        with torch.no_grad():
            base = torch.as_tensor(base_seq, dtype=torch.float32, device=self.device)
            depth = torch.as_tensor(depth_seq, dtype=torch.float32, device=self.device)
            if depth.dim() == 4:
                depth = depth.unsqueeze(0)
            if base.dim() == 1:
                base = base.unsqueeze(0)
            elif base.dim() == 2 and depth.dim() == 5 and base.size(0) == depth.size(1):
                base = base[-1:, :]

            # Visual Encoding: Batch process for efficiency
            # CNN is applied independently to each frame, so batch processing
            # is mathematically equivalent to frame-by-frame, but faster
            B, S, C, H, W = depth.shape
            if base.dim() == 2:
                base_for_cfc = base.unsqueeze(1).expand(-1, S, -1)
            else:
                base_for_cfc = base
            depth_reshaped = depth.reshape(B * S, C, H, W)
            depth_features = self.actor_visual_encoder(depth_reshaped)
            depth_features = depth_features.view(B, S, -1)

            # CFC Processing: Frame-by-frame inside CFCEncoder
            # For each t: concat(base_expanded_t, depth_features_t) -> CfC(hx_{t-1}) -> hx_t
            state_repr = self.actor_cfc(base_for_cfc, depth_features)

            # Actor
            action = self.actor(state_repr).cpu().data.numpy().flatten()
            
        if noise:
            current_noise = self._get_current_noise(progress_ratio)
            noise = self.rng.normal(0, current_noise, size=self.action_dim)
            action = action + noise

        action = np.clip(action, -1.0, 1.0)
        return action * self.action_scale + self.action_bias

    def train(self, progress_ratio=0.0):
        self.total_it += 1
        

        # Sample replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {}

        base, depth, action, reward, next_base, next_depth, done_flag = batch
        
        # Convert to tensors
        base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = (action - self.action_bias_tensor) / self.action_scale_tensor
        action = action.clamp(-1.0, 1.0)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_base = torch.as_tensor(next_base, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)
        done_flag = torch.as_tensor(done_flag, dtype=torch.float32, device=self.device)
        not_done = 1.0 - done_flag
        
        # Shapes:
        # depth: (B, Seq_Len, 1, H, W)
        # action, reward, not_done: (B, Dim) are from the *last* step of the sequence

        with torch.no_grad():
            # Target Policy Noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # Target Encoder (Next State)
            B, S, C, H, W = next_depth.shape
            if next_base.dim() == 2:
                next_base_for_cfc = next_base.unsqueeze(1).expand(-1, S, -1)
            else:
                next_base_for_cfc = next_base
            next_depth_reshaped = next_depth.view(B * S, C, H, W)
            next_depth_features = self.actor_visual_encoder_target(next_depth_reshaped)
            next_depth_features = next_depth_features.view(B, S, -1)
            
            next_state_repr = self.actor_cfc_target(next_base_for_cfc, next_depth_features)
            
            # Target Actor
            next_action = (self.actor_target(next_state_repr) + noise).clamp(-1.0, 1.0)

            # Target Critic (Next State, Next Action)
            # Use Critic Target Encoders? Generally we share or sync visual/cfc encoders for actor/critic
            # In init:
            # self.critic_visual_encoder = VisualEncoder(...) 
            # self.critic_cfc = CFCEncoder
            # Wait, in the init code I copied, actor and critic have SEPARATE encoders.
            # So I must use critic target encoders for Q-value calculation.
            
            next_depth_features_c = self.critic_visual_encoder_target(next_depth_reshaped)
            next_depth_features_c = next_depth_features_c.view(B, S, -1)
            next_state_repr_c = self.critic_cfc_target(next_base_for_cfc, next_depth_features_c)
            
            target_Q1, target_Q2 = self.critic_target(next_state_repr_c, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Current Q estimates
        B, S, C, H, W = depth.shape
        if base.dim() == 2:
            base_for_cfc = base.unsqueeze(1).expand(-1, S, -1)
        else:
            base_for_cfc = base
        depth_reshaped = depth.view(B * S, C, H, W)
        
        # Critic Update
        depth_features_c = self.critic_visual_encoder(depth_reshaped)
        depth_features_c = depth_features_c.view(B, S, -1)
        state_repr_c = self.critic_cfc(base_for_cfc, depth_features_c)
        
        current_Q1, current_Q2 = self.critic(state_repr_c, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip)
        self.critic_optimizer.step()

        actor_loss_val = 0
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            
            # Actor Update
            depth_features_a = self.actor_visual_encoder(depth_reshaped)
            depth_features_a = depth_features_a.view(B, S, -1)
            state_repr_a = self.actor_cfc(base_for_cfc, depth_features_a)
            
            q1_pi, _ = self.critic(state_repr_a, self.actor(state_repr_a))
            actor_loss = -q1_pi.mean()
            actor_loss_val = actor_loss.item()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
            self.actor_optimizer.step()

            # Soft update target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_visual_encoder.parameters(), self.critic_visual_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_cfc.parameters(), self.critic_cfc_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor_visual_encoder.parameters(), self.actor_visual_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor_cfc.parameters(), self.actor_cfc_target.parameters()):
                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val
        }

    def save(self, filename):
        torch.save(
            {
                "critic": self.critic.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor": self.actor.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "actor_visual_encoder": self.actor_visual_encoder.state_dict(),
                "actor_cfc": self.actor_cfc.state_dict(),
                "critic_visual_encoder": self.critic_visual_encoder.state_dict(),
                "critic_cfc": self.critic_cfc.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "actor_visual_encoder_target": self.actor_visual_encoder_target.state_dict(),
                "actor_cfc_target": self.actor_cfc_target.state_dict(),
                "critic_visual_encoder_target": self.critic_visual_encoder_target.state_dict(),
                "critic_cfc_target": self.critic_cfc_target.state_dict(),
                "total_it": self.total_it,
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "critic_target" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target"])

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"])

        self.actor_visual_encoder.load_state_dict(checkpoint["actor_visual_encoder"])
        self.actor_cfc.load_state_dict(checkpoint["actor_cfc"])
        if "actor_visual_encoder_target" in checkpoint:
            self.actor_visual_encoder_target.load_state_dict(checkpoint["actor_visual_encoder_target"])
        if "actor_cfc_target" in checkpoint:
            self.actor_cfc_target.load_state_dict(checkpoint["actor_cfc_target"])

        self.critic_visual_encoder.load_state_dict(checkpoint["critic_visual_encoder"])
        self.critic_cfc.load_state_dict(checkpoint["critic_cfc"])
        if "critic_visual_encoder_target" in checkpoint:
            self.critic_visual_encoder_target.load_state_dict(checkpoint["critic_visual_encoder_target"])
        if "critic_cfc_target" in checkpoint:
            self.critic_cfc_target.load_state_dict(checkpoint["critic_cfc_target"])

        if "total_it" in checkpoint:
            self.total_it = checkpoint["total_it"]
