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

    def __init__(self, base_dim, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"TD3-CFC Agent using device: {self.device}")

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

    def select_action(self, base_seq, depth_seq, noise=True):
        # base_seq: (Seq_Len, Base_Dim)
        # depth_seq: (Seq_Len, 1, H, W)
        with torch.no_grad():
            base = torch.as_tensor(base_seq, dtype=torch.float32, device=self.device).unsqueeze(0)     # (1, Seq_Len, Base_Dim)
            depth = torch.as_tensor(depth_seq, dtype=torch.float32, device=self.device).unsqueeze(0)   # (1, Seq_Len, 1, H, W)

            # Visual Encoding: Batch process for efficiency
            # CNN is applied independently to each frame, so batch processing
            # is mathematically equivalent to frame-by-frame, but faster
            B, S, C, H, W = depth.shape
            depth_reshaped = depth.reshape(B * S, C, H, W)
            depth_features = self.actor_visual_encoder(depth_reshaped)
            depth_features = depth_features.view(B, S, -1)

            # CFC Processing: Frame-by-frame inside CFCEncoder
            # For each t: concat(base_expanded_t, depth_features_t) -> CfC(hx_{t-1}) -> hx_t
            state_repr = self.actor_cfc(base, depth_features)

            # Actor
            action = self.actor(state_repr).cpu().data.numpy().flatten()
            
        if noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
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
        # base: (B, Seq_Len, Base_Dim)
        # depth: (B, Seq_Len, 1, H, W)
        # action, reward, not_done: (B, Dim) are from the *last* step of the sequence

        with torch.no_grad():
            # Target Policy Noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            # Target Encoder (Next State)
            B, S, C, H, W = next_depth.shape
            next_depth_reshaped = next_depth.view(B * S, C, H, W)
            next_depth_features = self.actor_visual_encoder_target(next_depth_reshaped)
            next_depth_features = next_depth_features.view(B, S, -1)
            
            next_state_repr = self.actor_cfc_target(next_base, next_depth_features)
            
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
            next_state_repr_c = self.critic_cfc_target(next_base, next_depth_features_c)
            
            target_Q1, target_Q2 = self.critic_target(next_state_repr_c, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Current Q estimates
        B, S, C, H, W = depth.shape
        depth_reshaped = depth.view(B * S, C, H, W)
        
        # Critic Update
        depth_features_c = self.critic_visual_encoder(depth_reshaped)
        depth_features_c = depth_features_c.view(B, S, -1)
        state_repr_c = self.critic_cfc(base, depth_features_c)
        
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
            state_repr_a = self.actor_cfc(base, depth_features_a)
            
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
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
        # Save encoders
        torch.save(self.actor_visual_encoder.state_dict(), filename + "_actor_vis")
        torch.save(self.actor_cfc.state_dict(), filename + "_actor_cfc")
        
        torch.save(self.critic_visual_encoder.state_dict(), filename + "_critic_vis")
        torch.save(self.critic_cfc.state_dict(), filename + "_critic_cfc")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.actor_visual_encoder.load_state_dict(torch.load(filename + "_actor_vis"))
        self.actor_cfc.load_state_dict(torch.load(filename + "_actor_cfc"))
        self.actor_visual_encoder_target.load_state_dict(self.actor_visual_encoder.state_dict())
        self.actor_cfc_target.load_state_dict(self.actor_cfc.state_dict())
        
        self.critic_visual_encoder.load_state_dict(torch.load(filename + "_critic_vis"))
        self.critic_cfc.load_state_dict(torch.load(filename + "_critic_cfc"))
        self.critic_visual_encoder_target.load_state_dict(self.critic_visual_encoder.state_dict())
        self.critic_cfc_target.load_state_dict(self.critic_cfc.state_dict())
