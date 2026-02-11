import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import Actor, Critic, VMambaVisualEncoder
# Use the buffer from existing algorithm if possible, or copy it.
# We will use the same SequenceReplayBuffer as it is generic.
from ..lstm_td3.buffer import SequenceReplayBuffer


class VMambaTD3Agent:
    """
    TD3 with VMamba (Visual) and Mamba (Sequence) for POMDP.
    State: Sequence of k steps.
    Each step: Base State + Visual Features (Depth).
    Visual Features from VMamba Backbone.
    Sequence processing by Mamba Block.
    """

    def __init__(self, base_dim, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"TD3-VMamba Agent using device: {self.device}")

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
        # CRITIC Encoders - 修改为支持4帧输入 (4通道)
        self.critic_visual_encoder = VMambaVisualEncoder(input_height=h, input_width=w, input_channels=4, args=args).to(self.device)
        self.critic_visual_encoder_target = VMambaVisualEncoder(input_height=h, input_width=w, input_channels=4, args=args).to(self.device)
        self.critic_visual_encoder_target.load_state_dict(self.critic_visual_encoder.state_dict())
        
        # 获取实际的视觉特征维度
        critic_visual_dim = self.critic_visual_encoder.feature_dim
        
        # ACTOR Encoders - 修改为支持4帧输入 (4通道)
        self.actor_visual_encoder = VMambaVisualEncoder(input_height=h, input_width=w, input_channels=4, args=args).to(self.device)
        self.actor_visual_encoder_target = VMambaVisualEncoder(input_height=h, input_width=w, input_channels=4, args=args).to(self.device)
        self.actor_visual_encoder_target.load_state_dict(self.actor_visual_encoder.state_dict())

        # 获取实际的视觉特征维度
        actor_visual_dim = self.actor_visual_encoder.feature_dim

        # State dim for Actor/Critic is visual feature + base state
        self.state_dim = critic_visual_dim + self.base_dim

        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers - 修复参数引用
        self.actor_params = list(self.actor.parameters()) + list(self.actor_visual_encoder.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        self.critic_params = list(self.critic.parameters()) + list(self.critic_visual_encoder.parameters())
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

    def _process_stacked_frames(self, base, depth_seq, visual_encoder, detach_encoder=False):
        """
        Process 4-frame stacked images and concatenate with base state.
        Args:
            base: (B, S) - current state
            depth_seq: (B, 4, H, W) - 4-frame stacked depth images
        Returns:
            state: (B, visual_dim + base_dim)
        """
        if depth_seq.dim() == 5 and depth_seq.size(2) == 1:
            depth_seq = depth_seq.squeeze(2)
        B, C, H, W = depth_seq.shape
        
        # Visual Encoding: 4-frame stacked input -> (B, L, C)
        visual_feat = visual_encoder(depth_seq)  # (B, L, feature_dim)
        
        if detach_encoder:
            visual_feat = visual_feat.detach()
            
        visual_pooled = torch.mean(visual_feat, dim=1)
        return torch.cat([visual_pooled, base], dim=-1)

    def select_action(self, base_seq, depth_seq, noise=True):
        # Inputs are numpy arrays of shape (K, ...)
        # 使用最后4帧进行通道堆叠
        current_base = base_seq[-1]  # (S,)
        
        # 取最后4帧，如果不足4帧则重复第一帧
        if len(depth_seq) >= 4:
            selected_frames = depth_seq[-4:]  # 最后4帧
        else:
            # 如果帧数不足4帧，重复第一帧到4帧
            selected_frames = [depth_seq[0]] * (4 - len(depth_seq)) + list(depth_seq)
        
        # 堆叠4帧为4通道输入 (4, H, W)
        stacked_depth = np.stack(selected_frames, axis=0)
        if stacked_depth.ndim == 4 and stacked_depth.shape[1] == 1:
            stacked_depth = np.squeeze(stacked_depth, axis=1)
        
        # Add batch dimension
        base = torch.as_tensor(current_base, dtype=torch.float32, device=self.device).unsqueeze(0)
        depth = torch.as_tensor(stacked_depth, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 4, H, W)
        
        with torch.no_grad():
            state = self._process_stacked_frames(
                base,
                depth,
                self.actor_visual_encoder
            )
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
        action = torch.as_tensor(action[:, -1, :], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward[:, -1], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done[:, -1], dtype=torch.float32, device=self.device)

        next_base = torch.as_tensor(next_base, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)

        # Normalize action
        action = (action - self.action_bias_tensor) / self.action_scale_tensor

        # 使用最后4帧进行通道堆叠
        current_base = base[:, -1, :]  # (B, S)
        next_current_base = next_base[:, -1, :]  # (B, S)
        
        # 堆叠最后4帧为4通道输入 (B, 4, H, W)
        if depth.shape[1] >= 4:
            current_depth = depth[:, -4:, :, :, :]  # 最后4帧
            next_current_depth = next_depth[:, -4:, :, :, :]  # 最后4帧
        else:
            # 如果帧数不足4帧，重复第一帧到4帧
            repeat_count = 4 - depth.shape[1]
            first_frame = depth[:, 0:1, :, :]  # (B, 1, H, W)
            repeated_frames = first_frame.repeat(1, repeat_count, 1, 1)  # (B, repeat_count, H, W)
            current_depth = torch.cat([repeated_frames, depth], dim=1)  # (B, 4, H, W)
            
            next_first_frame = next_depth[:, 0:1, :, :]
            next_repeated_frames = next_first_frame.repeat(1, repeat_count, 1, 1)
            next_current_depth = torch.cat([next_repeated_frames, next_depth], dim=1)
        
        # 统一为 (B, 4, H, W)
        if current_depth.dim() == 5 and current_depth.size(2) == 1:
            current_depth = current_depth.squeeze(2)
        if next_current_depth.dim() == 5 and next_current_depth.size(2) == 1:
            next_current_depth = next_current_depth.squeeze(2)

        # Critic Update
        state = self._process_stacked_frames(
            current_base,
            current_depth,
            self.critic_visual_encoder,
            detach_encoder=False
        )

        with torch.no_grad():
            next_state = self._process_stacked_frames(
                next_current_base,
                next_current_depth,
                self.critic_visual_encoder_target,
                detach_encoder=True
            )
            next_state_actor = self._process_stacked_frames(
                next_current_base,
                next_current_depth,
                self.actor_visual_encoder_target,
                detach_encoder=True
            )
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
        torch.nn.utils.clip_grad_norm_(self.critic_visual_encoder.parameters(), self.grad_clip)

        self.critic_optimizer.step()

        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            state_actor = self._process_stacked_frames(
                current_base,
                current_depth,
                self.actor_visual_encoder,
                detach_encoder=False
            )
            
            with torch.no_grad():
                state_critic = self._process_stacked_frames(
                    current_base,
                    current_depth,
                    self.critic_visual_encoder,
                    detach_encoder=False
                )

            q1, _ = self.critic(state_critic, self.actor(state_actor))
            actor_loss = -q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_visual_encoder.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Target network updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_visual_encoder.parameters(), self.critic_visual_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_visual_encoder.parameters(), self.actor_visual_encoder_target.parameters()):
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
            
            'critic_visual_encoder': self.critic_visual_encoder.state_dict(),
            'critic_visual_encoder_target': self.critic_visual_encoder_target.state_dict(),
            
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
            
            self.critic_visual_encoder.load_state_dict(checkpoint['critic_visual_encoder'])
            self.critic_visual_encoder_target.load_state_dict(checkpoint['critic_visual_encoder_target'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        else:
            print("Legacy checkpoint loading not fully supported for VMambaTD3...")
            # Implement if needed
            
        self.total_it = checkpoint.get('total_it', 0)

# Factory/Helper function if needed by external scripts
def make_agent(env, initial_obs, args, device=None):
    base = initial_obs["base"]
    depth = initial_obs["depth"]
    return VMambaTD3Agent(
        base_dim=base.shape[-1],
        depth_shape=depth.shape[-3:],
        action_space=env.action_space,
        args=args,
        device=device
    )
