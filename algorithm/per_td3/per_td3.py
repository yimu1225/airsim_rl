import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..td3.networks import Actor, Critic, Encoder
from ..per_buffer import PrioritizedReplayBuffer


class PERTD3Agent:
    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.base_dim = base_dim
        self.depth_shape = depth_shape  # (C, H, W)
        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.max_action_tensor = torch.from_numpy(self.max_action).float().to(self.device)
        self.min_action_tensor = torch.from_numpy(self.min_action).float().to(self.device)

        scale = (self.max_action - self.min_action) / 2.0
        bias = (self.max_action + self.min_action) / 2.0
        self.action_scale = torch.from_numpy(scale).float().to(self.device)
        self.action_bias = torch.from_numpy(bias).float().to(self.device)

        self.grad_clip = getattr(args, "grad_clip", 1.0)

        # Encoder
        C, depth_h, depth_w = depth_shape
        feature_dim = args.feature_dim
        
        self.actor_encoder = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_encoder = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        
        self.actor_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        
        self.critic_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        
        self.state_dim = self.base_dim + self.actor_encoder.repr_dim
        
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_params = list(self.actor.parameters()) + list(self.actor_encoder.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        self.critic_params = list(self.critic.parameters()) + list(self.critic_encoder.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        # Buffer with PER
        self.replay_buffer = PrioritizedReplayBuffer(args.buffer_size)

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        
        self.exploration_noise = args.exploration_noise


        self.total_it = 0

    def _encode(self, depth_batch: torch.Tensor, encoder_net) -> torch.Tensor:
        if depth_batch.dim() == 3:
            # Non-recurrent: (frames, H, W) -> (1, frames, H, W)
            depth_batch = depth_batch.unsqueeze(0)
        return encoder_net(depth_batch)

    def _concat_state(self, base: torch.Tensor, depth: torch.Tensor, encoder_net, detach_encoder: bool = False) -> torch.Tensor:
        depth_features = self._encode(depth, encoder_net)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base, depth_features], dim=1)

    def select_action(self, base_state, depth, noise: bool = True):
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            state = self._concat_state(base_tensor, depth_tensor, self.actor_encoder)
            action = self.actor(state).cpu().numpy().flatten()
        if noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
        action = np.clip(action, -1.0, 1.0)
        real_action = self.action_scale.cpu().numpy() * action + self.action_bias.cpu().numpy()
        return real_action

    def train(self, progress_ratio=0.0):
        self.total_it += 1
        
        if self.replay_buffer.size() < self.batch_size:
            return {}

        beta = 0.4 + (1.0 - 0.4) * progress_ratio
        samples, indices, importance_weights = self.replay_buffer.sample(self.batch_size, beta=beta)

        base_states, depths, actions, rewards, next_base_states, next_depths, dones = zip(*samples)

        base_states = torch.as_tensor(np.array(base_states), dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(np.array(depths), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_base_states = torch.as_tensor(np.array(next_base_states), dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(np.array(next_depths), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(importance_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_state_target = self._concat_state(next_base_states, next_depths, self.actor_encoder_target)
            next_action = (self.actor_target(next_state_target) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state_target, next_action)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        state = self._concat_state(base_states, depths, self.critic_encoder)
        current_q1, current_q2 = self.critic(state, actions)

        td_error1 = (current_q1 - target_q).abs()
        td_error2 = (current_q2 - target_q).abs()
        critic_loss = (weights * (F.mse_loss(current_q1, target_q, reduction='none') + 
                                   F.mse_loss(current_q2, target_q, reduction='none'))).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip)
        self.critic_optimizer.step()

        # Update PER priorities
        new_priorities = (td_error1 + td_error2).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, new_priorities)

        result = {"critic_loss": critic_loss.item()}

        if self.total_it % self.policy_freq == 0:
            state_actor = self._concat_state(base_states, depths, self.actor_encoder)
            
            with torch.no_grad():
                state_critic = self._concat_state(base_states, depths, self.critic_encoder)

            q1, _ = self.critic(state_critic, self.actor(state_actor))
            actor_loss = -q1.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_encoder.parameters(), self.actor_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            result["actor_loss"] = actor_loss.item()

        return result

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
