import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..aetd3.networks import Actor, Critic, Encoder, MetaNet
from ..per_buffer import PrioritizedReplayBuffer


class PERAETD3Agent:
    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.base_dim = base_dim
        self.depth_shape = depth_shape
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
        self.actor_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        
        self.critic_encoder = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())

        self.state_dim = self.base_dim + self.actor_encoder.repr_dim

        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Adaptive ensemble
        self.adaptive_k = int(args.adaptive_k)
        self.adaptive_reg = args.adaptive_reg
        meta_input_dim = self.state_dim + 2
        self.meta_net = MetaNet(meta_input_dim, self.adaptive_k).to(self.device)

        self.actor_params = list(self.actor.parameters()) + list(self.actor_encoder.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        self.critic_params = list(self.critic.parameters()) + list(self.critic_encoder.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)
        
        self.meta_optimizer = Adam(self.meta_net.parameters(), lr=args.adaptive_meta_lr)

        self.discount = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        
        self.exploration_noise = args.exploration_noise


        self.replay_buffer = PrioritizedReplayBuffer(args.buffer_size)
        self.total_it = 0

    def select_action(self, base_state, depth, noise: bool = True):
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            depth_features = self.actor_encoder(depth_tensor.unsqueeze(0) if depth_tensor.dim()==3 else depth_tensor)
            state = torch.cat([base_tensor, depth_features], dim=1)
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
            next_feat_actor = self.actor_encoder_target(next_depths)
            next_state_actor = torch.cat([next_base_states, next_feat_actor], dim=1)
            
            next_feat_critic = self.critic_encoder_target(next_depths)
            next_state_critic = torch.cat([next_base_states, next_feat_critic], dim=1)
            
            base_next_action = self.actor_target(next_state_actor)
            q_samples = []
            for _ in range(self.adaptive_k):
                noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                noisy_action = (base_next_action + noise).clamp(-1.0, 1.0)
                q1, q2 = self.critic_target(next_state_critic, noisy_action)
                q_samples.append(torch.min(q1, q2))
            
            q_samples = torch.stack(q_samples, dim=1)
            q_mean = q_samples.mean(dim=1)
            q_std = q_samples.std(dim=1)
            
            meta_input = torch.cat([next_state_critic, q_mean, q_std], dim=1)
            adaptive_weights = F.softmax(self.meta_net(meta_input), dim=1)
            
            target_q = torch.sum(adaptive_weights.unsqueeze(-1) * q_samples, dim=1)
            target_q = rewards + (1 - dones) * self.discount * target_q

        feat = self.critic_encoder(depths)
        state = torch.cat([base_states, feat], dim=1)
        current_Q1, current_Q2 = self.critic(state, actions)
        
        td_errors = (torch.abs(current_Q1 - target_q) + torch.abs(current_Q2 - target_q)) / 2.0
        
        critic_loss = (weights * (F.mse_loss(current_Q1, target_q, reduction='none') + 
                                  F.mse_loss(current_Q2, target_q, reduction='none'))).mean()
        
        if self.adaptive_reg > 0:
            # Add entropy regularization for meta-weights
            entropy = -(adaptive_weights * torch.log(adaptive_weights + 1e-8)).sum(dim=1).mean()
            critic_loss -= self.adaptive_reg * entropy

        self.critic_optimizer.zero_grad()
        self.meta_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        self.meta_optimizer.step()

        # Update PER priorities
        new_priorities = td_errors.detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, new_priorities)

        result = {"critic_loss": critic_loss.item()}

        if self.total_it % self.policy_freq == 0:
            feat_actor = self.actor_encoder(depths)
            state_actor = torch.cat([base_states, feat_actor], dim=1)
            
            with torch.no_grad():
                feat_critic_fixed = self.critic_encoder(depths)
                state_critic_fixed = torch.cat([base_states, feat_critic_fixed], dim=1)
                
            q1, _ = self.critic(state_critic_fixed, self.actor(state_actor))
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
            'meta_net': self.meta_net.state_dict(),
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.meta_net.load_state_dict(checkpoint['meta_net'])
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
