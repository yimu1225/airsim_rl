import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import Actor, Critic, Encoder, MetaNet
from .buffer import ReplayBuffer


class AETD3Agent:
    """Adaptive Ensemble TD3 (depth features only)."""

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
        
        # ACTOR Encoders
        self.actor_encoder = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        
        # CRITIC Encoders
        self.critic_encoder = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim, input_channels=C).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())

        self.state_dim = self.base_dim + self.actor_encoder.repr_dim

        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Adaptive ensemble
        self.adaptive_k = int(args.adaptive_k)
        self.adaptive_reg = args.adaptive_reg
        meta_input_dim = self.state_dim + 2  # state + Q_mean + Q_std
        self.meta_net = MetaNet(meta_input_dim, self.adaptive_k).to(self.device)

        # Optimizers (Merged params)
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

        
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.total_it = 0

    def _encode(self, depth_batch: torch.Tensor, encoder) -> torch.Tensor:
        if depth_batch.dim() == 3:
            # Non-recurrent: (frames, H, W) -> (1, frames, H, W)
            depth_batch = depth_batch.unsqueeze(0)
        return encoder(depth_batch)

    def _concat_state(self, base: torch.Tensor, depth: torch.Tensor, encoder, detach_encoder: bool = False) -> torch.Tensor:
        depth_features = self._encode(depth, encoder)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base, depth_features], dim=1)

    def select_action(self, base_state, depth, noise: bool = True):
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            state = self._concat_state(base_tensor, depth_tensor, self.actor_encoder)
            # Actor returns normalized action (-1, 1)
            action = self.actor(state).cpu().numpy().flatten()

        if noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
        
        # Clip to (-1, 1)
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to real action space
        real_action = self.action_scale.cpu().numpy() * action + self.action_bias.cpu().numpy()
        return real_action

    def train(self, progress_ratio=0.0):
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return

        base_states, depths, actions, rewards, next_base_states, next_depths, dones = self.replay_buffer.sample(self.batch_size)

        base_states = torch.as_tensor(base_states, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        # Actions from buffer are real actions, normalize them for training
        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = (real_actions - self.action_bias) / self.action_scale
        
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_base_states, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depths, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        # ----------------------------
        # CRITIC UPDATE
        # ----------------------------
        # Encode current observations
        states = self._concat_state(base_states, depths, self.critic_encoder)

        with torch.no_grad():
            # Encode next observations
            next_states = self._concat_state(next_base_states, next_depths, self.critic_encoder_target)

            # Target actor needs actor target encoder features
            next_states_actor = self._concat_state(next_base_states, next_depths, self.actor_encoder_target)
            
            # Target actor returns normalized action (-1, 1)
            base_next_action = self.actor_target(next_states_actor)
            q_samples = []
            for _ in range(self.adaptive_k):
                noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                noisy_action = (base_next_action + noise).clamp(-1.0, 1.0)
                q1, q2 = self.critic_target(next_states, noisy_action)
                q_min = torch.min(q1, q2)
                q_samples.append(q_min)
            q_samples = torch.stack(q_samples, dim=1)  # (batch, K, 1)

        q_mean = q_samples.mean(dim=1)
        q_std = q_samples.std(dim=1)
        
        # MetaNet input needs next_states (from critic encoder)
        meta_input = torch.cat([next_states, q_mean, q_std], dim=1)
        weights = F.softmax(self.meta_net(meta_input), dim=1)  # (batch, K)
        weighted_q = torch.sum(weights.unsqueeze(-1) * q_samples, dim=1)  # (batch, 1)
        target_Q = rewards + (1 - dones) * self.discount * weighted_q

        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        weight_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        adaptive_loss = -self.adaptive_reg * weight_entropy
        total_critic_loss = critic_loss + adaptive_loss

        self.critic_optimizer.zero_grad()
        self.meta_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_encoder.parameters(), max_norm=self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), max_norm=self.grad_clip)
        self.critic_optimizer.step()
        self.meta_optimizer.step()

        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            # ----------------------------
            # ACTOR UPDATE
            # ----------------------------
            # Encode with Actor Encoder
            states_for_actor = self._concat_state(base_states, depths, self.actor_encoder)
            
            # Get Critic features (detached) for Q valuation
            # We want to optimize Actor and Actor Encoder to maximize Q.
            with torch.no_grad():
                states_for_critic = self._concat_state(base_states, depths, self.critic_encoder)
            
            q1, _ = self.critic(states_for_critic, self.actor(states_for_actor))
            actor_loss = -q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.actor_encoder.parameters(), max_norm=self.grad_clip)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor_encoder.parameters(), self.actor_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 返回损失值用于调试
        return {
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
            'critic_loss': critic_loss.item(),
        }

    def save(self, filename: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_encoder": self.actor_encoder.state_dict(),
                "actor_encoder_target": self.actor_encoder_target.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic_encoder_target": self.critic_encoder_target.state_dict(),
                "meta_net": self.meta_net.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "meta_optimizer": self.meta_optimizer.state_dict(),
                "total_it": self.total_it,
            },
            filename,
        )

    def load(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        
        if "actor_encoder" in checkpoint:
            self.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
            self.actor_encoder_target.load_state_dict(checkpoint["actor_encoder_target"])
            self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
            self.critic_encoder_target.load_state_dict(checkpoint["critic_encoder_target"])
        else:
            self.actor_encoder.load_state_dict(checkpoint["encoder"])
            self.actor_encoder_target.load_state_dict(checkpoint["encoder"])
            self.critic_encoder.load_state_dict(checkpoint["encoder"])
            self.critic_encoder_target.load_state_dict(checkpoint["encoder"])
            
        self.meta_net.load_state_dict(checkpoint["meta_net"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        # meta_optimizer was added later or exists? Base code had it.
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)


def make_agent(env, initial_obs, args, device=None) -> AETD3Agent:
    base_state = initial_obs["base"]
    depth = initial_obs["depth"]
    agent = AETD3Agent(
        base_dim=base_state.shape[0],
        depth_shape=depth.shape,
        action_space=env.action_space,
        args=args,
        device=device,
    )
    return agent
