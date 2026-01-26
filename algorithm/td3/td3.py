import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import Actor, Critic, Encoder
from .buffer import ReplayBuffer
from ..ou_noise import OUNoise


class TD3Agent:
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
        _, depth_h, depth_w = depth_shape
        feature_dim = args.depth_feature_dim
        
        # Split Encoders for Actor and Critic
        self.actor_encoder = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim).to(self.device)
        self.critic_encoder = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim).to(self.device)
        
        # Target Encoders (Soft Update)
        self.actor_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim).to(self.device)
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        
        self.critic_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, feature_dim=feature_dim).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        
        # State dim = base_dim + encoder.repr_dim
        self.state_dim = self.base_dim + self.actor_encoder.repr_dim
        
        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        # Combine Actor + Actor Encoder parameters
        self.actor_params = list(self.actor.parameters()) + list(self.actor_encoder.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        # Combine Critic + Critic Encoder parameters
        self.critic_params = list(self.critic.parameters()) + list(self.critic_encoder.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = ReplayBuffer(args.buffer_size)

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        self.exploration_noise = args.exploration_noise

        # OU Noise initialization - replace Gaussian noise with OU noise
        self.ou_noise = OUNoise(
            size=self.action_dim,
            mu=0.0,
            theta=getattr(args, 'ou_theta', 0.15),
            sigma=getattr(args, 'ou_sigma', 0.2),
            sigma_min=getattr(args, 'ou_sigma_min', 0.01),
            dt=getattr(args, 'ou_dt', 1.0)
        )
        # Reset OU noise to initialize properly
        self.ou_noise.reset()

        self.total_it = 0

    def _encode(self, depth_batch: torch.Tensor, encoder_net) -> torch.Tensor:
        if depth_batch.dim() == 3:
            depth_batch = depth_batch.unsqueeze(1)
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
            # Use Actor Encoder
            state = self._concat_state(base_tensor, depth_tensor, self.actor_encoder)
            # Actor returns normalized action (-1, 1)
            action = self.actor(state).cpu().numpy().flatten()

        if noise:
            # Use OU noise instead of Gaussian noise
            ou_noise = self.ou_noise.sample()
            action = action + ou_noise
        
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

        # Encode current observations (Critic Encoder)
        encoded_depths_critic = self._encode(depths, self.critic_encoder)
        states_critic = torch.cat([base_states, encoded_depths_critic], dim=1)

        with torch.no_grad():
            # Encode next observations (Critic Target Encoder)
            next_encoded_depths_critic = self._encode(next_depths, self.critic_encoder_target)
            next_states_critic = torch.cat([next_base_states, next_encoded_depths_critic], dim=1)
            
            # Encode next observations (Actor Target Encoder) for Action Selection
            next_encoded_depths_actor = self._encode(next_depths, self.actor_encoder_target)
            next_states_actor = torch.cat([next_base_states, next_encoded_depths_actor], dim=1)
            
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # Target actor returns normalized action (-1, 1)
            next_actions = (self.actor_target(next_states_actor) + noise).clamp(-1.0, 1.0)

            target_Q1, target_Q2 = self.critic_target(next_states_critic, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(states_critic, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize Critic (and Critic Encoder)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()

        actor_loss = 0.0
        if self.total_it % self.policy_freq == 0:
            # Encode current observations (Actor Encoder)
            encoded_depths_actor = self._encode(depths, self.actor_encoder)
            states_actor = torch.cat([base_states, encoded_depths_actor], dim=1)
            
            # We need Q value for actor loss. 
            # Standard TD3: Actor optimizes Q(s, pi(s)).
            # But which Critic Encoder to use? The Critic's.
            # However, gradients must flow through Actor -> Actor Encoder.
            # And Actor -> Q -> Critic -> Critic Encoder?
            # Typically, we freeze Critic for Actor update.
            # So: state_actor -> Actor -> action
            #     state_critic -> Critic(action) -> Q
            # But state_critic depends on Critic Encoder. state_actor depends on Actor Encoder.
            # Ideally: q = critic(concat(base, critic_encoder(depth)), actor(concat(base, actor_encoder(depth))))
            # We want to optimize Actor parameters (including actor_encoder).
            # Critic parameters (including critic_encoder) are fixed.
            
            # Re-compute critic state features detached (or use critic encoder in eval mode / no grad)
            # Actually, `self.critic` is used. We just need to pass the same depth to critic encoder.
            # BUT, we want gradients to flow from Q to Action to Actor to ActorEncoder.
            # We DO NOT want gradients to flow into Critic Encoder here (it's fixed for actor update).
            
            with torch.no_grad():
                 encoded_depths_critic_fixed = self.critic_encoder(depths)
                 states_critic_fixed = torch.cat([base_states, encoded_depths_critic_fixed], dim=1)

            q1, _ = self.critic(states_critic_fixed, self.actor(states_actor))
            actor_loss = -q1.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=self.grad_clip)
            self.actor_optimizer.step()

            # Soft update targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_encoder.parameters(), self.actor_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
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
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_encoder": self.actor_encoder.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "actor_encoder_target": self.actor_encoder_target.state_dict(),
                "critic_encoder_target": self.critic_encoder_target.state_dict(),
                "total_it": self.total_it,
            },
            filename,
        )

    def load(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"])
        if "critic_target" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        # Backward compatibility or new structure loading
        if "actor_encoder" in checkpoint:
            self.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
            self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
            self.actor_encoder_target.load_state_dict(checkpoint["actor_encoder_target"])
            self.critic_encoder_target.load_state_dict(checkpoint["critic_encoder_target"])
        elif "encoder" in checkpoint:
             # If loading old model with shared encoder, load key 'encoder' to both
             self.actor_encoder.load_state_dict(checkpoint["encoder"])
             self.critic_encoder.load_state_dict(checkpoint["encoder"])
             self.actor_encoder_target.load_state_dict(checkpoint["encoder"])
             self.critic_encoder_target.load_state_dict(checkpoint["encoder"])
        self.total_it = checkpoint.get("total_it", 0)


def make_agent(env, initial_obs, args, device=None) -> TD3Agent:
    base_state = initial_obs["base"]
    depth = initial_obs["depth"]
    agent = TD3Agent(
        base_dim=base_state.shape[0],
        depth_shape=depth.shape,
        action_space=env.action_space,
        args=args,
        device=device,
    )
    return agent
