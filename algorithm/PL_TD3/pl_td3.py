import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..state_adapter import StateAdapter
from .networks import Actor, Critic, Encoder
from .buffer import ReplayBuffer


class PLTD3Agent:
    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
        self.critic_priv_dim = int(
            getattr(
                args,
                "critic_priv_dim",
                getattr(args, "distance_sensor_count", 108),
            )
        )
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

        C, depth_h, depth_w = depth_shape

        self.actor_encoder = Encoder(input_height=depth_h, input_width=depth_w, input_channels=C).to(self.device)
        self.critic_encoder = Encoder(input_height=depth_h, input_width=depth_w, input_channels=C).to(self.device)

        self.actor_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, input_channels=C).to(self.device)
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())

        self.critic_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, input_channels=C).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())

        self.actor_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)

        self.actor_base_adapter_target = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.actor_base_adapter_target.load_state_dict(self.actor_base_adapter.state_dict())

        self.critic_base_adapter_target = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target.load_state_dict(self.critic_base_adapter.state_dict())

        self.actor_state_dim = self.base_feature_dim + self.actor_encoder.repr_dim
        self.critic_state_dim = self.base_feature_dim + self.critic_encoder.repr_dim + self.critic_priv_dim

        self.actor = Actor(self.actor_state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.actor_state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.critic_state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.critic_state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_params = (
            list(self.actor.parameters())
            + list(self.actor_encoder.parameters())
            + list(self.actor_base_adapter.parameters())
        )
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)

        self.critic_params = (
            list(self.critic.parameters())
            + list(self.critic_encoder.parameters())
            + list(self.critic_base_adapter.parameters())
        )
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = ReplayBuffer(args.buffer_size, seed=seed)

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
            depth_batch = depth_batch.unsqueeze(0)
        return encoder_net(depth_batch)

    def _prepare_priv(self, priv_batch: torch.Tensor) -> torch.Tensor:
        if priv_batch.dim() == 1:
            priv_batch = priv_batch.view(1, -1)
        elif priv_batch.dim() > 2:
            priv_batch = priv_batch.view(priv_batch.size(0), -1)

        if priv_batch.size(1) != self.critic_priv_dim:
            if priv_batch.size(1) > self.critic_priv_dim:
                priv_batch = priv_batch[:, : self.critic_priv_dim]
            else:
                pad = torch.zeros(
                    priv_batch.size(0),
                    self.critic_priv_dim - priv_batch.size(1),
                    device=priv_batch.device,
                    dtype=priv_batch.dtype,
                )
                priv_batch = torch.cat([priv_batch, pad], dim=1)
        return priv_batch

    def _concat_actor_state(self, base: torch.Tensor, depth: torch.Tensor, encoder_net, base_adapter, detach_encoder: bool = False) -> torch.Tensor:
        base_features = base_adapter(base)
        depth_features = self._encode(depth, encoder_net)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base_features, depth_features], dim=1)

    def _concat_critic_state(
        self,
        base: torch.Tensor,
        depth: torch.Tensor,
        priv: torch.Tensor,
        encoder_net,
        base_adapter,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        base_features = base_adapter(base)
        depth_features = self._encode(depth, encoder_net)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base_features, depth_features, self._prepare_priv(priv)], dim=1)

    def _get_current_noise(self, progress_ratio: float) -> float:
        return max(float(self.exploration_noise), 1e-8)
    def select_action(self, base_state, depth, noise: bool = True, progress_ratio: float = 0.0):
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            state = self._concat_actor_state(base_tensor, depth_tensor, self.actor_encoder, self.actor_base_adapter)
            action = self.actor(state).cpu().numpy().flatten()

        if noise:
            current_noise = self._get_current_noise(progress_ratio)
            noise = self.rng.normal(0, current_noise, size=self.action_dim)
            action = action + noise

        action = np.clip(action, -1.0, 1.0)
        real_action = self.action_scale.cpu().numpy() * action + self.action_bias.cpu().numpy()
        return real_action

    def train(self, progress_ratio=0.0):
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return {}

        (
            base_states,
            depths,
            actions,
            rewards,
            next_base_states,
            next_depths,
            dones,
            critic_privs,
            next_critic_privs,
        ) = self.replay_buffer.sample(self.batch_size)

        base_states = torch.as_tensor(base_states, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        critic_privs = torch.as_tensor(critic_privs, dtype=torch.float32, device=self.device)

        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = (real_actions - self.action_bias) / self.action_scale

        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_base_states, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depths, dtype=torch.float32, device=self.device)
        next_critic_privs = torch.as_tensor(next_critic_privs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        states_critic = self._concat_critic_state(
            base_states,
            depths,
            critic_privs,
            self.critic_encoder,
            self.critic_base_adapter,
        )

        with torch.no_grad():
            next_states_critic = self._concat_critic_state(
                next_base_states,
                next_depths,
                next_critic_privs,
                self.critic_encoder_target,
                self.critic_base_adapter_target,
            )

            next_states_actor = self._concat_actor_state(
                next_base_states,
                next_depths,
                self.actor_encoder_target,
                self.actor_base_adapter_target,
            )

            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states_actor) + noise).clamp(-1.0, 1.0)

            target_Q1, target_Q2 = self.critic_target(next_states_critic, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(states_critic, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()

        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:
            states_actor = self._concat_actor_state(
                base_states,
                depths,
                self.actor_encoder,
                self.actor_base_adapter,
            )

            with torch.no_grad():
                states_critic_fixed = self._concat_critic_state(
                    base_states,
                    depths,
                    critic_privs,
                    self.critic_encoder,
                    self.critic_base_adapter,
                )

            q1, _ = self.critic(states_critic_fixed, self.actor(states_actor))
            actor_loss = -q1.mean()
            actor_loss_value = float(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=self.grad_clip)
            self.actor_optimizer.step()

            # Soft update targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_encoder.parameters(), self.actor_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_base_adapter.parameters(), self.actor_base_adapter_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_base_adapter.parameters(), self.critic_base_adapter_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        result = {
            "critic_loss": float(critic_loss.item()),
        }
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        return result

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
                "actor_base_adapter": self.actor_base_adapter.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic_base_adapter": self.critic_base_adapter.state_dict(),
                "actor_encoder_target": self.actor_encoder_target.state_dict(),
                "actor_base_adapter_target": self.actor_base_adapter_target.state_dict(),
                "critic_encoder_target": self.critic_encoder_target.state_dict(),
                "critic_base_adapter_target": self.critic_base_adapter_target.state_dict(),
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
            if "actor_base_adapter" in checkpoint:
                self.actor_base_adapter.load_state_dict(checkpoint["actor_base_adapter"])
            if "critic_base_adapter" in checkpoint:
                self.critic_base_adapter.load_state_dict(checkpoint["critic_base_adapter"])
            if "actor_base_adapter_target" in checkpoint:
                self.actor_base_adapter_target.load_state_dict(checkpoint["actor_base_adapter_target"])
            if "critic_base_adapter_target" in checkpoint:
                self.critic_base_adapter_target.load_state_dict(checkpoint["critic_base_adapter_target"])
        elif "encoder" in checkpoint:
            self.actor_encoder.load_state_dict(checkpoint["encoder"])
            self.critic_encoder.load_state_dict(checkpoint["encoder"])
            self.actor_encoder_target.load_state_dict(checkpoint["encoder"])
            self.critic_encoder_target.load_state_dict(checkpoint["encoder"])
        self.total_it = checkpoint.get("total_it", 0)


def make_agent(env, initial_obs, args, device=None) -> PLTD3Agent:
    base_state = initial_obs["base"]
    depth = initial_obs["depth"]
    agent = PLTD3Agent(
        base_dim=base_state.shape[0],
        depth_shape=depth.shape,
        action_space=env.action_space,
        args=args,
        device=device,
    )
    return agent
