import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..state_adapter import StateAdapter
from ..config_loader import get_algo_param
from .buffer import DualPrioritizedReplayBuffer
from .networks import Actor, Critic, Encoder


class PERTD3Agent:
    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
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

        self.state_dim = self.base_feature_dim + self.actor_encoder.repr_dim

        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_params = list(self.actor.parameters()) + list(self.actor_encoder.parameters()) + list(self.actor_base_adapter.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)

        self.critic_params = list(self.critic.parameters()) + list(self.critic_encoder.parameters()) + list(self.critic_base_adapter.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = DualPrioritizedReplayBuffer(
            capacity=args.buffer_size,
            success_capacity_ratio=get_algo_param(args, "per_td3_success_capacity_ratio", 0.3),
            success_sample_ratio=float(get_algo_param(args, "per_td3_mu_low", 0.15)),
            alpha=get_algo_param(args, "per_td3_alpha", 0.6),
            eps=get_algo_param(args, "per_td3_priority_eps", 1e-6),
            seed=seed,
        )

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size

        self.per_beta_start = get_algo_param(args, "per_td3_beta_start", 0.4)
        self.per_beta_final = get_algo_param(args, "per_td3_beta_final", 1.0)

        # Staircase schedule for success-prioritized sampling ratio mu.
        self.mu_low = float(get_algo_param(args, "per_td3_mu_low", 0.15))
        self.mu_mid = float(get_algo_param(args, "per_td3_mu_mid", 0.30))
        self.mu_high = float(get_algo_param(args, "per_td3_mu_high", 0.45))
        self.mu_step1 = float(get_algo_param(args, "per_td3_mu_step1", 0.25))
        self.mu_step2 = float(get_algo_param(args, "per_td3_mu_step2", 0.65))

        self.exploration_noise = args.exploration_noise

        # Optional sync mode for debugging asynchronous CUDA failures.
        self.cuda_sync_debug = bool(getattr(args, "cuda_sync_debug", False))

        self.total_it = 0

    def _sync_cuda(self, stage: str):
        if self.device.type != "cuda":
            return
        if not self.cuda_sync_debug:
            return
        try:
            torch.cuda.synchronize(self.device)
        except RuntimeError as exc:
            raise RuntimeError(f"CUDA failure during PER-TD3 stage '{stage}': {exc}") from exc

    @staticmethod
    def _ensure_finite(tensor: torch.Tensor, name: str):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite tensor detected in PER-TD3: {name}")

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                value = value.mean()
            return float(value.detach().cpu().item())
        return float(value)

    def _encode(self, depth_batch: torch.Tensor, encoder_net) -> torch.Tensor:
        if depth_batch.dim() == 3:
            depth_batch = depth_batch.unsqueeze(0)
        return encoder_net(depth_batch)

    def _concat_state(self, base: torch.Tensor, depth: torch.Tensor, encoder_net, base_adapter, detach_encoder: bool = False) -> torch.Tensor:
        base_features = base_adapter(base)
        depth_features = self._encode(depth, encoder_net)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base_features, depth_features], dim=1)

    def _get_current_noise(self, progress_ratio: float) -> float:
        return max(float(self.exploration_noise), 1e-8)
    def _get_current_beta(self, progress_ratio: float) -> float:
        p = float(np.clip(progress_ratio, 0.0, 1.0))
        return self.per_beta_start * (1.0 - p) + self.per_beta_final * p

    def _get_current_success_sample_ratio(self, progress_ratio: float) -> float:
        p = float(np.clip(progress_ratio, 0.0, 1.0))
        if p < self.mu_step1:
            mu = self.mu_low
        elif p < self.mu_step2:
            mu = self.mu_mid
        else:
            mu = self.mu_high
        return float(np.clip(mu, 0.0, 0.8))

    def select_action(self, base_state, depth, noise: bool = True, progress_ratio: float = 0.0):
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            state = self._concat_state(base_tensor, depth_tensor, self.actor_encoder, self.actor_base_adapter)
            action = self.actor(state).cpu().numpy().flatten()

        if noise:
            current_noise = self._get_current_noise(progress_ratio)
            noise_vec = self.rng.normal(0, current_noise, size=self.action_dim)
            action = action + noise_vec

        action = np.clip(action, -1.0, 1.0)
        real_action = self.action_scale.cpu().numpy() * action + self.action_bias.cpu().numpy()
        return real_action

    def train(self, progress_ratio=0.0):
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return {}

        current_mu = self._get_current_success_sample_ratio(progress_ratio)
        self.replay_buffer.success_sample_ratio = current_mu

        beta = self._get_current_beta(progress_ratio)
        sampled = self.replay_buffer.sample(self.batch_size, beta=beta)
        if sampled is None:
            return {}

        samples, refs, importance_weights, mix_info = sampled
        if isinstance(samples, tuple):
            base_states, depths, actions, rewards, next_base_states, next_depths, dones = samples
        else:
            base_states, depths, actions, rewards, next_base_states, next_depths, dones = zip(*samples)

        base_states = torch.as_tensor(np.asarray(base_states), dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(np.asarray(depths), dtype=torch.float32, device=self.device)
        real_actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        actions = (real_actions - self.action_bias) / self.action_scale
        actions = actions.clamp(-1.0, 1.0)

        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(np.asarray(next_base_states), dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(np.asarray(next_depths), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=self.device).view(-1, 1)
        weights = torch.as_tensor(importance_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        encoded_depths_critic = self._encode(depths, self.critic_encoder)
        base_features_critic = self.critic_base_adapter(base_states)
        states_critic = torch.cat([base_features_critic, encoded_depths_critic], dim=1)

        with torch.no_grad():
            next_encoded_depths_critic = self._encode(next_depths, self.critic_encoder_target)
            next_base_features_critic = self.critic_base_adapter_target(next_base_states)
            next_states_critic = torch.cat([next_base_features_critic, next_encoded_depths_critic], dim=1)

            next_encoded_depths_actor = self._encode(next_depths, self.actor_encoder_target)
            next_base_features_actor = self.actor_base_adapter_target(next_base_states)
            next_states_actor = torch.cat([next_base_features_actor, next_encoded_depths_actor], dim=1)

            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states_actor) + noise).clamp(-1.0, 1.0)

            target_Q1, target_Q2 = self.critic_target(next_states_critic, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(states_critic, actions)

        td_error = 0.5 * ((current_Q1 - target_Q).abs() + (current_Q2 - target_Q).abs())
        critic_loss = (
            weights
            * (
                F.mse_loss(current_Q1, target_Q, reduction="none")
                + F.mse_loss(current_Q2, target_Q, reduction="none")
            )
        ).mean()

        self._ensure_finite(target_Q, "target_Q")
        self._ensure_finite(current_Q1, "current_Q1")
        self._ensure_finite(current_Q2, "current_Q2")
        self._ensure_finite(critic_loss, "critic_loss")
        self._ensure_finite(td_error, "td_error")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()
        self._sync_cuda("critic_step")

        new_priorities = td_error.detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(refs, new_priorities)

        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:
            encoded_depths_actor = self._encode(depths, self.actor_encoder)
            base_features_actor = self.actor_base_adapter(base_states)
            states_actor = torch.cat([base_features_actor, encoded_depths_actor], dim=1)

            with torch.no_grad():
                encoded_depths_critic_fixed = self.critic_encoder(depths)
                base_features_critic_fixed = self.critic_base_adapter(base_states)
                states_critic_fixed = torch.cat([base_features_critic_fixed, encoded_depths_critic_fixed], dim=1)

            q1, _ = self.critic(states_critic_fixed, self.actor(states_actor))
            actor_loss = -q1.mean()
            self._ensure_finite(actor_loss, "actor_loss")
            actor_loss_value = float(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=self.grad_clip)
            self.actor_optimizer.step()
            self._sync_cuda("actor_step")

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

        self._sync_cuda("train_metrics")

        result = {
            "critic_loss": float(critic_loss.item()),
            "per_beta": float(beta),
            "replay/success_sample_ratio_target": float(current_mu),
            "replay/success_batch_fraction": mix_info["batch_success_fraction"],
            "replay/success_size": float(mix_info["success_size"]),
            "replay/regular_size": float(mix_info["regular_size"]),
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
