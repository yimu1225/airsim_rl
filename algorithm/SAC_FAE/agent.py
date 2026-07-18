# SPDX-License-Identifier: GPL-3.0-only
"""SAC agent with the official Focus autoencoder perception module."""

from __future__ import annotations

from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..SAC.buffer import ReplayBuffer
from ..config_loader import get_algo_param
from .networks import Actor, Critic, FocusAutoencoder, FocusEncoder, MeasurementEncoder


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.lerp_(source_param.data, float(tau))
        source_buffers = dict(source.named_buffers())
        for name, target_buffer in target.named_buffers():
            source_buffer = source_buffers[name]
            if target_buffer.is_floating_point():
                target_buffer.data.lerp_(source_buffer.data, float(tau))
            else:
                target_buffer.data.copy_(source_buffer.data)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class SACFAEAgent:
    """SAC_FAE adapted to 4x128x128 depth, 11-D base state, and 3-D actions."""

    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None) -> None:
        self.args = args
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if seed is not None:
            torch.manual_seed(int(seed))

        self.base_dim = int(base_dim)
        self.depth_shape = tuple(int(value) for value in depth_shape)
        if self.depth_shape != (4, 128, 128):
            raise ValueError(
                f"SAC_FAE requires depth_shape=(4, 128, 128), got {self.depth_shape}."
            )

        self.action_dim = int(action_space.shape[0])
        self.min_action = np.asarray(action_space.low, dtype=np.float32)
        self.max_action = np.asarray(action_space.high, dtype=np.float32)
        self.action_scale = torch.as_tensor(
            (self.max_action - self.min_action) / 2.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_bias = torch.as_tensor(
            (self.max_action + self.min_action) / 2.0,
            dtype=torch.float32,
            device=self.device,
        )
        if torch.any(self.action_scale <= 0):
            raise ValueError("SAC_FAE requires every action dimension to have a positive range.")

        feature_dim = int(get_algo_param(args, "sac_fae_feature_dim"))
        measurement_dim = int(get_algo_param(args, "sac_fae_measurement_dim"))
        num_filters = int(get_algo_param(args, "sac_fae_num_filters"))

        self.autoencoder = FocusAutoencoder(self.depth_shape, feature_dim, num_filters).to(self.device)
        self.encoder_target = FocusEncoder(self.depth_shape, feature_dim, num_filters).to(self.device)
        self.encoder_target.load_state_dict(self.autoencoder.encoder.state_dict())
        self.encoder_target.eval()
        self.measurement_encoder = MeasurementEncoder(self.base_dim, measurement_dim).to(self.device)
        self.measurement_encoder_target = MeasurementEncoder(self.base_dim, measurement_dim).to(self.device)
        self.measurement_encoder_target.load_state_dict(self.measurement_encoder.state_dict())

        state_dim = feature_dim + measurement_dim
        self.actor = Actor(
            state_dim,
            self.action_dim,
            hidden_dim=int(get_algo_param(args, "sac_fae_actor_hidden_dim")),
            hidden_layers=int(get_algo_param(args, "sac_fae_actor_hidden_layers")),
            log_std_min=float(get_algo_param(args, "sac_fae_log_std_min")),
            log_std_max=float(get_algo_param(args, "sac_fae_log_std_max")),
        ).to(self.device)
        self.critic = Critic(
            state_dim,
            self.action_dim,
            hidden_dim=int(get_algo_param(args, "sac_fae_critic_hidden_dim")),
            hidden_layers=int(get_algo_param(args, "sac_fae_critic_hidden_layers")),
        ).to(self.device)
        self.critic_target = Critic(
            state_dim,
            self.action_dim,
            hidden_dim=int(get_algo_param(args, "sac_fae_critic_hidden_dim")),
            hidden_layers=int(get_algo_param(args, "sac_fae_critic_hidden_layers")),
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=float(args.actor_lr))
        self.critic_autoencoder_params = list(
            chain(
                self.critic.parameters(),
                self.autoencoder.parameters(),
                self.measurement_encoder.parameters(),
            )
        )
        self.critic_autoencoder_optimizer = Adam(
            self.critic_autoencoder_params,
            lr=float(args.critic_lr),
        )

        self.ent_coef = get_algo_param(args, "ent_coef")
        target_entropy = get_algo_param(args, "target_entropy")
        self.target_entropy = (
            -float(self.action_dim)
            if target_entropy in (None, "auto")
            else float(target_entropy)
        )
        self.log_alpha = None
        self.alpha_optimizer = None
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_", maxsplit=1)[1])
            if init_value <= 0:
                raise ValueError("Automatic entropy coefficient must start above zero.")
            self.log_alpha = torch.log(
                torch.full((1,), init_value, dtype=torch.float32, device=self.device)
            ).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=float(args.actor_lr))
            self.alpha = init_value
            self.auto_entropy_tuning = True
        else:
            self.alpha = float(self.ent_coef)
            self.auto_entropy_tuning = False

        self.replay_buffer = ReplayBuffer(int(args.buffer_size), seed=seed)
        self.batch_size = int(args.batch_size)
        self.gamma = float(args.gamma)
        self.grad_clip = float(getattr(args, "grad_clip", 1.0))
        self.policy_freq = int(get_algo_param(args, "policy_freq"))
        self.target_update_interval = int(get_algo_param(args, "target_update_interval"))
        self.reconstruction_coef = float(
            get_algo_param(args, "sac_fae_reconstruction_coef")
        )
        self.critic_tau = float(get_algo_param(args, "sac_fae_critic_tau"))
        self.encoder_tau = float(get_algo_param(args, "sac_fae_encoder_tau"))
        self.total_it = 0

    def _prepare_depth(self, depth) -> torch.Tensor:
        tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 5 and tensor.shape[2] == 1:
            tensor = tensor.squeeze(2)
        if tensor.dim() != 4 or tuple(tensor.shape[1:]) != self.depth_shape:
            raise ValueError(
                f"SAC_FAE expected depth batch [B,4,128,128], got {tuple(tensor.shape)}."
            )
        return tensor

    def _prepare_base(self, base) -> torch.Tensor:
        tensor = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2 or tensor.shape[1] != self.base_dim:
            raise ValueError(
                f"SAC_FAE expected base batch [B,{self.base_dim}], got {tuple(tensor.shape)}."
            )
        return tensor

    def _state_features(
        self,
        base: torch.Tensor,
        depth: torch.Tensor,
        *,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target:
            visual = self.encoder_target(depth)
            measurement = self.measurement_encoder_target(base)
        else:
            visual = self.autoencoder.encoder(depth)
            measurement = self.measurement_encoder(base)
        return torch.cat((visual, measurement), dim=1), visual

    def select_action(
        self,
        base_state,
        depth,
        deterministic: bool = False,
        with_log_prob: bool = False,
        progress_ratio: float = 0.0,
    ):
        del progress_ratio
        base = self._prepare_base(base_state)
        depth_batch = self._prepare_depth(depth)
        inference_modules = (
            self.autoencoder.encoder,
            self.measurement_encoder,
            self.actor,
        )
        training_states = tuple(module.training for module in inference_modules)
        for module in inference_modules:
            module.eval()
        try:
            with torch.no_grad():
                state, _ = self._state_features(base, depth_batch)
                if with_log_prob and not deterministic:
                    action, log_prob = self.actor.action_log_prob(state)
                    scaled = self.action_scale * action + self.action_bias
                    return scaled.cpu().numpy().reshape(-1), log_prob.cpu().numpy()
                action = self.actor(state, deterministic=deterministic)
                scaled = self.action_scale * action + self.action_bias
                return scaled.cpu().numpy().reshape(-1)
        finally:
            for module, was_training in zip(inference_modules, training_states):
                module.train(was_training)

    def _current_alpha(self) -> torch.Tensor:
        if self.auto_entropy_tuning:
            return self.log_alpha.exp().detach()
        return torch.as_tensor(self.alpha, dtype=torch.float32, device=self.device)

    def train(self, progress_ratio: float = 0.0) -> dict[str, float]:
        del progress_ratio
        if self.replay_buffer.size() < self.batch_size:
            return {}
        self.total_it += 1

        (
            base_states,
            depths,
            actions,
            rewards,
            next_base_states,
            next_depths,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        base_states = self._prepare_base(base_states)
        depths = self._prepare_depth(depths)
        next_base_states = self._prepare_base(next_base_states)
        next_depths = self._prepare_depth(next_depths)
        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        normalized_actions = ((real_actions - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_actor_state, _ = self._state_features(next_base_states, next_depths)
            next_actions, next_log_prob = self.actor.action_log_prob(next_actor_state)
            next_target_state, _ = self._state_features(
                next_base_states,
                next_depths,
                target=True,
            )
            target_q1, target_q2 = self.critic_target(next_target_state, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * (
                torch.minimum(target_q1, target_q2)
                - self._current_alpha() * next_log_prob
            )

        current_state, visual_latent = self._state_features(base_states, depths)
        current_q1, current_q2 = self.critic(current_state, normalized_actions)
        critic_loss = 0.5 * (
            F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        )
        reconstruction = self.autoencoder.decoder(visual_latent)
        reconstruction_target = self.autoencoder.reconstruction_target(depths)
        reconstruction_loss = F.mse_loss(reconstruction, reconstruction_target)
        combined_loss = critic_loss + self.reconstruction_coef * reconstruction_loss

        self.critic_autoencoder_optimizer.zero_grad(set_to_none=True)
        combined_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_autoencoder_params, self.grad_clip)
        self.critic_autoencoder_optimizer.step()

        metrics = {
            "critic_loss": float(critic_loss.detach().item()),
            "reconstruction_loss": float(reconstruction_loss.detach().item()),
            "combined_critic_autoencoder_loss": float(combined_loss.detach().item()),
            "target_q_mean": float(target_q.mean().detach().item()),
            "alpha": float(self.alpha),
        }

        if self.total_it % self.policy_freq == 0:
            with torch.no_grad():
                actor_state, _ = self._state_features(base_states, depths)
            _set_requires_grad(self.critic, False)
            sampled_actions, log_prob = self.actor.action_log_prob(actor_state)
            q1_pi, q2_pi = self.critic(actor_state, sampled_actions)
            actor_loss = (self._current_alpha() * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            _set_requires_grad(self.critic, True)

            metrics["actor_loss"] = float(actor_loss.detach().item())
            metrics["mean_log_prob"] = float(log_prob.mean().detach().item())

            if self.auto_entropy_tuning:
                alpha_loss = -(
                    self.log_alpha * (log_prob + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().detach().item())
                metrics["alpha_loss"] = float(alpha_loss.detach().item())
                metrics["alpha"] = self.alpha

        if self.total_it % self.target_update_interval == 0:
            _soft_update(self.critic, self.critic_target, self.critic_tau)
            _soft_update(self.autoencoder.encoder, self.encoder_target, self.encoder_tau)
            _soft_update(
                self.measurement_encoder,
                self.measurement_encoder_target,
                self.encoder_tau,
            )
        return metrics

    def save(self, path: str) -> None:
        checkpoint = {
            "autoencoder": self.autoencoder.state_dict(),
            "encoder_target": self.encoder_target.state_dict(),
            "measurement_encoder": self.measurement_encoder.state_dict(),
            "measurement_encoder_target": self.measurement_encoder_target.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_autoencoder_optimizer": self.critic_autoencoder_optimizer.state_dict(),
            "total_it": self.total_it,
            "alpha": self.alpha,
        }
        if self.auto_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint["autoencoder"])
        self.encoder_target.load_state_dict(checkpoint["encoder_target"])
        self.encoder_target.eval()
        self.measurement_encoder.load_state_dict(checkpoint["measurement_encoder"])
        self.measurement_encoder_target.load_state_dict(
            checkpoint["measurement_encoder_target"]
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_autoencoder_optimizer.load_state_dict(
            checkpoint["critic_autoencoder_optimizer"]
        )
        self.total_it = int(checkpoint.get("total_it", 0))
        self.alpha = float(checkpoint.get("alpha", self.alpha))
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


__all__ = ["SACFAEAgent"]
