"""SAC operating only on frozen temporal-Mamba state plus base state."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .buffer import DualPrioritizedFeatureReplayBuffer
from .checkpoints import MODEL_VERSION, atomic_torch_save
from .config import SSVMConfig
from .networks import MambaPerceptionMemory


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(state_dim)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(self.norm(state))
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_probability = distribution.log_prob(raw).sum(-1, keepdim=True)
        correction = 2.0 * (np.log(2.0) - raw - F.softplus(-2.0 * raw))
        return action, log_probability - correction.sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        hidden = self.trunk(self.norm(state))
        mean = self.mean(hidden)
        if deterministic:
            return torch.tanh(mean)
        std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return torch.tanh(torch.distributions.Normal(mean, std).sample())


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = state_dim + action_dim

        def q_network() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1, self.q2 = q_network(), q_network()

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat((state, action), dim=-1)
        return self.q1(values), self.q2(values)


class SSVMSACAgent:
    """Owns frozen perception state and trainable SAC heads.

    SAC receives exactly ``concat(base_t, m_t)``. The encoder output ``z_t`` is
    never concatenated into actor or critic input.
    """

    def __init__(
        self,
        perception: MambaPerceptionMemory,
        base_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: SSVMConfig,
        *,
        device: torch.device | str,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.config = config
        self.perception = perception.to(self.device).freeze()
        self.base_dim = base_dim
        action_low = np.asarray(action_low, dtype=np.float32)
        action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = action_low.size
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = torch.as_tensor((action_high - action_low) / 2, device=self.device)
        self.action_bias = torch.as_tensor((action_high + action_low) / 2, device=self.device)

        state_dim = base_dim + config.memory_dim
        self.actor = SquashedGaussianActor(
            state_dim, self.action_dim, config.hidden_dim
        ).to(self.device)
        self.critic = TwinCritic(state_dim, self.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.log_alpha = torch.tensor(
            np.log(config.initial_alpha), device=self.device, requires_grad=config.auto_entropy
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
            if config.auto_entropy else None
        )
        self.target_entropy = -float(self.action_dim)
        self.replay = DualPrioritizedFeatureReplayBuffer(
            config.replay_capacity,
            base_dim,
            config.memory_dim,
            self.action_dim,
            success_capacity_ratio=config.sb_per_success_capacity_ratio,
            success_sample_ratio=config.sb_per_success_sample_ratio,
            alpha=config.sb_per_alpha,
            eps=config.sb_per_eps,
            seed=seed,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def reset(self) -> None:
        self.perception.reset(1)

    reset_memory = reset

    @torch.no_grad()
    def encode_frame(self, depth: np.ndarray) -> np.ndarray:
        frame = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        if frame.ndim == 2:
            frame = frame.unsqueeze(0)
        frame = frame / 255.0
        value = self.perception.step(frame.unsqueeze(0))
        return value.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def select_action(
        self, base: np.ndarray, memory: np.ndarray, *, deterministic: bool = False
    ) -> np.ndarray:
        state = torch.as_tensor(
            np.concatenate((base, memory)), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        normalized = self.actor(state, deterministic=deterministic)
        return (self.action_scale * normalized + self.action_bias).squeeze(0).cpu().numpy()

    def add_transition(
        self, *transition: Any, is_success: bool | float = False
    ) -> None:
        self.replay.add(*transition, is_success=is_success)

    def _sb_per_beta(self, progress_ratio: float) -> float:
        progress = float(np.clip(progress_ratio, 0.0, 1.0))
        return (
            self.config.sb_per_beta0 * (1.0 - progress)
            + self.config.sb_per_beta1 * progress
        )

    def _success_sample_ratio(self, progress_ratio: float) -> float:
        progress = float(np.clip(progress_ratio, 0.0, 1.0))
        if progress < self.config.sb_per_mu_step1:
            ratio = self.config.sb_per_mu_low
        elif progress < self.config.sb_per_mu_step2:
            ratio = self.config.sb_per_mu_mid
        else:
            ratio = self.config.sb_per_mu_high
        return float(np.clip(ratio, 0.0, 0.8))

    def update(
        self,
        batch_size: int | None = None,
        *,
        progress_ratio: float = 0.0,
    ) -> dict[str, float]:
        size = batch_size or self.config.batch_size
        if len(self.replay) < size:
            return {}
        beta = self._sb_per_beta(progress_ratio)
        success_sample_ratio = self._success_sample_ratio(progress_ratio)
        self.replay.success_sample_ratio = success_sample_ratio
        batch, replay_refs, replay_weights, replay_info = self.replay.sample(
            size, self.device, beta=beta
        )
        state = torch.cat((batch.base, batch.memory), dim=-1)
        next_state = torch.cat((batch.next_base, batch.next_memory), dim=-1)
        normalized_action = ((batch.action - self.action_bias) / self.action_scale).clamp(-1, 1)

        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.minimum(next_q1, next_q2) - self.alpha.detach() * next_log_probability
            target = batch.reward + (1.0 - batch.done) * self.config.gamma * next_q

        q1, q2 = self.critic(state, normalized_action)
        critic_loss_elements = 0.5 * (
            F.mse_loss(q1, target, reduction="none")
            + F.mse_loss(q2, target, reduction="none")
        )
        critic_loss = (critic_loss_elements * replay_weights).mean()
        td_errors = 0.5 * ((q1 - target).abs() + (q2 - target).abs())
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()
        self.replay.update_priorities(
            replay_refs, td_errors.detach().cpu().numpy().reshape(-1)
        )

        sampled_action, log_probability = self.actor.sample(state)
        actor_q1, actor_q2 = self.critic(state, sampled_action)
        actor_loss = (
            self.alpha.detach() * log_probability - torch.minimum(actor_q1, actor_q2)
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()

        alpha_loss = torch.zeros((), device=self.device)
        if self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_probability + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_parameter.lerp_(parameter, self.config.tau)
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "sb_per_beta": beta,
            "replay/success_sample_ratio_target": success_sample_ratio,
            "replay/success_batch_fraction": replay_info[
                "batch_success_fraction"
            ],
            "replay/success_size": replay_info["success_size"],
            "replay/regular_size": replay_info["regular_size"],
        }

    def checkpoint(self, stage: str, step: int) -> dict[str, Any]:
        return {
            "stage": stage,
            "model_version": MODEL_VERSION,
            "dataset_version": 1,
            "step": step,
            "config": self.config.to_dict(),
            "normalization": {"depth_divisor": 255.0},
            "base_dim": self.base_dim,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "perception": self.perception.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": (
                self.alpha_optimizer.state_dict() if self.alpha_optimizer else None
            ),
        }

    def save(self, path: str | Path, *, stage: str, step: int) -> Path:
        return atomic_torch_save(self.checkpoint(stage, step), path)

    def load_sac_state(self, checkpoint: dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if self.alpha_optimizer and checkpoint.get("alpha_optimizer"):
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
