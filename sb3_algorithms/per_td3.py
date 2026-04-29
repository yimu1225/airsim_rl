"""Prioritized replay TD3."""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from sb3_extensions.buffers import PrioritizedReplayBuffer


class PERTD3(TD3):
    def __init__(
        self,
        *args,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.per_alpha = float(per_alpha)
        self.per_beta = float(per_beta)
        self.per_eps = float(per_eps)
        kwargs.setdefault("replay_buffer_class", PrioritizedReplayBuffer)
        replay_kwargs = dict(kwargs.pop("replay_buffer_kwargs", {}) or {})
        replay_kwargs.setdefault("alpha", self.per_alpha)
        replay_kwargs.setdefault("beta", self.per_beta)
        replay_kwargs.setdefault("eps", self.per_eps)
        kwargs["replay_buffer_kwargs"] = replay_kwargs
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
                beta=self.per_beta,
            )
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.0
            td_errors = []
            for current_q in current_q_values:
                td_error = current_q - target_q_values
                td_errors.append(td_error.detach().abs())
                critic_loss = critic_loss + (F.mse_loss(current_q, target_q_values, reduction="none") * replay_data.weights).mean()
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            priorities = th.cat(td_errors, dim=1).max(dim=1).values + 1e-6
            self.replay_buffer.update_priorities(replay_data.indices, priorities)  # type: ignore[union-attr]

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/per_beta", self.per_beta)
