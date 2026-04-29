"""Prioritized replay buffer for Dict observations."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize


class PrioritizedDictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor | None
    weights: th.Tensor
    indices: np.ndarray


class PrioritizedReplayBuffer(DictReplayBuffer):
    """Proportional PER implementation compatible with SB3 DictReplayBuffer."""

    def __init__(
        self,
        *args,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, *args, **kwargs) -> None:
        insert_pos = self.pos
        super().add(*args, **kwargs)
        self.priorities[insert_pos] = self.max_priority

    def sample(
        self,
        batch_size: int,
        env: VecNormalize | None = None,
        beta: float | None = None,
    ) -> PrioritizedDictReplayBufferSamples:
        size = self.size()
        if size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        priorities = self.priorities[:size].copy()
        if not np.any(priorities > 0):
            priorities.fill(1.0)
        scaled = np.power(priorities + self.eps, self.alpha)
        probs = scaled / np.sum(scaled)

        batch_inds = np.random.choice(size, size=batch_size, replace=True, p=probs)
        samples = self._get_samples(batch_inds, env=env)

        beta_value = self.beta if beta is None else float(beta)
        weights = np.power(size * probs[batch_inds], -beta_value)
        weights /= max(float(weights.max()), self.eps)
        weights_t = th.as_tensor(weights.reshape(-1, 1), dtype=th.float32, device=self.device)

        return PrioritizedDictReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            discounts=samples.discounts,
            weights=weights_t,
            indices=batch_inds,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray | th.Tensor) -> None:
        priorities_np = priorities.detach().cpu().numpy() if isinstance(priorities, th.Tensor) else np.asarray(priorities)
        priorities_np = np.asarray(priorities_np, dtype=np.float32).reshape(-1)
        priorities_np = np.maximum(priorities_np, self.eps)
        self.priorities[np.asarray(indices, dtype=np.int64)] = priorities_np
        self.max_priority = max(self.max_priority, float(priorities_np.max(initial=self.max_priority)))

