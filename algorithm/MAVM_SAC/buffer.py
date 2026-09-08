"""Replay buffer for frozen MAVM representations."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from ..sum_tree_replay import SumTreePrioritySampler


class FeatureBatch(NamedTuple):
    base: torch.Tensor
    memory: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_base: torch.Tensor
    next_memory: torch.Tensor
    done: torch.Tensor


class FeatureReplayBuffer:
    """Stores m_t directly; valid only while encoder and memory are frozen."""

    def __init__(
        self,
        capacity: int,
        base_dim: int,
        memory_dim: int,
        action_dim: int,
        *,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.base = np.empty((capacity, base_dim), dtype=np.float32)
        self.memory = np.empty((capacity, memory_dim), dtype=np.float32)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.next_base = np.empty((capacity, base_dim), dtype=np.float32)
        self.next_memory = np.empty((capacity, memory_dim), dtype=np.float32)
        self.done = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        base: np.ndarray,
        memory: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_base: np.ndarray,
        next_memory: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.base[index] = base
        self.memory[index] = memory
        self.action[index] = action
        self.reward[index] = reward
        self.next_base[index] = next_base
        self.next_memory[index] = next_memory
        self.done[index] = done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, device: torch.device) -> FeatureBatch:
        if self.size < batch_size:
            raise ValueError(f"cannot sample {batch_size} transitions from {self.size}")
        indices = self.rng.integers(0, self.size, size=batch_size)
        tensor = lambda array: torch.as_tensor(array[indices], device=device)
        return FeatureBatch(
            tensor(self.base),
            tensor(self.memory),
            tensor(self.action),
            tensor(self.reward),
            tensor(self.next_base),
            tensor(self.next_memory),
            tensor(self.done),
        )


class PrioritizedFeatureReplayBuffer(FeatureReplayBuffer):
    """Feature replay with proportional TD-error prioritization."""

    def __init__(
        self,
        capacity: int,
        base_dim: int,
        memory_dim: int,
        action_dim: int,
        *,
        alpha: float = 0.6,
        eps: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        super().__init__(capacity, base_dim, memory_dim, action_dim, seed=seed)
        self.eps = float(eps)
        self.priority_sampler = SumTreePrioritySampler(
            capacity, alpha=alpha, eps=eps, seed=seed
        )

    @property
    def priorities(self) -> np.ndarray:
        return self.priority_sampler.priorities

    def add(self, *transition) -> None:
        index = self.position
        super().add(*transition)
        self.priority_sampler.set_priority(
            index, self.priority_sampler.max_priority
        )

    def sample(
        self, batch_size: int, device: torch.device, *, beta: float
    ) -> tuple[FeatureBatch, np.ndarray, np.ndarray]:
        if self.size <= 0:
            raise ValueError("cannot sample from an empty replay buffer")
        indices, probabilities = self.priority_sampler.sample(batch_size, self.size)
        tensor = lambda array: torch.as_tensor(array[indices], device=device)
        batch = FeatureBatch(
            tensor(self.base),
            tensor(self.memory),
            tensor(self.action),
            tensor(self.reward),
            tensor(self.next_base),
            tensor(self.next_memory),
            tensor(self.done),
        )
        weights = (self.size * probabilities) ** (-float(beta))
        weights /= max(float(weights.max()), self.eps)
        return batch, indices, np.asarray(weights, dtype=np.float32)

    def update_priorities(
        self, indices: list[int] | np.ndarray, priorities: list[float] | np.ndarray
    ) -> None:
        for index, priority in zip(indices, priorities):
            self.priority_sampler.set_priority(
                int(index), float(priority) + self.eps
            )


class DualPrioritizedFeatureReplayBuffer:
    """Success/regular episode pools, each using proportional PER.

    Transitions stay in an episode cache until ``done`` is observed. This is
    necessary because a transition from the start of an episode can only be
    classified after the episode's final success outcome is known.
    """

    def __init__(
        self,
        capacity: int,
        base_dim: int,
        memory_dim: int,
        action_dim: int,
        *,
        success_capacity_ratio: float = 0.3,
        success_sample_ratio: float = 0.3,
        alpha: float = 0.6,
        eps: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        if capacity <= 1:
            raise ValueError("dual replay capacity must be greater than one")
        self.eps = float(eps)
        capacity_ratio = float(np.clip(success_capacity_ratio, 0.05, 0.95))
        self.success_sample_ratio = float(
            np.clip(success_sample_ratio, 0.0, 1.0)
        )
        success_capacity = max(1, int(round(capacity * capacity_ratio)))
        regular_capacity = max(1, capacity - success_capacity)
        common = {
            "base_dim": base_dim,
            "memory_dim": memory_dim,
            "action_dim": action_dim,
            "alpha": alpha,
            "eps": eps,
        }
        self.success_buffer = PrioritizedFeatureReplayBuffer(
            success_capacity, seed=seed, **common
        )
        self.regular_buffer = PrioritizedFeatureReplayBuffer(
            regular_capacity,
            seed=None if seed is None else seed + 1,
            **common,
        )
        self.rng = np.random.default_rng(seed)
        self._episode_cache: list[tuple] = []
        self._episode_success = False

    def add(
        self,
        base: np.ndarray,
        memory: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_base: np.ndarray,
        next_memory: np.ndarray,
        done: bool,
        *,
        is_success: bool | float = False,
    ) -> None:
        self._episode_cache.append(
            (base, memory, action, reward, next_base, next_memory, done)
        )
        self._episode_success = self._episode_success or bool(is_success)
        if not bool(done):
            return

        target = (
            self.success_buffer if self._episode_success else self.regular_buffer
        )
        for transition in self._episode_cache:
            target.add(*transition)
        self._episode_cache.clear()
        self._episode_success = False

    def _split_batch(self, batch_size: int) -> tuple[int, int]:
        success_size = self.success_buffer.size
        regular_size = self.regular_buffer.size
        desired_success = int(round(batch_size * self.success_sample_ratio))
        num_success = min(desired_success, success_size)
        num_regular = min(batch_size - num_success, regular_size)

        remaining = batch_size - num_success - num_regular
        if remaining > 0:
            extra_success = min(remaining, success_size - num_success)
            num_success += extra_success
            remaining -= extra_success
        if remaining > 0:
            num_regular += min(remaining, regular_size - num_regular)
        return num_success, num_regular

    def sample(
        self, batch_size: int, device: torch.device, *, beta: float
    ) -> tuple[FeatureBatch, list[tuple[str, int]], torch.Tensor, dict[str, float]]:
        if len(self) < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from {len(self)}"
            )
        num_success, num_regular = self._split_batch(batch_size)
        batches: list[FeatureBatch] = []
        refs: list[tuple[str, int]] = []
        weights: list[np.ndarray] = []

        if num_success:
            batch, indices, pool_weights = self.success_buffer.sample(
                num_success, device, beta=beta
            )
            batches.append(batch)
            refs.extend(("success", int(index)) for index in indices)
            weights.append(pool_weights)
        if num_regular:
            batch, indices, pool_weights = self.regular_buffer.sample(
                num_regular, device, beta=beta
            )
            batches.append(batch)
            refs.extend(("regular", int(index)) for index in indices)
            weights.append(pool_weights)

        merged = FeatureBatch(
            *(torch.cat([getattr(batch, field) for batch in batches], dim=0)
              for field in FeatureBatch._fields)
        )
        merged_weights = np.concatenate(weights)
        order = self.rng.permutation(len(refs))
        order_tensor = torch.as_tensor(order, device=device)
        merged = FeatureBatch(
            *(getattr(merged, field)[order_tensor] for field in FeatureBatch._fields)
        )
        refs = [refs[index] for index in order]
        weight_tensor = torch.as_tensor(
            merged_weights[order], dtype=torch.float32, device=device
        ).unsqueeze(-1)
        sampled_total = max(1, num_success + num_regular)
        info = {
            "batch_success_fraction": num_success / sampled_total,
            "success_size": self.success_buffer.size,
            "regular_size": self.regular_buffer.size,
        }
        return merged, refs, weight_tensor, info

    def update_priorities(
        self,
        refs: list[tuple[str, int]],
        priorities: list[float] | np.ndarray,
    ) -> None:
        success_indices, success_priorities = [], []
        regular_indices, regular_priorities = [], []
        for (pool_name, index), priority in zip(refs, priorities):
            if pool_name == "success":
                success_indices.append(index)
                success_priorities.append(priority)
            else:
                regular_indices.append(index)
                regular_priorities.append(priority)
        if success_indices:
            self.success_buffer.update_priorities(
                success_indices, success_priorities
            )
        if regular_indices:
            self.regular_buffer.update_priorities(
                regular_indices, regular_priorities
            )

    def __len__(self) -> int:
        return self.success_buffer.size + self.regular_buffer.size
