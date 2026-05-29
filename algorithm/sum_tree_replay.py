from __future__ import annotations

import numpy as np


class SumTreePrioritySampler:
    """Binary sum tree for sampling integer indices by proportional priority."""

    def __init__(self, capacity, alpha=0.6, eps=1e-6, seed=None):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("SumTree capacity must be positive.")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity
        self.sum_tree = np.zeros((2 * tree_capacity,), dtype=np.float32)

    def _scaled_priority(self, priority):
        p = max(float(priority), self.eps)
        return float(p ** self.alpha)

    def set_priority(self, data_idx, priority):
        data_idx = int(data_idx)
        p = max(float(priority), self.eps)
        self.priorities[data_idx] = p
        self.max_priority = max(self.max_priority, p)

        tree_idx = self.tree_capacity + data_idx
        change = self._scaled_priority(p) - float(self.sum_tree[tree_idx])
        while tree_idx >= 1:
            self.sum_tree[tree_idx] += change
            tree_idx //= 2

    def total_priority(self):
        return float(self.sum_tree[1])

    def _find_prefixsum_idx(self, mass, current_size):
        idx = 1
        while idx < self.tree_capacity:
            left = idx * 2
            if mass <= self.sum_tree[left]:
                idx = left
            else:
                mass -= float(self.sum_tree[left])
                idx = left + 1
        data_idx = idx - self.tree_capacity
        if data_idx >= current_size:
            data_idx = current_size - 1
        return int(data_idx)

    def sample(self, batch_size, current_size):
        current_size = int(current_size)
        if current_size <= 0:
            raise ValueError("Cannot sample from an empty priority tree.")

        total = self.total_priority()
        if not np.isfinite(total) or total <= 0.0:
            indices = self.rng.integers(0, current_size, size=int(batch_size), endpoint=False)
            probs = np.full((int(batch_size),), 1.0 / float(current_size), dtype=np.float32)
            return indices.astype(np.int64), probs

        batch_size = int(batch_size)
        masses = self.rng.random(batch_size, dtype=np.float64) * total
        indices = np.ones((batch_size,), dtype=np.int64)
        while indices[0] < self.tree_capacity:
            left = indices * 2
            left_values = self.sum_tree[left]
            go_right = masses > left_values
            masses = masses - left_values * go_right
            indices = left + go_right.astype(np.int64)
        indices = indices - self.tree_capacity
        indices = np.minimum(indices, current_size - 1).astype(np.int64, copy=False)
        leaf_values = self.sum_tree[self.tree_capacity + indices]
        probs = np.maximum(leaf_values / max(total, self.eps), self.eps).astype(np.float32)
        return indices, probs

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.set_priority(int(idx), float(priority))


class SumTreePrioritizedReplayBuffer:
    """Proportional prioritized replay backed by a binary sum tree."""

    def __init__(
        self,
        capacity,
        alpha=0.6,
        eps=1e-6,
        seed=None,
        depth_field_indices=(),
        depth_dtype=np.float16,
        return_stacked=True,
        return_depth_float32=True,
    ):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.depth_dtype = depth_dtype
        self.return_stacked = bool(return_stacked)
        self.return_depth_float32 = bool(return_depth_float32)

        self.arrays = None
        self.field_shapes = None
        self.field_dtypes = None
        self.current_size = 0
        self.pos = 0
        self.priority_sampler = SumTreePrioritySampler(self.capacity, alpha=alpha, eps=eps, seed=seed)
        self.priorities = self.priority_sampler.priorities
        self.depth_field_indices = set(int(i) for i in depth_field_indices)

    def _dtype_for_field(self, field_idx):
        return self.depth_dtype if field_idx in self.depth_field_indices else np.float32

    @staticmethod
    def _encode_field(item, dtype):
        if dtype == np.uint8:
            return np.clip(np.rint(np.asarray(item, dtype=np.float32)), 0.0, 255.0).astype(np.uint8)
        return np.asarray(item, dtype=dtype)

    def _allocate(self, data):
        self.field_shapes = [np.asarray(item).shape for item in data]
        self.field_dtypes = [self._dtype_for_field(i) for i in range(len(data))]
        self.arrays = [
            np.zeros((self.capacity, *shape), dtype=dtype)
            for shape, dtype in zip(self.field_shapes, self.field_dtypes)
        ]

    def _validate_shapes(self, data):
        if len(data) != len(self.arrays):
            raise ValueError(f"Replay transition length mismatch: expected {len(self.arrays)}, got {len(data)}")
        for field_idx, (item, expected_shape) in enumerate(zip(data, self.field_shapes)):
            shape = np.asarray(item).shape
            if shape != expected_shape:
                raise ValueError(
                    f"Replay field {field_idx} shape mismatch: expected {expected_shape}, got {shape}"
                )

    def add(self, *data):
        if self.arrays is None:
            self._allocate(data)
        else:
            self._validate_shapes(data)

        for field_idx, item in enumerate(data):
            self.arrays[field_idx][self.pos] = self._encode_field(item, self.field_dtypes[field_idx])

        self.priority_sampler.set_priority(self.pos, self.priority_sampler.max_priority)
        self.pos = (self.pos + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.current_size == 0:
            return None

        indices, probs = self.priority_sampler.sample(int(batch_size), self.current_size)

        samples = []
        for field_idx, array in enumerate(self.arrays):
            values = array[indices]
            if self.return_depth_float32 and field_idx in self.depth_field_indices:
                values = values.astype(np.float32)
            samples.append(values)

        weights = (self.current_size * probs) ** (-float(beta))
        weights /= max(float(weights.max()), self.eps)
        weights = np.asarray(weights, dtype=np.float32)

        if self.return_stacked:
            return tuple(samples), indices, weights
        return list(zip(*samples)), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            p = float(priority) + self.eps
            self.priority_sampler.set_priority(int(idx), p)

    @property
    def max_priority(self):
        return self.priority_sampler.max_priority

    @property
    def sum_tree(self):
        return self.priority_sampler.sum_tree

    @property
    def tree_capacity(self):
        return self.priority_sampler.tree_capacity

    def __len__(self):
        return self.current_size

    def size(self):
        return self.current_size
