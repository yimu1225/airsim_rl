import numpy as np


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer using proportional prioritization."""

    def __init__(self, capacity, alpha=0.6, eps=1e-6, seed=None, depth_field_indices=()):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.eps = eps
        self.arrays = None
        self.field_shapes = None
        self.field_dtypes = None
        self.current_size = 0
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)
        self.depth_field_indices = set(int(i) for i in depth_field_indices)

    def _dtype_for_field(self, field_idx):
        return np.uint8 if field_idx in self.depth_field_indices else np.float32

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

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.current_size == 0:
            return None

        current_len = self.current_size
        priorities = self.priorities[:current_len]
        probs = priorities ** self.alpha
        prob_sum = probs.sum()
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            probs = np.full((current_len,), 1.0 / current_len, dtype=np.float32)
        else:
            probs /= prob_sum

        indices = self.rng.choice(current_len, batch_size, p=probs)
        samples = []
        for field_idx, array in enumerate(self.arrays):
            values = array[indices]
            if field_idx in self.depth_field_indices:
                values = values.astype(np.float32)
            samples.append(values)

        weights = (current_len * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return tuple(samples), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            p = float(priority) + self.eps
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def size(self):
        return self.current_size


class DualPrioritizedReplayBuffer:
    """Dual replay pools with PER in each pool."""

    def __init__(
        self,
        capacity,
        success_capacity_ratio=0.3,
        success_sample_ratio=0.5,
        alpha=0.6,
        eps=1e-6,
        seed=None,
    ):
        total_capacity = int(capacity)
        success_capacity_ratio = float(np.clip(success_capacity_ratio, 0.05, 0.95))
        self.success_sample_ratio = float(np.clip(success_sample_ratio, 0.0, 1.0))

        success_capacity = max(1, int(round(total_capacity * success_capacity_ratio)))
        regular_capacity = max(1, total_capacity - success_capacity)

        depth_field_indices = (1, 2, 7, 8)
        self.success_buffer = PrioritizedReplayBuffer(
            success_capacity, alpha=alpha, eps=eps, seed=seed, depth_field_indices=depth_field_indices
        )
        self.regular_buffer = PrioritizedReplayBuffer(
            regular_capacity,
            alpha=alpha,
            eps=eps,
            seed=None if seed is None else seed + 1,
            depth_field_indices=depth_field_indices,
        )

        self.rng = np.random.default_rng(seed)
        self._episode_cache = []
        self._episode_success = False
        self.critic_priv_dim = None

    @staticmethod
    def _flatten_priv(priv, target_dim=None):
        if priv is None:
            if target_dim is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros((target_dim,), dtype=np.float32)
        arr = np.asarray(priv, dtype=np.float32).reshape(-1)
        if target_dim is not None and arr.size != target_dim:
            raise ValueError(f"critic_priv dim mismatch: expected {target_dim}, got {arr.size}")
        return arr

    def add(
        self,
        base_state,
        depth,
        action,
        reward,
        next_base_state,
        next_depth,
        done,
        is_success=0.0,
        critic_priv=None,
        next_critic_priv=None,
        critic_depth=None,
        next_critic_depth=None,
    ):
        if critic_depth is None:
            critic_depth = depth
        if next_critic_depth is None:
            next_critic_depth = next_depth

        if self.critic_priv_dim is None:
            critic_priv_flat = self._flatten_priv(critic_priv)
            self.critic_priv_dim = int(critic_priv_flat.size)
        else:
            critic_priv_flat = self._flatten_priv(critic_priv, target_dim=self.critic_priv_dim)
        next_critic_priv_flat = self._flatten_priv(next_critic_priv, target_dim=self.critic_priv_dim)

        transition = (
            base_state,
            depth,
            critic_depth,
            critic_priv_flat,
            action,
            reward,
            next_base_state,
            next_depth,
            next_critic_depth,
            next_critic_priv_flat,
            done,
        )
        self._episode_cache.append(transition)
        if bool(is_success):
            self._episode_success = True

        if bool(done):
            target = self.success_buffer if self._episode_success else self.regular_buffer
            for item in self._episode_cache:
                target.add(*item)
            self._episode_cache = []
            self._episode_success = False

    def _split_batch(self, batch_size):
        success_size = self.success_buffer.size()
        regular_size = self.regular_buffer.size()

        desired_success = int(round(batch_size * self.success_sample_ratio))
        num_success = min(desired_success, success_size)
        num_regular = min(batch_size - num_success, regular_size)

        remaining = batch_size - (num_success + num_regular)
        if remaining > 0:
            extra_success = min(remaining, max(0, success_size - num_success))
            num_success += extra_success
            remaining -= extra_success
        if remaining > 0:
            extra_regular = min(remaining, max(0, regular_size - num_regular))
            num_regular += extra_regular

        return num_success, num_regular

    def sample(self, batch_size, beta=0.4):
        if self.size() == 0:
            return None

        n_success, n_regular = self._split_batch(batch_size)
        if n_success + n_regular <= 0:
            return None

        sample_batches = []
        refs = []
        weights = []
        sampled_success = 0
        sampled_regular = 0

        if n_success > 0:
            out = self.success_buffer.sample(n_success, beta=beta)
            if out is not None:
                s_batch, s_indices, s_weights = out
                sample_batches.append(s_batch)
                refs.extend([("success", int(idx)) for idx in s_indices])
                weights.append(np.asarray(s_weights, dtype=np.float32))
                sampled_success = len(s_indices)

        if n_regular > 0:
            out = self.regular_buffer.sample(n_regular, beta=beta)
            if out is not None:
                r_batch, r_indices, r_weights = out
                sample_batches.append(r_batch)
                refs.extend([("regular", int(idx)) for idx in r_indices])
                weights.append(np.asarray(r_weights, dtype=np.float32))
                sampled_regular = len(r_indices)

        if len(sample_batches) == 0:
            return None

        merged_weights = np.concatenate(weights, axis=0) if len(weights) > 1 else weights[0]
        if len(sample_batches) > 1:
            samples = tuple(
                np.concatenate([batch[field_idx] for batch in sample_batches], axis=0)
                for field_idx in range(len(sample_batches[0]))
            )
        else:
            samples = sample_batches[0]

        order = self.rng.permutation(merged_weights.shape[0])
        samples = tuple(field[order] for field in samples)
        refs = [refs[i] for i in order]
        merged_weights = merged_weights[order]

        sampled_total = max(1, sampled_success + sampled_regular)
        mix_info = {
            "batch_success_fraction": float(sampled_success) / float(sampled_total),
            "success_size": self.success_buffer.size(),
            "regular_size": self.regular_buffer.size(),
        }
        return samples, refs, merged_weights, mix_info

    def update_priorities(self, refs, priorities):
        success_indices, success_priorities = [], []
        regular_indices, regular_priorities = [], []

        for (pool_name, idx), priority in zip(refs, priorities):
            if pool_name == "success":
                success_indices.append(idx)
                success_priorities.append(priority)
            else:
                regular_indices.append(idx)
                regular_priorities.append(priority)

        if success_indices:
            self.success_buffer.update_priorities(success_indices, success_priorities)
        if regular_indices:
            self.regular_buffer.update_priorities(regular_indices, regular_priorities)

    def size(self):
        return self.success_buffer.size() + self.regular_buffer.size()
