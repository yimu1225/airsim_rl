import numpy as np

from ..sum_tree_replay import SumTreePrioritizedReplayBuffer

class PrioritizedReplayBuffer(SumTreePrioritizedReplayBuffer):
    """Prioritized replay buffer using proportional SumTree sampling."""

    def __init__(self, capacity, alpha=0.6, eps=1e-6, seed=None, depth_field_indices=()):
        super().__init__(
            capacity,
            alpha=alpha,
            eps=eps,
            seed=seed,
            depth_field_indices=depth_field_indices,
            depth_dtype=np.float16,
            return_stacked=True,
            return_depth_float32=False,
        )


class DualPrioritizedReplayBuffer:
    """Dual replay pools with PER in each pool."""

    def __init__(
        self,
        capacity,
        success_capacity_ratio=0.3,
        success_sample_ratio=0.30,
        alpha=0.6,
        eps=1e-6,
        seed=None,
    ):
        total_capacity = int(capacity)
        success_capacity_ratio = float(np.clip(success_capacity_ratio, 0.05, 0.95))
        self.success_sample_ratio = float(np.clip(success_sample_ratio, 0.0, 1.0))

        success_capacity = max(1, int(round(total_capacity * success_capacity_ratio)))
        regular_capacity = max(1, total_capacity - success_capacity)

        depth_field_indices = (1, 5, 7, 8)
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
        self.critic_priv_shape = None

    @staticmethod
    def _as_priv(priv, target_shape=None):
        if priv is None:
            if target_shape is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros(target_shape, dtype=np.float32)
        arr = np.asarray(priv, dtype=np.float32)
        if target_shape is not None and arr.shape != tuple(target_shape):
            raise ValueError(f"critic_depth shape mismatch: expected {target_shape}, got {arr.shape}")
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
    ):
        if self.critic_priv_shape is None:
            critic_depth = self._as_priv(critic_priv)
            self.critic_priv_shape = critic_depth.shape
        else:
            critic_depth = self._as_priv(critic_priv, target_shape=self.critic_priv_shape)
        next_critic_depth = self._as_priv(next_critic_priv, target_shape=self.critic_priv_shape)

        transition = (
            base_state,
            depth,
            action,
            reward,
            next_base_state,
            next_depth,
            done,
            critic_depth,
            next_critic_depth,
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
