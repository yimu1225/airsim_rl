import numpy as np


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer using proportional prioritization."""

    def __init__(self, capacity, alpha=0.6, eps=1e-6, seed=None):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.eps = eps
        self.storage = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    def add(self, *data):
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.pos] = data

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.storage) == 0:
            return None

        current_len = len(self.storage)
        priorities = self.priorities[:current_len]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = self.rng.choice(current_len, batch_size, p=probs)
        samples = [self.storage[idx] for idx in indices]

        weights = (current_len * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            p = float(priority) + self.eps
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def size(self):
        return len(self.storage)


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

        self.success_buffer = PrioritizedReplayBuffer(
            success_capacity, alpha=alpha, eps=eps, seed=seed
        )
        self.regular_buffer = PrioritizedReplayBuffer(
            regular_capacity, alpha=alpha, eps=eps, seed=None if seed is None else seed + 1
        )

        self.rng = np.random.default_rng(seed)
        self._episode_cache = []
        self._episode_success = False

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
    ):
        transition = (
            base_state,
            depth,
            action,
            reward,
            next_base_state,
            next_depth,
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

        samples = []
        refs = []
        weights = []

        if n_success > 0:
            out = self.success_buffer.sample(n_success, beta=beta)
            if out is not None:
                s_samples, s_indices, s_weights = out
                samples.extend(s_samples)
                refs.extend([("success", int(idx)) for idx in s_indices])
                weights.append(np.asarray(s_weights, dtype=np.float32))

        if n_regular > 0:
            out = self.regular_buffer.sample(n_regular, beta=beta)
            if out is not None:
                r_samples, r_indices, r_weights = out
                samples.extend(r_samples)
                refs.extend([("regular", int(idx)) for idx in r_indices])
                weights.append(np.asarray(r_weights, dtype=np.float32))

        if len(samples) == 0:
            return None

        merged_weights = np.concatenate(weights, axis=0) if len(weights) > 1 else weights[0]

        order = self.rng.permutation(len(samples))
        samples = [samples[i] for i in order]
        refs = [refs[i] for i in order]
        merged_weights = merged_weights[order]

        mix_info = {
            "batch_success_fraction": float(n_success) / float(max(1, n_success + n_regular)),
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
