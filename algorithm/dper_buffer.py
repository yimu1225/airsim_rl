import numpy as np

class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer using proportional prioritization.

    The buffer owns its own RNG so that each instance can be seeded
    independently.
    """
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
        return np.float16 if field_idx in self.depth_field_indices else np.float32

    def _allocate(self, data):
        self.field_shapes = [np.asarray(item).shape for item in data]
        self.field_dtypes = [self._dtype_for_field(i) for i in range(len(data))]
        self.arrays = [
            np.zeros((self.capacity, *shape), dtype=dtype)
            for shape, dtype in zip(self.field_shapes, self.field_dtypes)
        ]

    def add(self, *data):
        if self.arrays is None:
            self._allocate(data)
        elif len(data) != len(self.arrays):
            raise ValueError(f"Replay transition length mismatch: expected {len(self.arrays)}, got {len(data)}")

        for field_idx, item in enumerate(data):
            shape = np.asarray(item).shape
            if shape != self.field_shapes[field_idx]:
                raise ValueError(
                    f"Replay field {field_idx} shape mismatch: expected {self.field_shapes[field_idx]}, got {shape}"
                )
            self.arrays[field_idx][self.pos] = np.asarray(item, dtype=self.field_dtypes[field_idx])
            
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
        batch_fields = []
        for field_idx, array in enumerate(self.arrays):
            values = array[indices]
            if field_idx in self.depth_field_indices:
                values = values.astype(np.float32)
            batch_fields.append(values)
        samples = list(zip(*batch_fields))

        # Importance sampling weights
        weights = (current_len * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps
            self.max_priority = max(self.max_priority, priority + self.eps)

    def __len__(self):
        return self.current_size

    def size(self):
        return self.current_size
