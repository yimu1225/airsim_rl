import numpy as np
import torch

class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer using proportional prioritization."""
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.storage = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

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

        indices = np.random.choice(current_len, batch_size, p=probs)
        samples = [self.storage[idx] for idx in indices]

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
        return len(self.storage)

    def size(self):
        return len(self.storage)
