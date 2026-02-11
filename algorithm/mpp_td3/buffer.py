import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        state_np = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in state]
        next_state_np = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in next_state]
        
        data = (state_np, action, reward, next_state_np, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
            
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
            
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps
            self.max_priority = max(self.max_priority, priority + self.eps)

    def delete_oldest(self):
        # In a circular buffer, the replacement handles "deletion"
        # but if we need explicit sync, we might need a more complex structure
        pass
    
    def __len__(self):
        return len(self.buffer)

class MultiPoolAdaptiveBuffer:
    def __init__(self, total_capacity, random_capacity=3000, alpha=0.6):
        self.random_buffer = PrioritizedReplayBuffer(random_capacity, alpha=alpha)
        
        # Noisy and Clean pools are paired and have equal capacity
        # Total capacity - random capacity, then split by 2
        self.agent_pool_capacity = (total_capacity - random_capacity) // 2
        self.noisy_buffer = PrioritizedReplayBuffer(self.agent_pool_capacity, alpha=alpha)
        self.clean_buffer = PrioritizedReplayBuffer(self.agent_pool_capacity, alpha=alpha)

    def add_random(self, state, action, reward, next_state, done):
        self.random_buffer.add(state, action, reward, next_state, done)

    def add_agent_pair(self, state, noisy_action, clean_action, reward, next_state, done):
        # Simultaneous addition to maintain pairing
        # The positions in both buffers will be synchronized because they have the same capacity
        self.noisy_buffer.add(state, noisy_action, reward, next_state, done)
        self.clean_buffer.add(state, clean_action, reward, next_state, done)

    def sample(self, batch_size, weights, beta=0.4):
        """
        weights: [w_random, w_noisy, w_clean]
        """
        # Determine how many to sample from each pool
        counts = np.random.multinomial(batch_size, weights)
        
        all_samples = []
        all_indices = [] # (buffer_type, index_in_buffer)
        all_weights = []

        pool_info = [
            (self.random_buffer, counts[0], 0),
            (self.noisy_buffer, counts[1], 1),
            (self.clean_buffer, counts[2], 2)
        ]

        for buffer, count, btype in pool_info:
            if count > 0 and len(buffer) > 0:
                res = buffer.sample(min(count, len(buffer)), beta)
                if res:
                    samples, indices, w = res
                    all_samples.extend(samples)
                    all_indices.extend([(btype, idx) for idx in indices])
                    all_weights.extend(w)

        # If we didn't get enough samples (e.g. empty buffers), fill from whatever is available
        while len(all_samples) < batch_size:
            # Simple fallback: sample from any non-empty buffer
            valid_buffers = [(b, i) for i, b in enumerate([self.random_buffer, self.noisy_buffer, self.clean_buffer]) if len(b) > 0]
            if not valid_buffers: break
            
            buf, btype = valid_buffers[np.random.randint(len(valid_buffers))]
            res = buf.sample(1, beta)
            if res:
                samples, indices, w = res
                all_samples.extend(samples)
                all_indices.extend([(btype, idx) for idx in indices])
                all_weights.extend(w)

        return all_samples, all_indices, np.array(all_weights, dtype=np.float32)

    def update_priorities(self, indices_with_type, priorities):
        for (btype, idx), p in zip(indices_with_type, priorities):
            if btype == 0:
                self.random_buffer.update_priorities([idx], [p])
            elif btype == 1:
                self.noisy_buffer.update_priorities([idx], [p])
            elif btype == 2:
                self.clean_buffer.update_priorities([idx], [p])
