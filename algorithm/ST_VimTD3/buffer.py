import numpy as np


class SequenceReplayBuffer:
    """Replay buffer that stores and samples sequences of transitions for POMDP.
    Optimized with pre-allocated numpy arrays for faster sampling.
    """

    def __init__(self, max_size: int, sequence_length: int):
        self.max_size = int(max_size)
        self.seq_len = int(sequence_length)
        self.ptr = 0
        self.size = 0

        self.base_buf = None
        self.depth_buf = None
        self.gray_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.done_buf = None

    def add(self, base_state, depth, action, reward, done):
        if self.base_buf is None:
            self.base_dim = base_state.shape[0]
            self.depth_shape = depth.shape
            self.action_dim = action.shape[0]

            self.base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
            self.depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16)
            self.action_buf = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
            self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float32)
            self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)

        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = depth.astype(np.float16)
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        if self.size <= self.seq_len:
            return None

        valid_max_idx = self.size - self.seq_len
        batch_indices = []

        while len(batch_indices) < batch_size:
            remaining = batch_size - len(batch_indices)
            candidates = np.random.randint(0, valid_max_idx, size=remaining * 2)

            start_idxs = (self.ptr - self.size + candidates) % self.max_size
            is_valid = np.ones(len(candidates), dtype=bool)

            for k in range(self.seq_len - 1):
                check_idxs = (start_idxs + k) % self.max_size
                dones = self.done_buf[check_idxs].squeeze()
                is_valid = is_valid & (dones == 0)

            valid_candidates = candidates[is_valid]
            valid_start_idxs = (self.ptr - self.size + valid_candidates) % self.max_size
            batch_indices.extend(valid_start_idxs[:remaining])

        batch_indices = np.array(batch_indices[:batch_size])
        seq_indices = (batch_indices[:, None] + np.arange(self.seq_len + 1)) % self.max_size

        bases = self.base_buf[seq_indices]
        depths = self.depth_buf[seq_indices].astype(np.float32)
        actions = self.action_buf[seq_indices]
        rewards = self.reward_buf[seq_indices]
        dones = self.done_buf[seq_indices]

        batch_base = bases[:, :-1]
        batch_depth = depths[:, :-1]
        batch_action = actions[:, :-1]
        batch_reward = rewards[:, :-1]
        batch_done = dones[:, :-1]

        batch_next_base = bases[:, 1:]
        batch_next_depth = depths[:, 1:]

        return (
            batch_base,
            batch_depth,
            batch_action,
            batch_reward,
            batch_next_base,
            batch_next_depth,
            batch_done
        )

    def size_buffer(self):
        return self.size

    def size(self):
        return self.size
