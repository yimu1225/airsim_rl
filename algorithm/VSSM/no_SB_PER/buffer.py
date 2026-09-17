import numpy as np


class ReplayBuffer:
    """Replay buffer for ST-Vim-SAC sequence observations."""

    def __init__(self, max_size: int, seed=None):
        self.max_size = int(max_size)
        self.ptr = 0
        self.current_size = 0
        self.rng = np.random.default_rng(seed)

        self.base_buf = None
        self.depth_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.next_depth_buf = None
        self.done_buf = None

    def add(self, base_state, depth, action, reward, next_base_state, next_depth, done):
        if self.base_buf is None:
            base_shape = np.asarray(base_state).shape
            depth_shape = np.asarray(depth).shape
            action_shape = np.asarray(action).shape

            self.base_buf = np.zeros((self.max_size, *base_shape), dtype=np.float32)
            self.depth_buf = np.zeros((self.max_size, *depth_shape), dtype=np.float16)
            self.action_buf = np.zeros((self.max_size, *action_shape), dtype=np.float32)
            self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float32)
            self.next_base_buf = np.zeros((self.max_size, *base_shape), dtype=np.float32)
            self.next_depth_buf = np.zeros((self.max_size, *depth_shape), dtype=np.float16)
            self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)

        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = np.asarray(depth, dtype=np.float16)
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.next_depth_buf[self.ptr] = np.asarray(next_depth, dtype=np.float16)
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int):
        if self.current_size == 0:
            return None
        indices = self.rng.integers(0, self.current_size, size=batch_size)
        return (
            self.base_buf[indices],
            self.depth_buf[indices].astype(np.float32),
            self.action_buf[indices],
            self.reward_buf[indices],
            self.next_base_buf[indices],
            self.next_depth_buf[indices].astype(np.float32),
            self.done_buf[indices],
        )

    def size(self) -> int:
        return self.current_size
