import numpy as np


class SequenceReplayBuffer:
    """Single-step replay buffer with explicit next-state transition."""

    def __init__(self, max_size: int, sequence_length: int):
        self.max_size = int(max_size)
        self.seq_len = int(sequence_length)
        self.ptr = 0
        self.size = 0

        self.base_buf = None
        self.depth_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.next_depth_buf = None
        self.done_buf = None
        self.collision_buf = None

    def add(self, base_state, depth, action, reward, next_base_state, next_depth, done, collision=0.0):
        if self.base_buf is None:
            self.base_shape = np.asarray(base_state).shape
            self.depth_shape = np.asarray(depth).shape
            self.action_shape = np.asarray(action).shape

            self.base_buf = np.zeros((self.max_size, *self.base_shape), dtype=np.float32)
            self.depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16)
            self.action_buf = np.zeros((self.max_size, *self.action_shape), dtype=np.float32)
            self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float32)
            self.next_base_buf = np.zeros((self.max_size, *self.base_shape), dtype=np.float32)
            self.next_depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16)
            self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)
            self.collision_buf = np.zeros((self.max_size, 1), dtype=np.float32)

        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = depth.astype(np.float16)
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.next_depth_buf[self.ptr] = next_depth.astype(np.float16)
        self.done_buf[self.ptr] = done
        self.collision_buf[self.ptr] = collision

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        if self.size == 0:
            return None
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.base_buf[ind],
            self.depth_buf[ind].astype(np.float32),
            self.action_buf[ind],
            self.reward_buf[ind],
            self.next_base_buf[ind],
            self.next_depth_buf[ind].astype(np.float32),
            self.done_buf[ind],
            self.collision_buf[ind]
        )

    def size_buffer(self):
        return self.size

    def size(self):
        return self.size
