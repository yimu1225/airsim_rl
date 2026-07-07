import numpy as np


class ReplayBuffer:
    """Replay buffer storing noisy actor depth and clean critic depth sequences."""

    def __init__(self, max_size: int, seed=None):
        self.max_size = int(max_size)
        self.ptr = 0
        self.current_size = 0

        self.rng = np.random.default_rng(seed)

        self.base_buf = None
        self.depth_buf = None
        self.critic_priv_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.next_depth_buf = None
        self.next_critic_priv_buf = None
        self.done_buf = None

        self.critic_priv_shape = None

    @staticmethod
    def _as_priv(priv, target_shape=None):
        if priv is None:
            if target_shape is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros(target_shape, dtype=np.float32)
        arr = np.asarray(priv, dtype=np.float32)
        if target_shape is not None and arr.shape != tuple(target_shape):
            raise ValueError(f"critic_priv shape mismatch: expected {target_shape}, got {arr.shape}")
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
        critic_priv=None,
        next_critic_priv=None,
    ):
        critic_priv_arr = self._as_priv(critic_priv)
        next_critic_priv_arr = self._as_priv(next_critic_priv, target_shape=critic_priv_arr.shape)

        if self.base_buf is None:
            self.base_shape = np.asarray(base_state).shape
            self.depth_shape = np.asarray(depth).shape
            self.action_shape = np.asarray(action).shape
            self.critic_priv_shape = critic_priv_arr.shape

            self.base_buf = np.zeros((self.max_size, *self.base_shape), dtype=np.float32)
            self.depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16)
            self.critic_priv_buf = np.zeros((self.max_size, *self.critic_priv_shape), dtype=np.float16)
            self.action_buf = np.zeros((self.max_size, *self.action_shape), dtype=np.float32)
            self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float32)
            self.next_base_buf = np.zeros((self.max_size, *self.base_shape), dtype=np.float32)
            self.next_depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16)
            self.next_critic_priv_buf = np.zeros((self.max_size, *self.critic_priv_shape), dtype=np.float16)
            self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)
        else:
            critic_priv_arr = self._as_priv(critic_priv, target_shape=self.critic_priv_shape)
            next_critic_priv_arr = self._as_priv(next_critic_priv, target_shape=self.critic_priv_shape)

        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = np.asarray(depth, dtype=np.float16)
        self.critic_priv_buf[self.ptr] = critic_priv_arr
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.next_depth_buf[self.ptr] = np.asarray(next_depth, dtype=np.float16)
        self.next_critic_priv_buf[self.ptr] = next_critic_priv_arr
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int):
        if self.current_size == 0:
            return None
        ind = self.rng.integers(0, self.current_size, size=batch_size)

        return (
            self.base_buf[ind],
            self.depth_buf[ind].astype(np.float32),
            self.action_buf[ind],
            self.reward_buf[ind],
            self.next_base_buf[ind],
            self.next_depth_buf[ind].astype(np.float32),
            self.done_buf[ind],
            self.critic_priv_buf[ind].astype(np.float32),
            self.next_critic_priv_buf[ind].astype(np.float32),
        )

    def size(self) -> int:
        return self.current_size
