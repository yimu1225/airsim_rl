import numpy as np

from .disk_replay import DiskArrayFactory, zeros


class ReplayBuffer:
    """PL replay buffer; optionally stores arrays as temporary memmaps on disk."""

    def __init__(self, max_size: int, sequence_length: int | None = None, seed=None, disk_dir=None):
        self.max_size = int(max_size)
        self.seq_len = None if sequence_length is None else int(sequence_length)
        self.ptr = 0
        self.current_size = 0

        self.rng = np.random.default_rng(seed)
        self.disk_factory = DiskArrayFactory(root=disk_dir, prefix="pl_replay") if disk_dir else None

        self.base_buf = None
        self.depth_buf = None
        self.critic_priv_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.next_depth_buf = None
        self.next_critic_priv_buf = None
        self.done_buf = None

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

    def _alloc(self, shape, dtype, name):
        return zeros(shape, dtype=dtype, factory=self.disk_factory, name=name)

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
        critic_priv_flat = self._flatten_priv(critic_priv)
        next_critic_priv_flat = self._flatten_priv(next_critic_priv, target_dim=critic_priv_flat.size)

        if self.base_buf is None:
            self.base_shape = np.asarray(base_state).shape
            self.depth_shape = np.asarray(depth).shape
            self.action_shape = np.asarray(action).shape
            self.critic_priv_dim = int(critic_priv_flat.size)

            self.base_buf = self._alloc((self.max_size, *self.base_shape), np.float32, "base")
            self.depth_buf = self._alloc((self.max_size, *self.depth_shape), np.float16, "depth")
            self.critic_priv_buf = self._alloc((self.max_size, self.critic_priv_dim), np.float32, "critic_priv")
            self.action_buf = self._alloc((self.max_size, *self.action_shape), np.float32, "action")
            self.reward_buf = self._alloc((self.max_size, 1), np.float32, "reward")
            self.next_base_buf = self._alloc((self.max_size, *self.base_shape), np.float32, "next_base")
            self.next_depth_buf = self._alloc((self.max_size, *self.depth_shape), np.float16, "next_depth")
            self.next_critic_priv_buf = self._alloc(
                (self.max_size, self.critic_priv_dim), np.float32, "next_critic_priv"
            )
            self.done_buf = self._alloc((self.max_size, 1), np.float32, "done")
        else:
            critic_priv_flat = self._flatten_priv(critic_priv, target_dim=self.critic_priv_dim)
            next_critic_priv_flat = self._flatten_priv(next_critic_priv, target_dim=self.critic_priv_dim)

        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = np.asarray(depth, dtype=np.float16)
        self.critic_priv_buf[self.ptr] = critic_priv_flat
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.next_depth_buf[self.ptr] = np.asarray(next_depth, dtype=np.float16)
        self.next_critic_priv_buf[self.ptr] = next_critic_priv_flat
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
            self.reward_buf[ind].astype(np.float32),
            self.next_base_buf[ind],
            self.next_depth_buf[ind].astype(np.float32),
            self.done_buf[ind].astype(np.float32),
            self.critic_priv_buf[ind],
            self.next_critic_priv_buf[ind],
        )

    def size_buffer(self):
        return self.current_size

    def size(self) -> int:
        return self.current_size
