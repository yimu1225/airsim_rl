import numpy as np


class ReplayBuffer:
    """Replay buffer with pre-allocated numpy arrays.

    Each instance keeps its own RNG; a seed may be passed for determinism.
    """

    def __init__(self, max_size: int, seed=None):
        self.max_size = int(max_size)
        self.ptr = 0
        self.current_size = 0

        self.rng = np.random.default_rng(seed)

        self.base_buf = None
        self.depth_buf = None
        self.critic_depth_buf = None
        self.critic_priv_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.next_depth_buf = None
        self.next_critic_depth_buf = None
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

    @staticmethod
    def _encode_depth(depth):
        return np.clip(np.rint(np.asarray(depth, dtype=np.float32)), 0.0, 255.0).astype(np.uint8)

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
        critic_depth=None,
        next_critic_depth=None,
    ):
        if critic_depth is None:
            critic_depth = depth
        if next_critic_depth is None:
            next_critic_depth = next_depth

        critic_priv_flat = self._flatten_priv(critic_priv)
        next_critic_priv_flat = self._flatten_priv(next_critic_priv, target_dim=critic_priv_flat.size)

        if self.base_buf is None:
            self.base_dim = base_state.shape[0]
            self.depth_shape = depth.shape
            self.critic_depth_shape = np.asarray(critic_depth).shape
            self.action_dim = action.shape[0]
            self.critic_priv_dim = int(critic_priv_flat.size)

            # 基础状态和动作保持float32以保持数值精度
            self.base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
            self.next_base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
            self.action_buf = np.zeros((self.max_size, self.action_dim), dtype=np.float32)

            # 深度图像使用uint8进一步节省内存
            self.depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.uint8)
            self.next_depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.uint8)
            self.critic_depth_buf = np.zeros((self.max_size, *self.critic_depth_shape), dtype=np.uint8)
            self.next_critic_depth_buf = np.zeros((self.max_size, *self.critic_depth_shape), dtype=np.uint8)
            self.critic_priv_buf = np.zeros((self.max_size, self.critic_priv_dim), dtype=np.float32)
            self.next_critic_priv_buf = np.zeros((self.max_size, self.critic_priv_dim), dtype=np.float32)

            # 奖励和done标志使用float16进一步节省内存
            self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float16)
            self.done_buf = np.zeros((self.max_size, 1), dtype=np.float16)
        else:
            critic_priv_flat = self._flatten_priv(critic_priv, target_dim=self.critic_priv_dim)
            next_critic_priv_flat = self._flatten_priv(next_critic_priv, target_dim=self.critic_priv_dim)

        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = self._encode_depth(depth)
        self.critic_depth_buf[self.ptr] = self._encode_depth(critic_depth)
        self.critic_priv_buf[self.ptr] = critic_priv_flat
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.next_depth_buf[self.ptr] = self._encode_depth(next_depth)
        self.next_critic_depth_buf[self.ptr] = self._encode_depth(next_critic_depth)
        self.next_critic_priv_buf[self.ptr] = next_critic_priv_flat
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int):
        ind = self.rng.integers(0, self.current_size, size=batch_size)

        return (
            self.base_buf[ind],
            self.depth_buf[ind].astype(np.float32),  # 转换回float32用于训练
            self.critic_depth_buf[ind].astype(np.float32),
            self.critic_priv_buf[ind],
            self.action_buf[ind],
            self.reward_buf[ind].astype(np.float32),  # 转换回float32用于训练
            self.next_base_buf[ind],
            self.next_depth_buf[ind].astype(np.float32),  # 转换回float32用于训练
            self.next_critic_depth_buf[ind].astype(np.float32),
            self.next_critic_priv_buf[ind],
            self.done_buf[ind].astype(np.float32)  # 转换回float32用于训练
        )

    def size(self) -> int:
        return self.current_size
