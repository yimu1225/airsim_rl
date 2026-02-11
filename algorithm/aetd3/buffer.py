import numpy as np


class ReplayBuffer:
    """Replay buffer for AETD3 (base_state + depth) with pre-allocated numpy arrays."""

    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.ptr = 0
        self.current_size = 0

        self.base_buf = None
        self.depth_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.next_depth_buf = None
        self.done_buf = None

    def add(self, base_state, depth, action, reward, next_base_state, next_depth, done):
        if self.base_buf is None:
            self.base_dim = base_state.shape[0]
            self.depth_shape = depth.shape
            self.action_dim = action.shape[0]

            # 基础状态和动作保持float32以保持数值精度
            self.base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
            self.next_base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
            self.action_buf = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
            
            # 深度图像使用float16节省50%内存
            self.depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16)
            self.next_depth_buf = np.zeros((self.max_size, *self.depth_shape), dtype=np.float16)
            
            # 奖励和done标志使用float16进一步节省内存
            self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float16)
            self.done_buf = np.zeros((self.max_size, 1), dtype=np.float16)

        self.base_buf[self.ptr] = base_state
        self.depth_buf[self.ptr] = depth.astype(np.float16)
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.next_depth_buf[self.ptr] = next_depth.astype(np.float16)
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int):
        ind = np.random.randint(0, self.current_size, size=batch_size)

        return (
            self.base_buf[ind],
            self.depth_buf[ind].astype(np.float32),  # 转换回float32用于训练
            self.action_buf[ind],
            self.reward_buf[ind].astype(np.float32),  # 转换回float32用于训练
            self.next_base_buf[ind],
            self.next_depth_buf[ind].astype(np.float32),  # 转换回float32用于训练
            self.done_buf[ind].astype(np.float32)  # 转换回float32用于训练
        )

    def size(self) -> int:
        return self.current_size
