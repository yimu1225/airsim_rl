import numpy as np


class ReplayBuffer:
    """LazyFrames Sequence Replay Buffer for ST-SVimTD3 - 环境返回单帧，采样时重建序列
    
    大幅降低内存使用：只存储单帧而非序列，通过索引关系在采样时重建序列。
    包含collision标志。done的处理：序列不跨越done边界，采样时过滤无效索引。
    """

    def __init__(self, max_size: int, sequence_length: int, seed=None):
        """
        Args:
            max_size: 缓冲区容量
            sequence_length: 序列长度（时间步数）
            seed: 随机种子
        """
        self.max_size = int(max_size)
        self.seq_len = int(sequence_length)
        self.ptr = 0
        self.current_size = 0

        self.rng = np.random.default_rng(seed)

        # 缓冲区（延迟初始化）- 只存储单帧
        self.base_buf = None
        self.frame_buf = None  # (max_size, 1, H, W) 单帧
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.done_buf = None
        self.collision_buf = None

    def add(self, base_state, depth, action, reward, next_base_state, next_depth, done, collision=0.0):
        """
        添加经验（环境返回单帧或序列，只存储最后一帧）
        """
        if self.base_buf is None:
            self._init_buffers(base_state, depth, action)

        # 只存储最新单帧（最后一帧）
        depth = np.asarray(depth)
        if depth.ndim == 4:
            frame = depth[-1].astype(np.float16)
        elif depth.ndim == 3 and depth.shape[0] > 1:
            frame = depth[-1:].astype(np.float16)
        elif depth.ndim == 2:
            frame = depth[np.newaxis, ...].astype(np.float16)
        else:
            frame = depth.astype(np.float16)

        self.base_buf[self.ptr] = base_state
        self.frame_buf[self.ptr] = frame
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.done_buf[self.ptr] = done
        self.collision_buf[self.ptr] = collision

        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def _init_buffers(self, base_state, depth, action):
        """延迟初始化缓冲区"""
        base_state = np.asarray(base_state)
        depth = np.asarray(depth)
        action = np.asarray(action)

        if depth.ndim == 4:
            frame_shape = (depth.shape[1], depth.shape[2], depth.shape[3])
        elif depth.ndim == 3:
            if depth.shape[0] > 1:
                frame_shape = (1, depth.shape[1], depth.shape[2])
            else:
                frame_shape = depth.shape
        elif depth.ndim == 2:
            frame_shape = (1, depth.shape[0], depth.shape[1])
        else:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")

        self.base_shape = base_state.shape if base_state.ndim > 0 else (1,)
        self.frame_shape = frame_shape
        self.action_shape = action.shape if action.ndim > 0 else (1,)

        self.base_buf = np.zeros((self.max_size, *self.base_shape), dtype=np.float32)
        self.next_base_buf = np.zeros((self.max_size, *self.base_shape), dtype=np.float32)
        self.action_buf = np.zeros((self.max_size, *self.action_shape), dtype=np.float32)
        self.frame_buf = np.zeros((self.max_size, *frame_shape), dtype=np.float16)
        self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)
        self.collision_buf = np.zeros((self.max_size, 1), dtype=np.float32)

    def _is_valid_index(self, idx: int) -> bool:
        """检查索引是否有效（序列不跨越done边界）"""
        if self.current_size < self.seq_len:
            return False
        
        for offset in range(1, self.seq_len):
            check_idx = (idx - offset) % self.max_size
            if self.done_buf[check_idx] > 0.5:
                return False
            if check_idx >= self.current_size and self.current_size < self.max_size:
                return False
        return True

    def _get_valid_indices(self, batch_size: int) -> np.ndarray:
        """获取有效的采样索引"""
        valid_indices = []
        attempts = 0
        max_attempts = batch_size * 100
        
        while len(valid_indices) < batch_size and attempts < max_attempts:
            idx = self.rng.integers(self.seq_len - 1, self.current_size)
            if self._is_valid_index(idx):
                valid_indices.append(idx)
            attempts += 1
        
        if len(valid_indices) < batch_size:
            if len(valid_indices) == 0:
                return None
            while len(valid_indices) < batch_size:
                valid_indices.append(valid_indices[self.rng.integers(len(valid_indices))])
        
        return np.array(valid_indices, dtype=np.int64)

    def _build_sequences(self, indices: np.ndarray) -> np.ndarray:
        """从索引重建序列"""
        batch_size = len(indices)
        C, H, W = self.frame_shape

        sequences = np.zeros((batch_size, self.seq_len, C, H, W), dtype=np.float32)

        for i, idx in enumerate(indices):
            for j in range(self.seq_len):
                offset = j - self.seq_len + 1
                frame_idx = int((idx + offset) % self.max_size)
                sequences[i, j] = self.frame_buf[frame_idx].astype(np.float32)

        return sequences

    def sample(self, batch_size: int):
        """采样经验，动态重建序列"""
        if self.current_size < self.seq_len:
            return None

        ind = self._get_valid_indices(batch_size)
        if ind is None:
            return None

        depth_sequences = self._build_sequences(ind)
        next_ind = (ind + 1) % self.max_size
        next_depth_sequences = self._build_sequences(next_ind)
        terminal_mask = self.done_buf[ind].reshape(-1) > 0.5
        next_depth_sequences[terminal_mask] = 0.0

        return (
            self.base_buf[ind],
            depth_sequences,
            self.action_buf[ind],
            self.reward_buf[ind],
            self.next_base_buf[ind],
            next_depth_sequences,
            self.done_buf[ind],
            self.collision_buf[ind]
        )

    def size_buffer(self):
        return self.current_size

    def size(self):
        return self.current_size
