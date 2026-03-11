import numpy as np


class ReplayBuffer:
    """LazyFrames Replay Buffer for GAM-Mamba-TD3 - 环境返回单帧，采样时重建堆叠帧
    
    大幅降低内存使用：只存储单帧而非堆叠帧，通过索引关系在采样时重建序列。
    done的处理：序列不跨越done边界，采样时过滤无效索引
    """

    def __init__(self, max_size: int, n_frames: int = 4, seed=None):
        """
        Args:
            max_size: 缓冲区容量
            n_frames: 堆叠帧数（默认4帧）
            seed: 随机种子
        """
        self.max_size = int(max_size)
        self.n_frames = n_frames
        self.ptr = 0
        self.current_size = 0

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # 缓冲区（延迟初始化）- 只存储单帧
        self.base_buf = None
        self.frame_buf = None  # (max_size, 1, H, W) 单帧
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.done_buf = None

    def add(self, base_state, depth, action, reward, next_base_state, next_depth, done):
        """
        添加经验（环境返回单帧或堆叠帧，只存储最后一帧）
        """
        if self.base_buf is None:
            self._init_buffers(base_state, depth, action)

        # 只存储最新单帧（最后一帧）
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[0] > 1:
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

        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def _init_buffers(self, base_state, depth, action):
        """延迟初始化缓冲区"""
        base_state = np.asarray(base_state)
        depth = np.asarray(depth)
        action = np.asarray(action)

        if depth.ndim == 3:
            if depth.shape[0] > 1:
                frame_shape = (1, depth.shape[1], depth.shape[2])
            else:
                frame_shape = depth.shape
        elif depth.ndim == 2:
            frame_shape = (1, depth.shape[0], depth.shape[1])
        else:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")

        self.base_dim = base_state.shape[0] if base_state.ndim > 0 else 1
        self.frame_shape = frame_shape
        self.action_dim = action.shape[0] if action.ndim > 0 else 1

        self.base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
        self.next_base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.frame_buf = np.zeros((self.max_size, *frame_shape), dtype=np.float16)
        self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float16)
        self.done_buf = np.zeros((self.max_size, 1), dtype=np.float16)

    def _is_valid_index(self, idx: int) -> bool:
        """检查索引是否有效（不跨越done边界）"""
        if self.current_size < self.n_frames:
            return False
        
        for offset in range(1, self.n_frames):
            check_idx = (idx - offset) % self.max_size
            if self.done_buf[check_idx] > 0.5:
                return False
            if check_idx >= self.current_size:
                return False
        return True

    def _get_valid_indices(self, batch_size: int) -> np.ndarray:
        """获取有效的采样索引"""
        valid_indices = []
        attempts = 0
        max_attempts = batch_size * 100
        
        while len(valid_indices) < batch_size and attempts < max_attempts:
            idx = self.rng.integers(self.n_frames - 1, self.current_size)
            if self._is_valid_index(idx):
                valid_indices.append(idx)
            attempts += 1
        
        if len(valid_indices) < batch_size:
            if len(valid_indices) == 0:
                return None
            while len(valid_indices) < batch_size:
                valid_indices.append(valid_indices[self.rng.integers(len(valid_indices))])
        
        return np.array(valid_indices, dtype=np.int64)

    def _build_stacked_frames(self, indices: np.ndarray) -> np.ndarray:
        """从索引重建堆叠帧"""
        batch_size = len(indices)
        _, H, W = self.frame_shape

        stacked = np.zeros((batch_size, self.n_frames, H, W), dtype=np.float32)

        for i, idx in enumerate(indices):
            for j in range(self.n_frames):
                offset = j - self.n_frames + 1
                frame_idx = int((idx + offset) % self.max_size)
                stacked[i, j] = self.frame_buf[frame_idx][0].astype(np.float32)

        return stacked

    def sample(self, batch_size: int):
        """采样经验，动态重建堆叠帧"""
        if self.current_size < self.n_frames:
            return None

        ind = self._get_valid_indices(batch_size)
        if ind is None:
            return None

        depths = self._build_stacked_frames(ind)
        next_ind = (ind + 1) % self.max_size
        next_depths = self._build_stacked_frames(next_ind)
        terminal_mask = self.done_buf[ind].reshape(-1) > 0.5
        next_depths[terminal_mask] = 0.0

        return (
            self.base_buf[ind],
            depths,
            self.action_buf[ind],
            self.reward_buf[ind].astype(np.float32),
            self.next_base_buf[ind],
            next_depths,
            self.done_buf[ind].astype(np.float32)
        )

    def size(self) -> int:
        return self.current_size
