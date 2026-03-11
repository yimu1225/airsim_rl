import numpy as np


class ReplayBuffer:
    """LazyFrames Replay Buffer - 环境返回单帧，采样时重建堆叠帧
    
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

        self.rng = np.random.default_rng(seed)

        # 缓冲区（延迟初始化）- 只存储单帧
        self.base_buf = None
        self.frame_buf = None  # (max_size, 1, H, W) 单帧
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.done_buf = None
        # 不存储next_frame_buf，通过索引计算

    def add(self, base_state, depth, action, reward, next_base_state, next_depth, done):
        """
        添加经验（环境返回单帧或堆叠帧，只存储最后一帧）
        
        Args:
            base_state: 基础状态
            depth: 深度图像 - 可以是单帧(1,H,W)或多帧堆叠(n_frames,H,W)
            action: 动作
            reward: 奖励
            next_base_state: 下一基础状态
            next_depth: 下一深度图像 - 可以是单帧或多帧堆叠
            done: 是否结束
        """
        if self.base_buf is None:
            self._init_buffers(base_state, depth, action)

        # 只存储最新单帧（最后一帧）
        # 如果depth是堆叠的，取最后一帧
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[0] > 1:
            # 多帧堆叠，取最后一帧
            frame = depth[-1:].astype(np.float16)  # (1, H, W)
        elif depth.ndim == 2:
            # (H, W) -> (1, H, W)
            frame = depth[np.newaxis, ...].astype(np.float16)
        else:
            # 已经是单帧 (1, H, W)
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

        # depth可能是堆叠帧，推断单帧形状
        if depth.ndim == 3:
            if depth.shape[0] > 1:
                # 堆叠帧，单帧形状为 (1, H, W)
                frame_shape = (1, depth.shape[1], depth.shape[2])
            else:
                # 单帧 (1, H, W)
                frame_shape = depth.shape
        elif depth.ndim == 2:
            # (H, W) -> (1, H, W)
            frame_shape = (1, depth.shape[0], depth.shape[1])
        else:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")

        self.base_dim = base_state.shape[0] if base_state.ndim > 0 else 1
        self.frame_shape = frame_shape  # (1, H, W)
        self.action_dim = action.shape[0] if action.ndim > 0 else 1

        # 基础状态和动作保持float32
        self.base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
        self.next_base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.max_size, self.action_dim), dtype=np.float32)

        # 只存储单帧 float16节省内存
        self.frame_buf = np.zeros((self.max_size, *frame_shape), dtype=np.float16)

        # 奖励和done标志
        self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float16)
        self.done_buf = np.zeros((self.max_size, 1), dtype=np.float16)

    def _is_valid_index(self, idx: int) -> bool:
        """
        检查索引是否有效（不跨越done边界）
        
        有效条件：从idx-n_frames+1到idx之间没有done标志
        即：序列[idx-n_frames+1, ..., idx]都在同一个episode内
        """
        if self.current_size < self.n_frames:
            return False
        
        # 检查序列中是否有done（除了idx本身）
        for offset in range(1, self.n_frames):
            check_idx = (idx - offset) % self.max_size
            # 如果check_idx的done为True，则check_idx+1是新episode的开始
            # 这意味着序列会跨越done边界
            if self.done_buf[check_idx] > 0.5:
                return False
            # 如果已经检查到buffer的开头，停止
            if check_idx >= self.current_size:
                return False
        return True

    def _get_valid_indices(self, batch_size: int) -> np.ndarray:
        """
        获取有效的采样索引（不跨越done边界）
        """
        valid_indices = []
        attempts = 0
        max_attempts = batch_size * 100  # 防止无限循环
        
        while len(valid_indices) < batch_size and attempts < max_attempts:
            idx = self.rng.integers(self.n_frames - 1, self.current_size)
            if self._is_valid_index(idx):
                valid_indices.append(idx)
            attempts += 1
        
        if len(valid_indices) < batch_size:
            # 有效索引不足，用最后一个有效索引填充（或者随机重复）
            if len(valid_indices) == 0:
                # 没有有效索引，返回None
                return None
            while len(valid_indices) < batch_size:
                valid_indices.append(valid_indices[self.rng.integers(len(valid_indices))])
        
        return np.array(valid_indices, dtype=np.int64)

    def _build_stacked_frames(self, indices: np.ndarray) -> np.ndarray:
        """
        从索引重建堆叠帧
        
        Args:
            indices: 采样索引 (batch_size,)，假设已经验证有效
        
        Returns:
            stacked: (batch_size, n_frames, H, W)
        """
        batch_size = len(indices)
        _, H, W = self.frame_shape

        # 输出: (batch, n_frames, H, W)
        stacked = np.zeros((batch_size, self.n_frames, H, W), dtype=np.float32)

        for i, idx in enumerate(indices):
            for j in range(self.n_frames):
                # j=0是最早的帧，j=n_frames-1是最新的帧
                offset = j - self.n_frames + 1  # 负数偏移
                frame_idx = int((idx + offset) % self.max_size)
                stacked[i, j] = self.frame_buf[frame_idx][0].astype(np.float32)

        return stacked

    def sample(self, batch_size: int):
        """
        采样经验，动态重建堆叠帧
        只采样不跨越done边界的有效transition
        
        Returns:
            base_states: (batch, base_dim)
            depths: (batch, n_frames, H, W) - 重建的堆叠帧
            actions: (batch, action_dim)
            rewards: (batch, 1)
            next_base_states: (batch, base_dim)
            next_depths: (batch, n_frames, H, W) - 重建的下一状态堆叠帧
            dones: (batch, 1)
        """
        if self.current_size < self.n_frames:
            return None

        # 获取有效索引（不跨越done边界）
        ind = self._get_valid_indices(batch_size)
        if ind is None:
            return None

        # 重建当前状态堆叠帧
        depths = self._build_stacked_frames(ind)

        # 重建下一状态堆叠帧
        next_ind = (ind + 1) % self.max_size
        next_depths = self._build_stacked_frames(next_ind)
        terminal_mask = self.done_buf[ind].reshape(-1) > 0.5
        next_depths[terminal_mask] = 0.0

        return (
            self.base_buf[ind],
            depths,  # (batch, n_frames, H, W)
            self.action_buf[ind],
            self.reward_buf[ind].astype(np.float32),
            self.next_base_buf[ind],
            next_depths,  # (batch, n_frames, H, W)
            self.done_buf[ind].astype(np.float32)
        )

    def size(self) -> int:
        return self.current_size
