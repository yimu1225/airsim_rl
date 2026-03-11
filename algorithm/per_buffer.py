import numpy as np
import torch


class PrioritizedReplayBuffer:
    """LazyFrames Prioritized Replay Buffer - 环境返回单帧，采样时重建堆叠帧
    
    大幅降低内存使用：只存储单帧而非堆叠帧，通过索引关系在采样时重建序列。
    done的处理：序列不跨越done边界，采样时过滤无效索引
    """
    
    def __init__(self, capacity, n_frames: int = 4, alpha=0.6, eps=1e-6, seed=None):
        """
        Args:
            capacity: 缓冲区容量
            n_frames: 堆叠帧数（默认4帧）
            alpha: 优先级指数
            eps: 优先级epsilon
            seed: 随机种子
        """
        self.capacity = capacity
        self.n_frames = n_frames
        self.alpha = alpha
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        
        # 使用numpy数组替代列表以提高效率
        self.base_buf = None
        self.frame_buf = None  # 只存储单帧
        self.action_buf = None
        self.reward_buf = None
        self.next_base_buf = None
        self.done_buf = None
        
        self.priorities = None
        self.pos = 0
        self.current_size = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """
        添加经验
        
        Args:
            state: (base_state, depth)
            action: 动作
            reward: 奖励
            next_state: (next_base_state, next_depth)
            done: 是否结束
        """
        base_state, depth = state
        next_base_state, next_depth = next_state
        
        # 转换为numpy并只存储单帧
        base_state = np.asarray(base_state)
        depth = np.asarray(depth)
        action = np.asarray(action)
        next_base_state = np.asarray(next_base_state)
        next_depth = np.asarray(next_depth)
        
        # 只取最新单帧
        if depth.ndim == 3 and depth.shape[0] > 1:
            frame = depth[-1:].astype(np.float16)
        elif depth.ndim == 2:
            frame = depth[np.newaxis, ...].astype(np.float16)
        else:
            frame = depth.astype(np.float16)
        
        if self.base_buf is None:
            self._init_buffers(base_state, frame, action)
        
        self.base_buf[self.pos] = base_state
        self.frame_buf[self.pos] = frame
        self.action_buf[self.pos] = action
        self.reward_buf[self.pos] = reward
        self.next_base_buf[self.pos] = next_base_state
        self.done_buf[self.pos] = done
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def _init_buffers(self, base_state, frame, action):
        """延迟初始化缓冲区"""
        self.base_dim = base_state.shape[0] if base_state.ndim > 0 else 1
        self.frame_shape = frame.shape  # (1, H, W)
        self.action_dim = action.shape[0] if action.ndim > 0 else 1
        
        self.base_buf = np.zeros((self.capacity, self.base_dim), dtype=np.float32)
        self.frame_buf = np.zeros((self.capacity, *self.frame_shape), dtype=np.float16)
        self.action_buf = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_base_buf = np.zeros((self.capacity, self.base_dim), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

    def _is_valid_index(self, idx: int) -> bool:
        """检查索引是否有效（不跨越done边界）"""
        if self.current_size < self.n_frames:
            return False
        
        for offset in range(1, self.n_frames):
            check_idx = (idx - offset) % self.capacity
            if self.done_buf[check_idx] > 0.5:
                return False
            if check_idx >= self.current_size and self.current_size < self.capacity:
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
                return np.array([], dtype=np.int64)
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
                frame_idx = int((idx + offset) % self.capacity)
                stacked[i, j] = self.frame_buf[frame_idx][0].astype(np.float32)

        return stacked

    def sample(self, batch_size, beta=0.4):
        """采样经验，使用PER和LazyFrames重建"""
        if self.current_size < self.n_frames:
            return None, None, None
        
        # 获取有效索引
        valid_indices = self._get_valid_indices(batch_size)
        if len(valid_indices) == 0:
            return None, None, None
        
        # 从有效索引中根据优先级采样
        priorities = self.priorities[valid_indices]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        sampled_indices = self.rng.choice(len(valid_indices), min(batch_size, len(valid_indices)), p=probs)
        indices = valid_indices[sampled_indices]
        
        # 重要性采样权重
        total = len(valid_indices)
        weights = (total * probs[sampled_indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # 重建堆叠帧
        depths = self._build_stacked_frames(indices)
        next_ind = (indices + 1) % self.capacity
        next_depths = self._build_stacked_frames(next_ind)
        terminal_mask = self.done_buf[indices].reshape(-1) > 0.5
        next_depths[terminal_mask] = 0.0

        # 构造samples
        samples = []
        for i, idx in enumerate(indices):
            sample = (
                self.base_buf[idx],
                depths[i],  # (n_frames, H, W)
                self.action_buf[idx],
                self.reward_buf[idx],
                self.next_base_buf[idx],
                next_depths[i],  # (n_frames, H, W)
                self.done_buf[idx]
            )
            samples.append(sample)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps
            self.max_priority = max(self.max_priority, priority + self.eps)

    def __len__(self):
        return self.current_size

    def size(self):
        return self.current_size
