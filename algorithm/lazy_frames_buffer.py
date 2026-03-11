"""
LazyFrames Replay Buffer Utilities

通过索引关系在采样时重建序列，显著降低内存使用。
环境只返回最新单帧，buffer只存储单帧，采样时通过索引重建序列。

关键设计：
1. 环境返回单帧 (C, H, W)
2. Buffer只存储单帧，不存储序列
3. 采样时通过索引关系动态重建序列
4. 对于done的处理：如果跨越episode边界，使用零填充或复制第一帧
"""

import numpy as np
from typing import Tuple, Optional, List


class LazyFrames:
    """
    通过索引引用底层帧存储，采样时动态重建序列。
    避免存储重复的序列数据，大幅降低内存使用。
    """
    
    def __init__(self, frame_buffer: np.ndarray, indices: np.ndarray, 
                 seq_len: int, frame_shape: tuple):
        """
        Args:
            frame_buffer: 底层帧存储数组 (max_size, C, H, W)
            indices: 采样索引数组 (batch_size,)
            seq_len: 序列长度
            frame_shape: 单帧形状 (C, H, W)
        """
        self.frame_buffer = frame_buffer
        self.indices = indices
        self.seq_len = seq_len
        self.frame_shape = frame_shape
        self.batch_size = len(indices)
        
    def reconstruct(self) -> np.ndarray:
        """
        重建序列帧 (batch, seq_len, C, H, W)
        注意：这里假设帧数据已经是堆叠好的 (n_frames*1, H, W) -> (C, H, W)
        """
        # 对于每个采样索引，重建其前seq_len帧的序列
        sequences = np.zeros((self.batch_size, self.seq_len, *self.frame_shape), dtype=np.float32)
        
        for i, idx in enumerate(self.indices):
            for j in range(self.seq_len):
                frame_idx = (idx - self.seq_len + 1 + j) % len(self.frame_buffer)
                sequences[i, j] = self.frame_buffer[frame_idx].astype(np.float32)
        
        return sequences
    
    def __repr__(self):
        return f"LazyFrames(batch={self.batch_size}, seq_len={self.seq_len}, shape={self.frame_shape})"


def build_sequences_from_indices(frame_buffer: np.ndarray, indices: np.ndarray,
                                  seq_len: int, done_buffer: Optional[np.ndarray] = None,
                                  padding_mode: str = 'zero') -> np.ndarray:
    """
    从索引构建序列，处理done边界
    
    Args:
        frame_buffer: 帧存储 (max_size, C, H, W)
        indices: 采样索引 (batch_size,)
        seq_len: 序列长度
        done_buffer: done标志存储 (max_size,)，用于检测episode边界
        padding_mode: 跨越边界时的填充模式 ('zero', 'repeat', 'none')
    
    Returns:
        sequences: (batch_size, seq_len, C, H, W)
    """
    batch_size = len(indices)
    frame_shape = frame_buffer.shape[1:]  # (C, H, W)
    sequences = np.zeros((batch_size, seq_len, *frame_shape), dtype=np.float32)
    
    for i, idx in enumerate(indices):
        for j in range(seq_len):
            # 计算在序列中的位置（从最早到最晚）
            offset = j - seq_len + 1  # 负数：-seq_len+1, ..., -1, 0
            frame_idx = int((idx + offset) % len(frame_buffer))
            
            # 检查是否跨越episode边界
            if done_buffer is not None and padding_mode != 'none':
                # 检查从frame_idx到idx之间是否有done
                has_done = False
                check_idx = frame_idx
                while check_idx != idx:
                    if done_buffer[check_idx] > 0.5:
                        has_done = True
                        break
                    check_idx = (check_idx + 1) % len(done_buffer)
                
                if has_done:
                    if padding_mode == 'zero':
                        sequences[i, j] = 0
                        continue
                    elif padding_mode == 'repeat':
                        # 使用序列的第一帧填充
                        frame_idx = int((idx - seq_len + 1) % len(frame_buffer))
            
            sequences[i, j] = frame_buffer[frame_idx].astype(np.float32)
    
    return sequences


def build_stacked_frames(frame_buffer: np.ndarray, indices: np.ndarray,
                         n_frames: int, done_buffer: Optional[np.ndarray] = None,
                         padding_mode: str = 'zero') -> np.ndarray:
    """
    构建堆叠帧 (用于非序列型算法，如标准TD3)
    将n_frames帧堆叠到通道维度
    
    Args:
        frame_buffer: 帧存储 (max_size, 1, H, W) - 单帧
        indices: 采样索引 (batch_size,)
        n_frames: 堆叠帧数
        done_buffer: done标志存储
        padding_mode: 填充模式
    
    Returns:
        stacked: (batch_size, n_frames, H, W)
    """
    batch_size = len(indices)
    frame_shape = frame_buffer.shape[1:]  # (1, H, W)
    _, H, W = frame_shape
    
    # 输出形状: (batch, n_frames, H, W)
    stacked = np.zeros((batch_size, n_frames, H, W), dtype=np.float32)
    
    for i, idx in enumerate(indices):
        for j in range(n_frames):
            offset = j - n_frames + 1
            frame_idx = int((idx + offset) % len(frame_buffer))
            
            # 检查done边界
            if done_buffer is not None and padding_mode != 'none':
                has_done = False
                check_idx = frame_idx
                while check_idx != idx:
                    if done_buffer[check_idx] > 0.5:
                        has_done = True
                        break
                    check_idx = (check_idx + 1) % len(done_buffer)
                
                if has_done:
                    if padding_mode == 'zero':
                        continue  # 保持为0
                    elif padding_mode == 'repeat':
                        frame_idx = int((idx - n_frames + 1) % len(frame_buffer))
            
            stacked[i, j] = frame_buffer[frame_idx][0].astype(np.float32)  # [0] because (1, H, W)
    
    return stacked


class LazyReplayBuffer:
    """
    基于LazyFrames的回放缓冲区基类
    环境返回单帧，buffer只存储单帧，采样时重建序列或堆叠帧
    """
    
    def __init__(self, max_size: int, n_frames: int = 4, seed=None):
        """
        Args:
            max_size: 缓冲区最大容量
            n_frames: 堆叠帧数（用于构建状态）
            seed: 随机种子
        """
        self.max_size = int(max_size)
        self.n_frames = n_frames
        self.ptr = 0
        self.current_size = 0
        
        self.rng = np.random.default_rng(seed)
        
        # 缓冲区（延迟初始化）
        self.frame_buf = None  # 只存储单帧 (max_size, 1, H, W)
        self.base_buf = None
        self.action_buf = None
        self.reward_buf = None
        self.done_buf = None
        
        # 不存储next_frame，通过索引计算
        self.next_base_buf = None
    
    def add(self, base_state, frame, action, reward, next_base_state, done):
        """
        添加单帧经验
        
        Args:
            base_state: 基础状态向量
            frame: 单帧图像 (1, H, W)
            action: 动作
            reward: 奖励
            next_base_state: 下一基础状态
            done: 是否结束
        """
        if self.frame_buf is None:
            self._init_buffers(base_state, frame, action)
        
        # 确保frame是单帧 (1, H, W)
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = frame[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        self.frame_buf[self.ptr] = frame.astype(np.float16)
        self.base_buf[self.ptr] = base_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_base_buf[self.ptr] = next_base_state
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)
    
    def _init_buffers(self, base_state, frame, action):
        """延迟初始化缓冲区"""
        base_state = np.asarray(base_state)
        frame = np.asarray(frame)
        action = np.asarray(action)
        
        # frame形状: (1, H, W) - 单帧
        if frame.ndim == 2:
            frame_shape = (1, *frame.shape)
        else:
            frame_shape = frame.shape  # (1, H, W)
        
        self.base_dim = base_state.shape[0] if base_state.ndim > 0 else 1
        self.frame_shape = frame_shape  # (1, H, W)
        self.action_dim = action.shape[0] if action.ndim > 0 else 1
        
        # 只存储单帧
        self.frame_buf = np.zeros((self.max_size, *frame_shape), dtype=np.float16)
        self.base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
        self.next_base_buf = np.zeros((self.max_size, self.base_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((self.max_size, 1), dtype=np.float16)
        self.done_buf = np.zeros((self.max_size, 1), dtype=np.float16)

    def _is_valid_index(self, idx: int, history_len: int) -> bool:
        """检查索引是否有效：历史窗口不跨越done边界，且不访问未写入区域。"""
        if self.current_size < history_len:
            return False

        for offset in range(1, history_len):
            check_idx = (idx - offset) % self.max_size
            if self.done_buf[check_idx] > 0.5:
                return False
            if check_idx >= self.current_size and self.current_size < self.max_size:
                return False
        return True

    def _get_valid_indices(self, batch_size: int, history_len: int) -> Optional[np.ndarray]:
        """采样有效索引：允许采样终止步本身，但历史窗口不能跨越done边界。"""
        valid_indices = []
        attempts = 0
        max_attempts = batch_size * 100

        while len(valid_indices) < batch_size and attempts < max_attempts:
            idx = self.rng.integers(history_len - 1, self.current_size)
            if self._is_valid_index(idx, history_len):
                valid_indices.append(idx)
            attempts += 1

        if len(valid_indices) == 0:
            return None

        while len(valid_indices) < batch_size:
            valid_indices.append(valid_indices[self.rng.integers(len(valid_indices))])

        return np.array(valid_indices, dtype=np.int64)
    
    def sample(self, batch_size: int, get_sequences: bool = False):
        """
        采样经验
        
        Args:
            batch_size: 批量大小
            get_sequences: 是否返回序列 (batch, n_frames, H, W) 否则返回单帧堆叠
        
        Returns:
            base_states, frames/sequences, actions, rewards, next_base_states, next_frames/next_sequences, dones
        """
        raise NotImplementedError
    
    def size(self) -> int:
        return self.current_size
    
    def __len__(self):
        return self.current_size


class LazySequenceReplayBuffer(LazyReplayBuffer):
    """
    序列型LazyFrames缓冲区
    返回序列 (batch, seq_len, C, H, W) 供序列模型使用
    """
    
    def __init__(self, max_size: int, seq_len: int, seed=None):
        super().__init__(max_size, n_frames=seq_len, seed=seed)
        self.seq_len = seq_len
    
    def sample(self, batch_size: int):
        """
        采样序列
        返回: (base_states, depth_sequences, actions, rewards, next_base_states, next_depth_sequences, dones)
        """
        if self.current_size < self.seq_len:
            return None
        
        # 采样有效索引（历史窗口不跨越done边界）
        ind = self._get_valid_indices(batch_size, self.seq_len)
        if ind is None:
            return None
        
        # 重建当前状态序列
        depth_sequences = build_sequences_from_indices(
            self.frame_buf, ind, self.seq_len, self.done_buf, padding_mode='none'
        )
        
        # 重建下一个状态序列
        next_ind = (ind + 1) % self.max_size
        next_depth_sequences = build_sequences_from_indices(
            self.frame_buf, next_ind, self.seq_len, self.done_buf, padding_mode='none'
        )
        terminal_mask = self.done_buf[ind].reshape(-1) > 0.5
        next_depth_sequences[terminal_mask] = 0.0
        
        return (
            self.base_buf[ind].astype(np.float32),
            depth_sequences,  # (batch, seq_len, 1, H, W)
            self.action_buf[ind].astype(np.float32),
            self.reward_buf[ind].astype(np.float32),
            self.next_base_buf[ind].astype(np.float32),
            next_depth_sequences,  # (batch, seq_len, 1, H, W)
            self.done_buf[ind].astype(np.float32)
        )


class LazyStackedReplayBuffer(LazyReplayBuffer):
    """
    堆叠帧型LazyFrames缓冲区
    返回堆叠帧 (batch, n_frames, H, W) 供CNN模型使用
    """
    
    def __init__(self, max_size: int, n_frames: int = 4, seed=None):
        super().__init__(max_size, n_frames, seed)
    
    def sample(self, batch_size: int):
        """
        采样堆叠帧
        返回: (base_states, stacked_frames, actions, rewards, next_base_states, next_stacked_frames, dones)
        """
        if self.current_size < self.n_frames:
            return None
        
        # 采样有效索引（历史窗口不跨越done边界）
        ind = self._get_valid_indices(batch_size, self.n_frames)
        if ind is None:
            return None
        
        # 重建当前状态堆叠帧
        stacked_frames = build_stacked_frames(
            self.frame_buf, ind, self.n_frames, self.done_buf, padding_mode='none'
        )
        
        # 重建下一个状态堆叠帧
        next_ind = (ind + 1) % self.max_size
        next_stacked_frames = build_stacked_frames(
            self.frame_buf, next_ind, self.n_frames, self.done_buf, padding_mode='none'
        )
        terminal_mask = self.done_buf[ind].reshape(-1) > 0.5
        next_stacked_frames[terminal_mask] = 0.0
        
        return (
            self.base_buf[ind].astype(np.float32),
            stacked_frames,  # (batch, n_frames, H, W)
            self.action_buf[ind].astype(np.float32),
            self.reward_buf[ind].astype(np.float32),
            self.next_base_buf[ind].astype(np.float32),
            next_stacked_frames,  # (batch, n_frames, H, W)
            self.done_buf[ind].astype(np.float32)
        )
