"""
确定性环境参数采样器

基于基础种子和 episode 数生成确定性的环境参数，
确保：
1. 相同种子 + 相同 episode → 完全相同环境（不同算法、不同时间、UE4重启后）
2. 不同种子 → 不同环境序列
"""

import hashlib
from typing import List, Any


class DeterministicSampler:
    """
    确定性采样器，基于 (base_seed, change_counter, param_name) 生成确定性参数
    """
    
    def __init__(self, base_seed: int):
        """
        Args:
            base_seed: 基础随机种子，不同种子产生不同环境序列
        """
        self.base_seed = base_seed
    
    def _get_hash(self, change_counter: int, param_name: str, salt: str = "") -> int:
        """生成确定性哈希值"""
        hash_input = f"seed_{self.base_seed}_change_{change_counter}_{param_name}_{salt}"
        hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()
        return int(hash_hex, 16)
    
    def choice(self, change_counter: int, param_name: str, options: List[Any]) -> Any:
        """从列表中确定性选择一个值"""
        if len(options) == 0:
            raise ValueError(f"Options list for {param_name} is empty")
        hash_val = self._get_hash(change_counter, param_name)
        idx = hash_val % len(options)
        return options[idx]
    
    def uniform(self, change_counter: int, param_name: str, low: float, high: float) -> float:
        """生成确定性的均匀分布随机数"""
        hash_val = self._get_hash(change_counter, param_name)
        normalized = (hash_val % (2**32)) / (2**32)
        return low + normalized * (high - low)
    
    def choice_sign(self, change_counter: int, param_name: str) -> int:
        """确定性选择正负号"""
        hash_val = self._get_hash(change_counter, param_name, "sign")
        return 1 if hash_val % 2 == 0 else -1


def get_deterministic_end_point(arena_size, change_counter: int, base_seed: int) -> List[float]:
    """基于 change_counter 和 base_seed 确定性生成终点坐标"""
    from settings_folder import settings
    
    sampler = DeterministicSampler(base_seed)
    goal_halo = settings.slow_down_activation_distance
    
    idx0_quanta = float((arena_size[0] - goal_halo)) / 2
    idx1_quanta = float((arena_size[1] - goal_halo)) / 2
    
    rnd_idx0 = sampler.uniform(change_counter, "End_x", 2, idx0_quanta)
    rnd_idx1 = sampler.uniform(change_counter, "End_y", 2, idx1_quanta)
    rnd_idx2 = sampler.uniform(change_counter, "End_z", 1.0, 3.0)
    
    rnd_idx0 = rnd_idx0 * sampler.choice_sign(change_counter, "End_x_sign")
    rnd_idx1 = rnd_idx1 * sampler.choice_sign(change_counter, "End_y_sign")
    
    return [rnd_idx0, rnd_idx1, rnd_idx2]
