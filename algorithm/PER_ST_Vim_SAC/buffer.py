import numpy as np

from ..sum_tree_replay import SumTreePrioritizedReplayBuffer


class PrioritizedReplayBuffer(SumTreePrioritizedReplayBuffer):
    """Single-pool prioritized replay buffer for ST-Vim-SAC."""

    def __init__(self, capacity, alpha=0.6, eps=1e-6, seed=None):
        super().__init__(
            capacity,
            alpha=alpha,
            eps=eps,
            seed=seed,
            depth_field_indices=(1, 5),
            depth_dtype=np.float16,
            return_stacked=True,
        )
