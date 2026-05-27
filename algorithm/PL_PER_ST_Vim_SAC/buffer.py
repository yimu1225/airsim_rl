import numpy as np

from ..sum_tree_replay import SumTreePrioritizedReplayBuffer


class PrioritizedReplayBuffer(SumTreePrioritizedReplayBuffer):
    """Single-pool PER buffer for PL ST-Vim-SAC transitions."""

    def __init__(self, capacity, alpha=0.6, eps=1e-6, seed=None):
        super().__init__(
            capacity,
            alpha=alpha,
            eps=eps,
            seed=seed,
            depth_field_indices=(1, 5, 7, 8),
            depth_dtype=np.float16,
            return_stacked=True,
        )
        self.critic_priv_shape = None

    @staticmethod
    def _as_priv(priv, target_shape=None):
        if priv is None:
            if target_shape is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros(target_shape, dtype=np.float32)
        arr = np.asarray(priv, dtype=np.float32)
        if target_shape is not None and arr.shape != tuple(target_shape):
            raise ValueError(f"critic_depth shape mismatch: expected {target_shape}, got {arr.shape}")
        return arr

    def add(
        self,
        base_state,
        depth,
        action,
        reward,
        next_base_state,
        next_depth,
        done,
        is_success=0.0,
        critic_priv=None,
        next_critic_priv=None,
    ):
        del is_success
        if self.critic_priv_shape is None:
            critic_depth = self._as_priv(critic_priv)
            self.critic_priv_shape = critic_depth.shape
        else:
            critic_depth = self._as_priv(critic_priv, target_shape=self.critic_priv_shape)
        next_critic_depth = self._as_priv(next_critic_priv, target_shape=self.critic_priv_shape)

        super().add(
            base_state,
            depth,
            action,
            reward,
            next_base_state,
            next_depth,
            done,
            critic_depth,
            next_critic_depth,
        )
