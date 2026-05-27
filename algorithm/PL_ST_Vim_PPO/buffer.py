import numpy as np
import torch

from algorithm.ST_Vim_PPO.buffer import RolloutBuffer


class PLRolloutBuffer(RolloutBuffer):
    """PPO rollout buffer with per-step privileged critic observations.

    Lazy initialisation: ``critic_priv_dim`` is inferred from the first
    ``add()`` call so that the buffer automatically matches the runtime
    clean critic-depth observation shape.
    """

    def __init__(self, *args, **kwargs):
        self.critic_priv_shape = None
        self.critic_privs = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _as_priv(priv, target_shape=None):
        if priv is None:
            if target_shape is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros(target_shape, dtype=np.float32)
        arr = np.asarray(priv, dtype=np.float32)
        if target_shape is not None and arr.shape != tuple(target_shape):
            raise ValueError(f"critic_depth shape mismatch: expected {target_shape}, got {arr.shape}")
        return arr.astype(np.float32, copy=False)

    def add(self, base_state, depth, action, reward, value, log_prob, done, critic_priv=None):
        idx = self.ptr % self.buffer_size

        critic_priv_arr = self._as_priv(critic_priv)

        # Lazy initialisation on first privileged observation
        if self.critic_priv_shape is None:
            self.critic_priv_shape = critic_priv_arr.shape
            if critic_priv_arr.size > 0:
                self.critic_privs = np.zeros(
                    (self.buffer_size, *self.critic_priv_shape), dtype=np.float16
                )

        super().add(base_state, depth, action, reward, value, log_prob, done)

        if self.critic_priv_shape is not None and np.prod(self.critic_priv_shape, dtype=np.int64) > 0:
            self.critic_privs[idx] = self._as_priv(critic_priv, target_shape=self.critic_priv_shape)

    def get_trajectory(self):
        data = super().get_trajectory()
        path_slice = slice(self.path_start_idx, self.ptr)

        if self.critic_priv_shape is None or np.prod(self.critic_priv_shape, dtype=np.int64) <= 0:
            data["critic_privs"] = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        else:
            data["critic_privs"] = torch.as_tensor(
                self.critic_privs[path_slice],
                dtype=torch.float32,
                device=self.device,
            )
        return data
