import numpy as np
import torch

from algorithm.ST_Vim_PPO.buffer import RolloutBuffer


class PLRolloutBuffer(RolloutBuffer):
    """PPO rollout buffer with per-step privileged critic observations.

    Lazy initialisation: ``critic_priv_dim`` is inferred from the first
    ``add()`` call so that the buffer automatically matches the runtime
    privileged observation size (e.g. distance sensor count).
    """

    def __init__(self, *args, **kwargs):
        self.critic_priv_dim = None
        self.critic_privs = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _flatten_priv(priv, target_dim=None):
        if priv is None:
            if target_dim is None:
                return np.zeros((0,), dtype=np.float32)
            return np.zeros((target_dim,), dtype=np.float32)
        arr = np.asarray(priv, dtype=np.float32).reshape(-1)
        if target_dim is not None and arr.size != target_dim:
            raise ValueError(f"critic_priv dim mismatch: expected {target_dim}, got {arr.size}")
        return arr.astype(np.float32, copy=False)

    def add(self, base_state, depth, action, reward, value, log_prob, done, critic_priv=None):
        idx = self.ptr % self.buffer_size

        critic_priv_flat = self._flatten_priv(critic_priv)

        # Lazy initialisation on first privileged observation
        if self.critic_priv_dim is None:
            self.critic_priv_dim = int(critic_priv_flat.size)
            if self.critic_priv_dim > 0:
                self.critic_privs = np.zeros(
                    (self.buffer_size, self.critic_priv_dim), dtype=np.float32
                )

        super().add(base_state, depth, action, reward, value, log_prob, done)

        if self.critic_priv_dim is not None and self.critic_priv_dim > 0:
            self.critic_privs[idx] = self._flatten_priv(critic_priv, target_dim=self.critic_priv_dim)

    def get_trajectory(self):
        data = super().get_trajectory()
        path_slice = slice(self.path_start_idx, self.ptr)

        if self.critic_priv_dim is None or self.critic_priv_dim <= 0:
            data["critic_privs"] = torch.empty((0, 0), dtype=torch.float32, device=self.device)
        else:
            data["critic_privs"] = torch.as_tensor(
                self.critic_privs[path_slice],
                dtype=torch.float32,
                device=self.device,
            )
        return data
