import numpy as np
import torch

from algorithm.ST_Vim_PPO.buffer import RolloutBuffer


class PLRolloutBuffer(RolloutBuffer):
    """PPO rollout buffer with per-step privileged critic observations."""

    def __init__(self, *args, critic_priv_dim: int, **kwargs):
        self.critic_priv_dim = int(critic_priv_dim)
        super().__init__(*args, **kwargs)
        self.critic_privs = np.zeros((self.buffer_size, self.critic_priv_dim), dtype=np.float32)

    def _flatten_priv(self, critic_priv):
        if self.critic_priv_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        if critic_priv is None:
            return np.zeros((self.critic_priv_dim,), dtype=np.float32)

        arr = np.asarray(critic_priv, dtype=np.float32).reshape(-1)
        if arr.size > self.critic_priv_dim:
            arr = arr[: self.critic_priv_dim]
        elif arr.size < self.critic_priv_dim:
            arr = np.pad(arr, (0, self.critic_priv_dim - arr.size), mode="constant")
        return arr.astype(np.float32, copy=False)

    def add(self, base_state, depth, action, reward, value, log_prob, done, critic_priv=None):
        idx = self.ptr % self.buffer_size
        super().add(base_state, depth, action, reward, value, log_prob, done)
        self.critic_privs[idx] = self._flatten_priv(critic_priv)

    def get_trajectory(self):
        data = super().get_trajectory()
        path_slice = slice(self.path_start_idx, self.ptr)
        data["critic_privs"] = torch.as_tensor(
            self.critic_privs[path_slice],
            dtype=torch.float32,
            device=self.device,
        )
        return data
