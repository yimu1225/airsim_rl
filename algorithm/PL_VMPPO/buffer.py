import numpy as np
import torch

from algorithm.VMPPO.buffer import RolloutBuffer


class PLRolloutBuffer(RolloutBuffer):
    """PPO rollout buffer with per-step privileged critic observations.

    The critic observation width is fixed from the agent config, so missing
    sensor reads can be represented as zeros instead of breaking mini-batch
    indexing during the PPO update.
    """

    def __init__(self, *args, critic_priv_dim=0, **kwargs):
        self.critic_priv_dim = int(critic_priv_dim)
        self.critic_priv_shape = (self.critic_priv_dim,)
        self.critic_privs = None
        super().__init__(*args, **kwargs)
        if self.critic_priv_dim > 0:
            self.critic_privs = np.zeros(
                (self.buffer_size, self.critic_priv_dim), dtype=np.float16
            )

    def _as_priv(self, priv):
        if self.critic_priv_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        if priv is None:
            return np.zeros((self.critic_priv_dim,), dtype=np.float32)

        arr = np.asarray(priv, dtype=np.float32).reshape(-1)
        if arr.size > self.critic_priv_dim:
            arr = arr[: self.critic_priv_dim]
        elif arr.size < self.critic_priv_dim:
            arr = np.pad(arr, (0, self.critic_priv_dim - arr.size), mode="constant")
        return arr.astype(np.float32, copy=False)

    def add(self, base_state, depth, action, reward, value, log_prob, done, critic_priv=None):
        idx = self.ptr

        super().add(base_state, depth, action, reward, value, log_prob, done)

        if self.critic_priv_dim > 0:
            self.critic_privs[idx] = self._as_priv(critic_priv)

    def get_trajectory(self):
        data = super().get_trajectory()
        path_slice = slice(self.path_start_idx, self.ptr)

        if self.critic_priv_dim <= 0:
            n_samples = data["base_states"].shape[0]
            data["critic_privs"] = torch.empty((n_samples, 0), dtype=torch.float32, device=self.device)
        else:
            data["critic_privs"] = torch.as_tensor(
                self.critic_privs[path_slice],
                dtype=torch.float32,
                device=self.device,
            )
        return data
