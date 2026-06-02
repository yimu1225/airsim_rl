import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

from mamba_ssm import Mamba

from ..cnn_modules import CNN


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class DepthCNNEncoder(CNN):
    """Single-frame depth encoder used before Mamba history modeling."""

    def __init__(self, input_height: int, input_width: int, output_dim: int = 64):
        super().__init__(
            input_height=input_height,
            input_width=input_width,
            input_channels=1,
            output_dim=output_dim,
            frame_wise=False,
        )


class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaHistoryEncoder(nn.Module):
    """Encode history tokens [visual, base, previous_action] into sequence states."""

    def __init__(
        self,
        token_dim: int,
        history_dim: int = 128,
        n_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, history_dim),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList(
            [
                MambaBlock(history_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.output_norm = nn.LayerNorm(history_dim)
        self.repr_dim = int(history_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected history tokens with shape (B,T,D), got {tuple(tokens.shape)}")
        x = self.token_proj(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output_norm(x)


class Actor(nn.Module):
    """Squashed Gaussian SAC actor."""

    def __init__(self, repr_dim: int, action_shape, hidden_dim: int):
        super().__init__()
        self.action_dim = int(action_shape[0])
        self.input_norm = nn.LayerNorm(repr_dim)
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Linear(hidden_dim, self.action_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def distribution_params(self, obs: torch.Tensor):
        h = self.trunk(self.input_norm(obs))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def action_log_prob(self, obs: torch.Tensor):
        mean, log_std = self.distribution_params(obs)
        dist = pyd.Normal(mean, log_std.exp())
        gaussian_action = dist.rsample()
        action = torch.tanh(gaussian_action)
        log_prob = dist.log_prob(gaussian_action).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - gaussian_action - F.softplus(-2 * gaussian_action))).sum(
            dim=-1,
            keepdim=True,
        )
        return action, log_prob

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.distribution_params(obs)
        if deterministic:
            return torch.tanh(mean)
        dist = pyd.Normal(mean, log_std.exp())
        return torch.tanh(dist.sample())


class Critic(nn.Module):
    """Twin Q critic for SAC."""

    def __init__(self, repr_dim: int, action_shape, hidden_dim: int):
        super().__init__()
        input_dim = int(repr_dim) + int(action_shape[0])
        self.input_norm = nn.LayerNorm(input_dim)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        x = self.input_norm(x)
        return self.q1(x), self.q2(x)
