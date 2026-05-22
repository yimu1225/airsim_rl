import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from mamba_ssm import Mamba
from ..cnn_modules import CNN
from ..config_loader import get_algo_param


class TruncatedNormal(pyd.Normal):
    """Truncated Normal distribution for SAC action sampling."""
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(self.norm(x))


class TemporalMambaStack(nn.Module):
    def __init__(self, dim, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(max(1, int(n_layers)))]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MambaEncoder(nn.Module):
    """
    Mamba encoder: CNN for per-frame spatial features + Temporal Mamba for sequence modeling.
    """

    def __init__(self, args):
        super().__init__()
        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        height = depth_shape[1]
        width = depth_shape[2]

        self.seq_len = getattr(args, "n_frames", 4)
        self.spatial_encoder = CNN(
            input_height=height,
            input_width=width,
            input_channels=1,
        )
        self.embed_dim = self.spatial_encoder.repr_dim

        self.temporal_layers = int(get_algo_param(args, "mamba_sac_temporal_depth", 2))
        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=get_algo_param(args, "mamba_d_state", 16),
            d_conv=get_algo_param(args, "mamba_d_conv", 4),
            expand=get_algo_param(args, "mamba_expand", 2),
        )

        self.flatten_all_tokens = bool(get_algo_param(args, "mamba_sac_flatten_all_tokens", True))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim

    def forward(self, depth_seq):
        if depth_seq.dim() == 2:
            # (H, W) -> (1, 1, 1, H, W)
            depth_seq = depth_seq.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif depth_seq.dim() == 3:
            # (T, H, W) -> (1, T, 1, H, W)
            depth_seq = depth_seq.unsqueeze(0).unsqueeze(2)
        elif depth_seq.dim() == 4:
            # (B, T, H, W) -> (B, T, 1, H, W)
            depth_seq = depth_seq.unsqueeze(2)
        elif depth_seq.dim() != 5:
            raise ValueError(f"Unsupported depth_seq shape: {tuple(depth_seq.shape)}")

        bsz, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        frames = depth_seq.reshape(bsz * seq_len, channels, height, width)
        frame_tokens = self.spatial_encoder(frames).view(bsz, seq_len, self.embed_dim)
        temporal_tokens = self.temporal_mamba(frame_tokens)

        if self.flatten_all_tokens:
            return temporal_tokens.reshape(bsz, seq_len * self.embed_dim)
        return temporal_tokens[:, -1, :]


class Actor(nn.Module):
    """SAC Actor with stochastic policy (Gaussian)."""
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.action_dim = action_shape[0]

        # LayerNorm on input representation
        self.input_norm = nn.LayerNorm(repr_dim)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Mean and log_std heads
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, with_log_prob=False):
        """
        Args:
            obs: encoded state representation
            compute_pi: whether to sample action
            compute_log_pi: whether to compute log probability
            with_log_prob: if True, return log_prob along with action
        Returns:
            action (detached if not with_log_prob), mean, log_std
        """
        obs = self.input_norm(obs)
        h = self.trunk(obs)

        mean = self.mean_linear(h)
        log_std = self.log_std_linear(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        if not compute_pi:
            return mean, log_std

        # Reparameterization trick
        std = log_std.exp()
        dist = pyd.Normal(mean, std)

        # Sample action
        z = dist.rsample()
        action = torch.tanh(z)

        if compute_log_pi:
            # Log probability correction for tanh squashing
            log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
            log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(dim=-1, keepdim=True)

            if with_log_prob:
                return action, log_prob, mean, log_std
            return action, log_prob, mean, log_std

        return action, mean, log_std

    def get_action(self, obs, deterministic=False):
        """Get action for evaluation (deterministic or sampled)."""
        mean, log_std = self.forward(obs, compute_pi=False)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            dist = pyd.Normal(mean, std)
            z = dist.rsample()
            action = torch.tanh(z)
        return action


class Critic(nn.Module):
    """Twin Q-network Critic for SAC."""

    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.input_norm = nn.LayerNorm(repr_dim + action_shape[0])

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = self.input_norm(x)
        return self.q1(x), self.q2(x)
