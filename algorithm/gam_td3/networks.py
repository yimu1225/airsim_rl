import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class TruncatedNormal(pyd.Normal):
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
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class GAMAttention(nn.Module):
    """
    Global Attention Mechanism (GAM): channel attention + spatial attention.
    Input/Output shape: (B, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)

        self.channel_attn = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=7, padding=3, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=7, padding=3, padding_mode="replicate", bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape

        x_permute = x.permute(0, 2, 3, 1).contiguous().view(bsz, -1, channels)
        x_att_permute = self.channel_attn(x_permute).view(bsz, height, width, channels)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).contiguous()

        x = x * x_channel_att
        x_spatial_att = self.spatial_attn(x)
        out = x * x_spatial_att
        return out


class ConvGAMEncoder(nn.Module):
    """
    Conv feature extractor + GAM attention encoder.
    Input:  (B, C, H, W)
    Output: (B, feature_dim)
    """

    def __init__(self, input_height: int, input_width: int, feature_dim: int, input_channels: int = 1):
        super().__init__()

        f1, f2, f3, f4 = 8, 16, 24, 32

        self.conv_backbone = nn.Sequential(
            nn.Conv2d(input_channels, f1, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            GAMAttention(f3),
            nn.Conv2d(f3, f4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            feat = self.conv_backbone(dummy)
            token_dim = feat.shape[1]

        self.proj = nn.Linear(token_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.repr_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_backbone(x)
        x = x.flatten(1)
        out = self.norm(self.proj(x))
        return out


class Encoder(ConvGAMEncoder):
    """Alias for agent compatibility."""


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.input_norm = nn.LayerNorm(repr_dim)

        self.policy = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, std=None):
        mu = self.input_norm(obs)
        mu = self.policy(mu)
        mu = torch.tanh(mu)
        return mu


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.input_norm = nn.LayerNorm(repr_dim + action_shape[0])

        self.Q1 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        h_action = self.input_norm(h_action)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2
