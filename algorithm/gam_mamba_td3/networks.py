import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from mamba_ssm import Mamba


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
            nn.Conv2d(channels, reduced_channels, kernel_size=7, padding=3, padding_mode="replicate", bias=False,),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=7, padding=3, padding_mode="replicate", bias=False,),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape

        # Channel attention
        x_permute = x.permute(0, 2, 3, 1).contiguous().view(bsz, -1, channels)
        x_att_permute = self.channel_attn(x_permute).view(bsz, height, width, channels)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).contiguous()

        x = x * x_channel_att

        # Spatial attention
        x_spatial_att = self.spatial_attn(x)

        out = x * x_spatial_att
        return out


class MambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class ConvGAMMambaEncoder(nn.Module):
    """
    Conv feature extractor + GAM attention + Mamba stack encoder.
    Input:  (B, C, H, W)
    Output: (B, feature_dim)
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        feature_dim: int,
        input_channels: int = 1,
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()

        f1, f2, f3, f4, f5 = 4, 8, 16, 32, 16

        self.conv_backbone = nn.Sequential(
            nn.Conv2d(input_channels, f1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            GAMAttention(f1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(f1, f2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(f3, f4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f4, f5, kernel_size=1),
            nn.BatchNorm2d(f5),
            nn.ReLU(inplace=True),
        )

        # self.gam = GAMAttention(f5)  # Removed, now embedded in conv_backbone

        self.mamba_stack = nn.ModuleList(
            [
                MambaBlock(
                    dim=f5,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                )
                for _ in range(max(1, mamba_layers))
            ]
        )

        self.proj = nn.Linear(f5, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self.repr_dim = feature_dim

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            feat = self.conv_backbone(dummy)
            self.token_h, self.token_w = feat.shape[-2], feat.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_backbone(x)
        # x = self.gam(x)  # Removed, now embedded in conv_backbone

        bsz, ch, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        for layer in self.mamba_stack:
            tokens = layer(tokens)

        pooled = tokens.mean(dim=1)
        out = self.norm(self.proj(pooled))
        return out


class Encoder(ConvGAMMambaEncoder):
    """
    Alias for agent compatibility.
    """


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
