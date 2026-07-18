# SPDX-License-Identifier: GPL-3.0-only
"""Networks for SAC_FAE adapted to this project's AirSim interface.

The Focus/DeFocus autoencoder structure is adapted from the official
LHL6666/SAC_FAE release at commit 2c235ade9cf3df4258e8fee4008a31b673d9a94f:
https://github.com/LHL6666/SAC_FAE

The release fixes its linear bottleneck to a 120x160 input. This adaptation
keeps the published operations while fixing the contract to this project's
four stacked 128x128 depth frames.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn


class FocusConcat(nn.Module):
    """Losslessly move each 2x2 spatial neighborhood into channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ),
            dim=1,
        )


class DeFocusBlock(nn.Module):
    """Inverse of :class:`FocusConcat` for a gain of two."""

    def __init__(self, gain: int = 2) -> None:
        super().__init__()
        self.gain = int(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        gain = self.gain
        divisor = gain * gain
        if channels % divisor != 0:
            raise ValueError(
                f"DeFocusBlock requires channels divisible by {divisor}, got {channels}."
            )
        x = x.view(batch, gain, gain, channels // divisor, height, width)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(batch, channels // divisor, height * gain, width * gain)


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """Convert the environment's 0..255 depth representation to 0..1."""

    depth = depth.float()
    if depth.numel() and float(depth.detach().amax().item()) > 1.5:
        depth = depth / 255.0
    return depth.clamp(0.0, 1.0)


class FocusEncoder(nn.Module):
    """Official Focus encoder with a runtime-derived bottleneck shape."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        feature_dim: int = 384,
        num_filters: int = 32,
    ) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(f"FocusEncoder expects (C,H,W), got {obs_shape}.")

        channels, height, width = (int(value) for value in obs_shape)
        if (channels, height, width) != (4, 128, 128):
            raise ValueError(
                "SAC_FAE expects exactly 4x128x128 stacked depth input, "
                f"got {channels}x{height}x{width}."
            )
        if num_filters < 4 or num_filters % 2 != 0:
            raise ValueError("num_filters must be an even integer greater than or equal to 4.")

        base_channels = int(num_filters) // 2
        focused_channels = base_channels * 4
        self.obs_shape = (channels, height, width)
        self.feature_dim = int(feature_dim)
        self.latent_shape = (focused_channels, height // 4, width // 4)

        self.base_block = nn.Sequential(
            nn.Conv2d(
                channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels, affine=True),
            nn.SELU(inplace=True),
        )
        self.conv_block = nn.Sequential(
            FocusConcat(),
            nn.Conv2d(focused_channels, focused_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(focused_channels, affine=True),
            nn.SELU(inplace=True),
            nn.Conv2d(focused_channels, focused_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(focused_channels, affine=True),
            nn.SELU(inplace=True),
        )
        self.encoder_layer = nn.Linear(math.prod(self.latent_shape), self.feature_dim)
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        self.repr_dim = self.feature_dim

    def forward(self, obs: torch.Tensor, stop_gradient: bool = False) -> torch.Tensor:
        obs = normalize_depth(obs)
        features = self.conv_block(self.base_block(obs))
        if tuple(features.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Unexpected Focus feature shape {tuple(features.shape[1:])}; "
                f"expected {self.latent_shape}."
            )
        features = features.flatten(start_dim=1)
        if stop_gradient:
            features = features.detach()
        return torch.tanh(self.layer_norm(self.encoder_layer(features)))


class FocusDecoder(nn.Module):
    """Official DeFocus decoder adapted to the encoder's runtime shape."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        feature_dim: int,
        latent_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        channels, _, _ = (int(value) for value in obs_shape)
        latent_channels, _, _ = (int(value) for value in latent_shape)
        if latent_channels % 4 != 0:
            raise ValueError("FocusDecoder latent channels must be divisible by 4.")

        self.obs_shape = tuple(int(value) for value in obs_shape)
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.decoder_layer = nn.Linear(int(feature_dim), math.prod(self.latent_shape))
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels, affine=True),
            nn.SELU(inplace=True),
            nn.ConvTranspose2d(latent_channels, latent_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(latent_channels, affine=True),
            nn.SELU(inplace=True),
            DeFocusBlock(gain=2),
            nn.ConvTranspose2d(latent_channels // 4, channels, 4, 2, 1, bias=True),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.decoder_layer(latent), inplace=False)
        features = features.view(latent.shape[0], *self.latent_shape)
        reconstructed = torch.tanh(self.deconv_block(features))
        if tuple(reconstructed.shape[1:]) != self.obs_shape:
            raise ValueError(
                f"Unexpected reconstruction shape {tuple(reconstructed.shape[1:])}; "
                f"expected {self.obs_shape}."
            )
        return reconstructed


class FocusAutoencoder(nn.Module):
    """Focus encoder and decoder with a shared latent representation."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        feature_dim: int = 384,
        num_filters: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = FocusEncoder(obs_shape, feature_dim, num_filters)
        self.decoder = FocusDecoder(obs_shape, feature_dim, self.encoder.latent_shape)
        self.repr_dim = self.encoder.repr_dim

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(depth)
        return latent, self.decoder(latent)

    @staticmethod
    def reconstruction_target(depth: torch.Tensor) -> torch.Tensor:
        """Map normalized depth into the decoder's tanh range."""

        return normalize_depth(depth).mul(2.0).sub(1.0)


class MeasurementEncoder(nn.Module):
    """Two-layer measurement encoder from the SAC_FAE paper."""

    def __init__(self, input_dim: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(feature_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(feature_dim), int(feature_dim)),
            nn.SiLU(inplace=True),
        )
        self.repr_dim = int(feature_dim)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        return self.net(base)


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for _ in range(int(hidden_layers)):
        layers.extend((nn.Linear(current_dim, int(hidden_dim)), nn.SiLU(inplace=True)))
        current_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Squashed-Gaussian SAC policy using the paper's SiLU MLP."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        hidden_layers: int = 4,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.trunk = _mlp(state_dim, hidden_dim, hidden_layers)
        self.mean = nn.Linear(int(hidden_dim), self.action_dim)
        self.log_std = nn.Linear(int(hidden_dim), self.action_dim)

    def distribution_params(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(state)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def action_log_prob(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_params(state)
        gaussian = pyd.Normal(mean, log_std.exp())
        raw_action = gaussian.rsample()
        action = torch.tanh(raw_action)
        log_prob = gaussian.log_prob(raw_action).sum(dim=-1, keepdim=True)
        correction = 2.0 * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
        return action, log_prob - correction.sum(dim=-1, keepdim=True)

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.distribution_params(state)
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(pyd.Normal(mean, log_std.exp()).sample())


class Critic(nn.Module):
    """Twin SAC Q-functions using the paper's SiLU MLP."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        input_dim = int(state_dim) + int(action_dim)
        self.q1_trunk = _mlp(input_dim, hidden_dim, hidden_layers)
        self.q2_trunk = _mlp(input_dim, hidden_dim, hidden_layers)
        self.q1_out = nn.Linear(int(hidden_dim), 1)
        self.q2_out = nn.Linear(int(hidden_dim), 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((state, action), dim=-1)
        return self.q1_out(self.q1_trunk(inputs)), self.q2_out(self.q2_trunk(inputs))


__all__ = [
    "Actor",
    "Critic",
    "DeFocusBlock",
    "FocusAutoencoder",
    "FocusConcat",
    "FocusDecoder",
    "FocusEncoder",
    "MeasurementEncoder",
    "normalize_depth",
]
