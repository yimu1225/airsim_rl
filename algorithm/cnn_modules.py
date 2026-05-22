import torch.nn as nn
import torch
import numpy as np
from gymnasium import spaces


class NatureCNN(nn.Module):
    """
    Local copy of Stable-Baselines3 NatureCNN.

    This keeps the original DQN Nature convolutional backbone local so it can be
    edited without changing the vendored/stable-baselines3 implementation.
    """
    def __init__(
        self,
        observation_space,
        features_dim=512,
        normalized_image=False,
    ):
        super().__init__()
        del normalized_image
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"NatureCNN expects spaces.Box, got {type(observation_space)!r}")
        if observation_space.shape is None or len(observation_space.shape) != 3:
            raise ValueError(
                f"NatureCNN expects channel-first image shape (C,H,W), got {observation_space.shape}"
            )

        self._observation_space = observation_space
        self._features_dim = int(features_dim)
        n_input_channels = int(observation_space.shape[0])

        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape, dtype=torch.float32)
            n_flatten = self.flatten(self._forward_conv(sample)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, self._features_dim), nn.ReLU())
        self.repr_dim = self._features_dim

    @property
    def features_dim(self):
        return self._features_dim

    def _forward_conv(self, observations):
        x = self.relu1(self.conv1(observations))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x

    def _forward_head(self, features):
        return self.linear(self.flatten(features))

    def forward(self, observations):
        return self._forward_head(self._forward_conv(observations))


class BaseStateExpander(nn.Module):
    """
    Expands base state to a higher dimension for temporal processing.
    Used by all temporal algorithms (LSTM, GRU, CFC) to enrich low-dimensional
    base state before combining with visual features.
    """
    def __init__(self, base_dim, expanded_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(base_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.ReLU(inplace=True)
        )
        self.repr_dim = expanded_dim
    
    def forward(self, x):
        """
        Args:
            x: (B, K, base_dim) or (B, base_dim)
        Returns:
            expanded: (B, K, expanded_dim) or (B, expanded_dim)
        """
        return self.fc(x)


class CNN(nn.Module):
    """
    Unified visual encoder based on Stable-Baselines3 NatureCNN.
    Accepts depth input and returns projected visual features.

    Behavior:
    - input_channels == 1: standard single-frame encoder.
    - input_channels  > 1: frame-wise shared encoder. The channel count is treated
      as sequence length, and each frame is encoded by the same CNN module.
    """
    def __init__(
        self,
        input_height,
        input_width,
        input_channels=1,
        output_dim=64,
        frame_wise=None,
        flatten_all_tokens=True,
    ):
        super().__init__()

        self.input_channels = int(input_channels)
        if frame_wise is None:
            frame_wise = self.input_channels > 1
        self.frame_wise = bool(frame_wise)
        self.flatten_all_tokens = bool(flatten_all_tokens)
        self.sequence_length = self.input_channels if self.frame_wise else 1
        backbone_in_channels = 1 if self.frame_wise else self.input_channels

        observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(backbone_in_channels, int(input_height), int(input_width)),
            dtype=np.float32,
        )
        self.net = NatureCNN(observation_space, features_dim=output_dim, normalized_image=True)
        self.single_frame_dim = output_dim
        self.repr_dim = (
            output_dim * self.sequence_length
            if self.frame_wise and self.flatten_all_tokens
            else output_dim
        )

    def _prepare_frame_sequence(self, x):
        """
        Convert input into (B, T, H, W) for frame-wise processing.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 5:
            # (B, T, C, H, W) -> require single-channel frame input
            if x.size(2) != 1:
                raise ValueError(f"Frame-wise CNN expects C=1, got C={x.size(2)}")
            x = x.squeeze(2)

        if x.dim() != 4:
            raise ValueError(f"Unsupported frame-wise input shape: {tuple(x.shape)}")

        if x.size(1) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {x.size(1)}"
            )
        return x

    def forward(self, x):
        x = x.float()
        if x.numel() > 0 and float(x.detach().max().item()) > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)

        if self.frame_wise:
            seq = self._prepare_frame_sequence(x)  # (B, T, H, W)
            batch_size, seq_len, height, width = seq.shape
            frames = seq.reshape(batch_size * seq_len, 1, height, width)
            feats = self.net(frames).view(batch_size, seq_len, self.single_frame_dim)
            if self.flatten_all_tokens:
                return feats.reshape(batch_size, seq_len * self.single_frame_dim)
            return feats[:, -1, :]

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            # (C, H, W) where C=input_channels -> (1, C, H, W)
            if x.size(0) == self.input_channels and self.input_channels != 1:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)

        return self.net(x)
