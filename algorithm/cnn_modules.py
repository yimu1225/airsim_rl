import torch
import torch.nn as nn
from torchvision import models


class BaseStateExpander(nn.Module):
    """
    Expands base state to a higher dimension for temporal processing.
    Used by all temporal algorithms (LSTM, GRU, CFC) to enrich low-dimensional
    base state before combining with visual features.
    """
    def __init__(self, base_dim, expanded_dim=32):
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
    Unified visual encoder based on MobileNetV3-Small.
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
        output_dim=128,
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

        width_mult = 0.35
        try:
            mobilenet = models.mobilenet_v3_small(weights=None, width_mult=width_mult)
        except TypeError:
            mobilenet = models.mobilenet_v3_small(pretrained=False, width_mult=width_mult)

        self.net = mobilenet.features

        if backbone_in_channels != 3:
            first_conv = self.net[0][0]
            self.net[0][0] = nn.Conv2d(
                backbone_in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            nn.init.kaiming_normal_(self.net[0][0].weight, mode='fan_out', nonlinearity='relu')
            if self.net[0][0].bias is not None:
                nn.init.zeros_(self.net[0][0].bias)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Infer pooled feature size from backbone output to stay robust across torchvision versions.
        probe_h = max(32, int(input_height))
        probe_w = max(32, int(input_width))
        with torch.no_grad():
            probe = torch.zeros(1, backbone_in_channels, probe_h, probe_w)
            feat = self.pool(self.net(probe))
            feature_dim = int(feat.view(1, -1).shape[-1])

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )
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
        if self.frame_wise:
            seq = self._prepare_frame_sequence(x)  # (B, T, H, W)
            batch_size, seq_len, height, width = seq.shape
            frames = seq.reshape(batch_size * seq_len, 1, height, width)
            feats = self.net(frames)
            feats = self.pool(feats).view(batch_size * seq_len, -1)
            feats = self.proj(feats).view(batch_size, seq_len, self.single_frame_dim)
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

        x = self.net(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.proj(x)
        return x
