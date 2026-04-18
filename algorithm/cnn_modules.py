import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Unified visual encoder based on MobileNetV2.
    Accepts arbitrary input channels and returns 128-d projected features.
    """
    def __init__(self, input_height, input_width, input_channels=1, output_dim=128):
        del input_height, input_width  # MobileNetV2 supports dynamic spatial resolution.
        super().__init__()

        width_mult = 0.35
        try:
            mobilenet = models.mobilenet_v2(weights=None, width_mult=width_mult)
        except TypeError:
            mobilenet = models.mobilenet_v2(pretrained=False, width_mult=width_mult)

        self.net = mobilenet.features

        if input_channels != 3:
            first_conv = self.net[0][0]
            self.net[0][0] = nn.Conv2d(
                input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(mobilenet.last_channel, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )
        self.repr_dim = output_dim

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if needed
        x = self.net(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x
