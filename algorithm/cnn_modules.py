import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Unified CNN for processing single input channel.
    Can be used for depth, motion, or any single-channel input.
    Used by all algorithms for consistent feature extraction.
    Now uses Pooling layers instead of adaptive_avg_pool2d.
    """
    def __init__(self, input_height, input_width, input_channels=1):
        super().__init__()

        # Feature expansion factors
        f1 = 8
        f2 = 16
        f3 = 32
        f4 = 16
        # f5 = 48
        f5 = 8

        self.net = nn.Sequential(
            # 第一层: 128x128 -> 32x32 (stride=4, 下采样4倍)
            nn.Conv2d(input_channels, f1, kernel_size=8, stride=4, padding=2),
            # nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
           
            # 第二层: 32x32 -> 16x16 (stride=2, 下采样2倍)
            nn.Conv2d(f1, f2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),

            # 第三层: 16x16 -> 8x8 (stride=2, 下采样2倍)
            nn.Conv2d(f2, f3, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),

            # 第四层: 8x8 -> 4x4 (stride=2, 下采样2倍)
            nn.Conv2d(f3, f4, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(f4),
            nn.ReLU(inplace=True),
            
            # # 第五层 1x1 conv (保持空间维度)
            # nn.Conv2d(f4, f5, kernel_size=1),
            # # nn.BatchNorm2d(f5),
            # nn.ReLU(inplace=True),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            dummy_output = self.net(dummy_input)
            self.n_flatten = dummy_output.view(1, -1).size(1)

        self.repr_dim = self.n_flatten

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if needed
        x = self.net(x)
        x = x.view(x.size(0), -1)  # Flatten directly
        return x
