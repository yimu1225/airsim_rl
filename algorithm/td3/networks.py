import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
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
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


# RunningMeanStd removed — wasn't integrated in agent pipeline.


class Encoder(nn.Module):
    """
    TD3 visual encoder backed by ResNet18.
    """
    def __init__(self, input_height, input_width, input_channels=1):
        del input_height, input_width  # ResNet18 supports dynamic spatial resolution.
        super().__init__()

        # Build a vanilla ResNet18 (no pretrained weights by default).
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            # Backward compatibility for older torchvision versions.
            backbone = models.resnet18(pretrained=False)

        # Adapt first conv to arbitrary input channels (e.g., stacked depth frames).
        if input_channels != 3:
            backbone.conv1 = nn.Conv2d(
                input_channels,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=False,
            )

        # Drop final FC head and keep global pooled 512-d features.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.repr_dim = 512

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        feat = self.backbone(x)
        return feat.view(feat.size(0), -1)
   

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # LayerNorm on input representation to normalize features
        self.input_norm = nn.LayerNorm(repr_dim)

        self.policy = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, std=None):
        # apply input normalization
        obs = self.input_norm(obs)
        mu = self.policy(obs)
        mu = torch.tanh(mu) 
        
        # Return normalized action (-1, 1)
        return mu


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # LayerNorm for critic input (repr + action)
        self.input_norm = nn.LayerNorm(repr_dim + action_shape[0])
        
        self.Q1 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        h_action = torch.cat([obs, action], dim=-1)
        h_action = self.input_norm(h_action)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2
