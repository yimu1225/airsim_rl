import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from ..cnn_modules import CNN



class VisualEncoder(CNN):
    """
    VisualEncoder that uses unified CNN from cnn_modules.
    Processes Depth features only for sequence-based algorithms.
    Depth sequences are processed by CNN and then fed to temporal modules.
    """
    def __init__(self, input_height, input_width, feature_dim, input_channels=1):
        super().__init__(input_height, input_width, feature_dim, input_channels)


class GRUEncoder(nn.Module):
    """
    Processes a sequence of visual features using GRU frame-by-frame.
    Base state is not processed by GRU, only concatenated at the end.
    """
    def __init__(self, visual_feature_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.visual_feature_dim = visual_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU input: only visual_feature
        self.gru = nn.GRU(visual_feature_dim, hidden_dim, num_layers, batch_first=True)
      
    def forward(self, visual_seq):
        """
        Process sequence frame by frame.
        Args:
            visual_seq: (B, K, visual_feature_dim) - sequence of CNN features
        Returns:
            h: (B, hidden_dim) - final hidden state
        """
        B, K, _ = visual_seq.shape
        device = visual_seq.device
        
        # Initialize hidden state
        h = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        
        # Process each frame sequentially
        for t in range(K):
            # Get current frame's visual feature
            visual_t = visual_seq[:, t, :]  # (B, visual_feature_dim)
            visual_t = visual_t.unsqueeze(1)  # (B, 1, visual_feature_dim)
            
            # GRU forward for this timestep
            _, h = self.gru(visual_t, h)
        
        # Return the final hidden state
        return h[-1]  # (B, hidden_dim)


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.LayerNorm(repr_dim),
                                    nn.Linear(repr_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, std=None):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        return mu


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.LayerNorm(repr_dim + action_shape[0]),
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.LayerNorm(repr_dim + action_shape[0]),
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2
