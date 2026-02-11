import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from ..cnn_modules import CNN, BaseStateExpander

# from perception.d_lgmd import LGMD # Moved definition here as requested


class VisualEncoder(CNN):
    """
    VisualEncoder that uses unified CNN from cnn_modules.
    Processes Depth features only for sequence-based algorithms.
    Depth sequences are processed by CNN and then fed to temporal modules.
    """
    def __init__(self, input_height, input_width, feature_dim, input_channels=1):
        super().__init__(input_height, input_width, feature_dim, input_channels=input_channels)


class GRUEncoder(nn.Module):
    """
    Processes a sequence of visual features and base states using GRU.
    Each frame is processed sequentially: v_t (CNN feature) + s_t (base state) -> h_t
    """
    def __init__(self, expanded_base_dim, visual_feature_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.expanded_base_dim = expanded_base_dim
        self.visual_feature_dim = visual_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU input: visual_feature + expanded_base
        gru_input_dim = expanded_base_dim + visual_feature_dim
        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers, batch_first=True)
      
    def forward(self, base_expanded_seq, visual_seq):
        """
        Process sequence frame by frame.
        Args:
            base_expanded_seq: (B, K, expanded_base_dim) - sequence of expanded base states
            visual_seq: (B, K, visual_feature_dim) - sequence of CNN features
        Returns:
            h: (B, hidden_dim) - final hidden state
        """
        B, K, _ = base_expanded_seq.shape
        device = base_expanded_seq.device
        
        # Initialize hidden state
        h = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)
        
        # Process each frame sequentially
        for t in range(K):
            # Get current frame's expanded base state and visual feature
            base_expanded_t = base_expanded_seq[:, t, :]  # (B, expanded_base_dim)
            visual_t = visual_seq[:, t, :]  # (B, visual_feature_dim)
            
            # Concatenate expanded base state and visual feature
            gru_input_t = torch.cat([base_expanded_t, visual_t], dim=-1)  # (B, gru_input_dim)
            gru_input_t = gru_input_t.unsqueeze(1)  # (B, 1, gru_input_dim)
            
            # GRU forward for this timestep
            _, h = self.gru(gru_input_t, h)
        
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


class MetaNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
