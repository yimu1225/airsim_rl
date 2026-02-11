import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from ..cnn_modules import CNN, BaseStateExpander
from ..D_LGMD import D_LGMD




class DepthEncoder(nn.Module):
    """
    Depth Encoder for temporal processing with GRU.
    Only processes depth images for the temporal module.
    """
    def __init__(self, input_height, input_width, feature_dim):
        super().__init__()
        
        # Depth CNN (1 channel input)
        self.depth_cnn = CNN(input_height, input_width, feature_dim)
        
        self.repr_dim = feature_dim

    def forward(self, depth):
        # depth: (B*K, 1, H, W) - flattened temporal dimension
        
        # Depth path
        d = self.depth_cnn(depth)
        return d


class MotionEncoder(nn.Module):
    """
    Motion Encoder for direct concatenation with temporal output.
    Processes motion features from D-LGMD separately.
    """
    def __init__(self, input_height, input_width, feature_dim):
        super().__init__()
        
        # Motion CNN (1 channel input)
        self.motion_cnn = CNN(input_height, input_width, feature_dim)
        
        self.repr_dim = feature_dim

    def forward(self, motion_map):
        # motion_map: (B*K, 1, H, W) - flattened temporal dimension
        
        # Motion path
        m = self.motion_cnn(motion_map)
        return m


class GRUEncoder(nn.Module):
    """
    Processes sequences frame-by-frame with GRU.
    For D-LGMD variant: combines base state and depth features only.
    Motion features are processed separately and concatenated later.
    """
    def __init__(self, expanded_base_dim, depth_feature_dim, hidden_dim, num_layers=1):
        super().__init__()
        # Only depth features go into GRU
        gru_input_dim = expanded_base_dim + depth_feature_dim
        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers, batch_first=True)
      
    def forward(self, base_expanded_seq, depth_visual_seq):
        """
        Args:
            base_expanded_seq: (B, K, expanded_base_dim)
            depth_visual_seq: (B, K, depth_feature_dim) - only depth features
        Returns:
            final_hidden: (B, hidden_dim)
        """
        # Concatenate and process frame by frame
        combined = torch.cat([base_expanded_seq, depth_visual_seq], dim=2)  # (B, K, gru_input_dim)
        
        # GRU processing
        self.gru.flatten_parameters()
        output, hn = self.gru(combined)
        
        # Return final hidden state
        return hn[-1]  # (B, hidden_dim)


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
