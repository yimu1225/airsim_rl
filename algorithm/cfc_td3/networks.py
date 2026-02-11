import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC
from ..cnn_modules import CNN


class VisualEncoder(CNN):
    """
    VisualEncoder that uses unified CNN from cnn_modules.
    Processes Depth features only for sequence-based algorithms.
    Depth sequences are processed by CNN and then fed to temporal modules.
    """
    def __init__(self, input_height, input_width, feature_dim, input_channels=1):
        super().__init__(input_height, input_width, feature_dim, input_channels=input_channels)


class CFCEncoder(nn.Module):
    """
    Processes sequences using CfC (Closed-form Continuous-time Neural Network).
    Base state is used directly without expansion.
    """
    def __init__(self, base_dim, visual_feature_dim, wiring):
        super().__init__()
        # CfC input: visual_feature + base state
        cfc_input_dim = base_dim + visual_feature_dim
        self.cfc = CfC(cfc_input_dim, wiring, batch_first=True)

    def forward(self, base_seq, visual_seq):
        """
        Process sequence frame by frame through CfC.
        Args:
            base_seq: (B, K, base_dim)
            visual_seq: (B, K, visual_feature_dim)
        Returns:
            h: (B, motor_neurons) - last timestep output
        """
        B, K, _ = base_seq.shape
        device = base_seq.device
        
        # CfC maintains internal state across timesteps
        # We process frame by frame to ensure temporal consistency
        hx = None  # CfC manages its own hidden state
        
        for t in range(K):
            # Get current frame's expanded base state and visual feature
            base_t = base_seq[:, t, :]  # (B, base_dim)
            visual_t = visual_seq[:, t, :]  # (B, visual_feature_dim)
            
            # Concatenate base state and visual feature
            cfc_input_t = torch.cat([base_t, visual_t], dim=-1)  # (B, cfc_input_dim)
            cfc_input_t = cfc_input_t.unsqueeze(1)  # (B, 1, cfc_input_dim)
            
            # CfC forward for this timestep
            output_t, hx = self.cfc(cfc_input_t, hx)
        
        # Return the final output (last timestep)
        return output_t.squeeze(1)  # (B, motor_neurons)


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
