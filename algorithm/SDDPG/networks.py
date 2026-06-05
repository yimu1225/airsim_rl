import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cnn_modules import CNN


def build_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU,
              final_activation=None, use_layer_norm_input=False):
    """Build a MLP with configurable hidden layers and optional final activation."""
    layers = []
    if use_layer_norm_input:
        layers.append(nn.LayerNorm(input_dim))
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation(inplace=True))
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class SubNetwork1(nn.Module):
    """
    Subnetwork 1 for perception-related state so (depth/visual features).
    Architecture (matching paper Fig. 4a):
        CNN -> Dense 32 | ReLU -> Dense 16 | ReLU -> Dense ao_dim | Tanh -> ao
    """
    def __init__(self, depth_shape, hidden_dims, out_dim, encoder_output_dim=64):
        super().__init__()
        C, H, W = depth_shape
        self.encoder = CNN(
            input_height=H,
            input_width=W,
            input_channels=C,
            output_dim=encoder_output_dim,
            frame_wise=(C > 1),
            flatten_all_tokens=True,
        )
        # Paper: [32, 16] hidden + Tanh output
        self.mlp = build_mlp(
            self.encoder.repr_dim, hidden_dims, out_dim,
            final_activation=nn.Tanh, use_layer_norm_input=True,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, depth):
        """
        Args:
            depth: (B, C, H, W)
        Returns:
            ao: (B, out_dim)   with Tanh, in [-1, 1]
            so_repr: (B, encoder.repr_dim)
        """
        so_repr = self.encoder(depth)
        ao = self.mlp(so_repr)
        return ao, so_repr


class SubNetwork2(nn.Module):
    """
    Subnetwork 2 for target-related state sg (paper Fig. 4a says input is sg).
    Architecture:
        Input(sg) -> Dense 32 | ReLU -> Dense 16 | ReLU -> Dense ag_dim | Tanh -> ag
    """
    def __init__(self, sg_dim, hidden_dims, out_dim):
        super().__init__()
        self.mlp = build_mlp(
            sg_dim, hidden_dims, out_dim,
            final_activation=nn.Tanh, use_layer_norm_input=True,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, sg):
        """
        Args:
            sg: (B, sg_dim)
        Returns:
            ag: (B, out_dim)   with Tanh, in [-1, 1]
        """
        return self.mlp(sg)


class GlobalActor(nn.Module):
    """
    Global actor network (paper Fig. 4a).
    Architecture:
        concat(ao, ag, S) -> Dense 400 | ReLU -> Dense 300 | ReLU -> Dense(A) | Tanh -> action
    """
    def __init__(self, input_dim, hidden_dims, action_dim):
        super().__init__()
        self.mlp = build_mlp(
            input_dim, hidden_dims, action_dim,
            final_activation=nn.Tanh, use_layer_norm_input=True,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, ao, ag, state_repr):
        """
        Args:
            ao: (B, sub1_out_dim)
            ag: (B, sub2_out_dim)
            state_repr: (B, state_dim)  # full state S = [so_repr, sg_repr, su_repr]
        Returns:
            action: (B, action_dim), in [-1, 1]
        """
        x = torch.cat([ao, ag, state_repr], dim=-1)
        return self.mlp(x)


class Critic(nn.Module):
    """
    Critic (Q-network). Paper says "takes the same structure with the global network".
    Architecture:
        concat(state_repr, action) -> Dense 400 | ReLU -> Dense 300 | ReLU -> 1
    """
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.input_norm = nn.LayerNorm(state_dim + action_dim)
        self.Q = build_mlp(state_dim + action_dim, hidden_dims, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state_repr, action):
        """
        Args:
            state_repr: (B, state_dim)
            action: (B, action_dim)
        Returns:
            q: (B, 1)
        """
        x = torch.cat([state_repr, action], dim=-1)
        x = self.input_norm(x)
        q = self.Q(x)
        return q, None  # Return None as second value for compatibility
