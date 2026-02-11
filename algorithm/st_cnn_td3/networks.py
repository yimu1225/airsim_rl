import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add workspace root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(current_dir))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Import CNN modules

from ..cnn_modules import CNN
CNN_AVAILABLE = True


# Import Mamba from virtual environment
from mamba_ssm import Mamba

# Define a simple config class for compatibility
class MambaConfig:
    def __init__(self, d_model, n_layers=1, d_state=16, expand_factor=2, d_conv=4):
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_conv = d_conv

MAMBA_AVAILABLE = True


class TemporalMambaBlock(nn.Module):
    """
    1D Mamba Block for Temporal processing.
    Input/Output: (Batch, SeqLen, Dim)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if MAMBA_AVAILABLE:
            config = MambaConfig(
                d_model=dim,
                n_layers=1,
                d_state=d_state,
                expand_factor=expand,
                d_conv=d_conv
            )
            self.mamba = Mamba(config)
        else:
            self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x):
        if MAMBA_AVAILABLE:
            return self.mamba(x)
        else:
            out, _ = self.gru(x)
            return out


class SpatialSelfAttention(nn.Module):
    """
    Spatial Self-Attention to refine features.
    Input/Output: (Batch, PatchNum, Dim)
    """
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x


class STE_CNN_Encoder(nn.Module):
    """
    Spatio-Temporal Encoder with CNN (STE-CNN)
    Input: (B, T, C, H, W)
    Output: (B, D)
    """
    def __init__(self, 
                 img_size, 
                 in_chans=1, 
                 feature_dim=128,
                 args=None):
        super().__init__()
        
        # 1. Spatial: CNN
        self.spatial_encoder = CNN(
            input_height=img_size[0],
            input_width=img_size[1], 
            feature_dim=feature_dim,
            input_channels=in_chans
        )
        
        self.hidden_dim = feature_dim # D
        
        # 2. Temporal: 1D Mamba
        self.temporal_encoder = TemporalMambaBlock(
            dim=self.hidden_dim,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand
        )
        
        # 3. Spatial Refine: Self-Attention
        # Ensure num_heads divides hidden_dim
        if self.hidden_dim % args.vmamba_num_heads != 0:
            # Adjust num_heads if necessary, or let it fail? 
            # NatureCNN output channels is 64. 64 % 4 == 0. 64 % 8 == 0. Usually fine.
            pass
            
        self.spatial_refine = SpatialSelfAttention(
            dim=self.hidden_dim,
            num_heads=args.vmamba_num_heads,
            dropout=args.attention_dropout
        )
        
        # Output dim is self.hidden_dim
        self.out_dim = self.hidden_dim

    def forward(self, x):
        # Input: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Step 1: Spatial (Merge B & T)
        x = x.view(B * T, C, H, W)
        x = self.spatial_encoder(x) # -> (B*T, D) - CNN returns (B*T, D)
        
        B_T, D = x.shape
        L = 1  # CNN returns flattened features, not spatial map
        
        # Step 2: Temporal
        # Reshape to (B, T, D) for temporal processing
        x = x.view(B, T, D)
        # x = x.permute(1, 0, 2) # REMOVED: Do not permute, keep (B, T, D) for Mamba/GRU
        
        # Pass through 1D Mamba (processes sequence of each batch)
        x = self.temporal_encoder(x) # -> (B, T, D)
        
        # Take Last Token
        x = x[:, -1, :] # -> (B, D)
        
        # Step 3: Spatial Refine
        # Reshape to (B, 1, D) for self-attention (treat as single spatial location)
        x = x.view(B, 1, D)
        
        # Self-Attention
        x = self.spatial_refine(x) # -> (B, 1, D)
        
        # Step 4: Pool
        # Global Average Pooling (already just 1 spatial location)
        x = x.squeeze(1) # -> (B, D)
        
        return x


class StateMLP(nn.Module):
    """
    Encode state vector.
    """
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    """
    Actor Head.
    """
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """
    Critic Head (Twin Q-Networks).
    """
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.q1_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, action):
        xu = torch.cat([x, action], dim=1)
        return self.q1_net(xu), self.q2_net(xu)
