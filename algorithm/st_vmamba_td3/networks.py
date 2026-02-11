import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add workspace root to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(current_dir))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Import VMamba from virtual environment
from vmamba.vmamba import VSSBlock, VSSM as VMambaModel

# Create compatibility classes for VMamba
class PatchPartition(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, channel_first=False):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.channel_first = channel_first
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        if not self.channel_first:
            x = x.permute(0, 2, 3, 1)  # (B, H/patch_size, W/patch_size, embed_dim)
        return x

class DownsampleV3(nn.Module):
    def __init__(self, dim=96, out_dim=192):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        # x is (B, H, W, C) - NHWC format
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.down(x)  # (B, out_dim, H/2, W/2)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H/2, W/2, out_dim)
        x = self.norm(x)  # Apply normalization
        return x

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


class VMambaRLTiny(nn.Module):
    """
    Lightweight VMamba for RL.
    Forward returns (B, L, D) feature map.
    """
    def __init__(self, 
                 in_chans=1,
                 args=None):
        # 直接从args读取参数
        hidden_dim = args.vmamba_hidden_dim
        num_vss_blocks = args.vmamba_num_vss_blocks
        drop_path_rate = args.vmamba_drop_path_rate
        layer_scale_init = args.vmamba_layer_scale_init
        patch_size = args.vmamba_patch_size
        ssm_d_state = args.vmamba_ssm_d_state
        ssm_ratio = args.vmamba_ssm_ratio
        mlp_ratio = args.vmamba_mlp_ratio
        super().__init__()
        
        total_depth = sum(num_vss_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Stem
        self.stem = PatchPartition(
            in_chans=in_chans, 
            embed_dim=hidden_dim, 
            patch_size=patch_size,
            channel_first=False
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_dim = hidden_dim
        offset = 0

        # 动态构建 Stages
        for i, num_blocks in enumerate(num_vss_blocks):
            # 构建当前 Stage 的 Blocks
            stage_blocks = nn.Sequential(
                *[VSSBlock(
                    hidden_dim=current_dim, 
                    drop_path=dpr[offset + j], 
                    layer_scale_init=layer_scale_init,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    mlp_ratio=mlp_ratio
                ) for j in range(num_blocks)]
            )
            self.stages.append(stage_blocks)
            offset += num_blocks
            
            # 构建 Downsample层 (除了最后一个 Stage)
            if i < len(num_vss_blocks) - 1:
                down = DownsampleV3(dim=current_dim, out_dim=current_dim * 2)
                self.downsamples.append(down)
                current_dim = current_dim * 2
        
        self.feature_dim = current_dim
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.stem(x)            # (B, H/4, W/4, C)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        
        # Flatten to (B, L, D)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        return x


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
                n_layers=1, # Single block as requested, or maybe we want a "Layer"
                d_state=d_state,
                expand_factor=expand,
                d_conv=d_conv
            )
            # Mamba class in mambapy usually contains stack of layers. 
            # If we want a single block, n_layers=1 is appropriate.
            self.mamba = Mamba(config)
        else:
            self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x):
        # x: (B, L, D)
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
        # x: (B, L, D)
        # Self-attention expects (B, L, D) if batch_first=True
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x


class STE_Encoder(nn.Module):
    """
    Spatio-Temporal Encoder (STE)
    Input: (B, T, C, H, W)
    Output: (B, D)
    """
    def __init__(self, 
                 img_size, 
                 in_chans=1, 
                 args=None):
        super().__init__()
        
        # 1. Spatial: VMamba
        self.spatial_encoder = VMambaRLTiny(
            in_chans=in_chans,
            args=args
        )
        
        self.hidden_dim = self.spatial_encoder.feature_dim # D
        
        # 2. Temporal: 1D Mamba
        # Processes history for each patch.
        self.temporal_encoder = TemporalMambaBlock(
            dim=self.hidden_dim,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand
        )
        
        # 3. Spatial Refine: Self-Attention
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
        x = self.spatial_encoder(x) # -> (B*T, L, D)
        
        _, L, D = x.shape
        
        # Step 2: Temporal
        # Reshape to (B, T, L, D)
        x = x.view(B, T, L, D)
        # Permute to (B*L, T, D) to treat each patch sequence as a sample
        x = x.permute(0, 2, 1, 3).reshape(B * L, T, D)
        
        # Pass through 1D Mamba
        x = self.temporal_encoder(x) # -> (B*L, T, D)
        
        # Take Last Token
        x = x[:, -1, :] # -> (B*L, D)
        
        # Step 3: Spatial Refine
        # Reshape to (B, L, D)
        x = x.view(B, L, D)
        
        # Self-Attention
        x = self.spatial_refine(x) # -> (B, L, D)
        
        # Step 4: Pool
        # Global Average Pooling on dim=1 (L)
        x = torch.mean(x, dim=1) # -> (B, D)
        
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
    Input: features from encoder.
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
    Input: features and action.
    """
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Q1
        self.q1_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2
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
