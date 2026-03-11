import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """
    Mamba Block with LayerNorm and Residual connection.
    Standard practice for stacking Mamba layers to prevent instability/NaNs.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for video input.
    
    Input: (B, C, T, H, W) - batch, channels, time, height, width
    Output: (B, N, D) - batch, num_patches, embed_dim
    """
    def __init__(self, patch_size=(2, 4, 4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size  # (T, H, W)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            x: (B, N, D) where N = T' * H' * W'
        """
        B, C, T, H, W = x.shape
        
        # Pad if needed
        pad_t = (self.patch_size[0] - T % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        
        # 3D conv: (B, C, T, H, W) -> (B, D, T', H', W')
        x = self.proj(x)
        
        # Flatten spatial and temporal: (B, D, T', H', W') -> (B, D, N) -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x


class VisionMamba3D(nn.Module):
    """
    Pure 3D Vision Mamba for spatiotemporal feature extraction.
    
    Architecture:
        Input: (B, C, T, H, W)
        -> 3D Patch Embedding
        -> Positional Encoding
        -> Mamba Blocks (统一处理时空)
        -> Pooling
        -> Output: (B, D)
    """
    def __init__(
        self,
        img_size=(64, 64),
        patch_size=(2, 4, 4),
        in_chans=1,
        seq_len=4,
        embed_dim=96,
        depth=4,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_cls_token=True,
        pool_type='mean'  # 'mean', 'max', 'cls', 'last'
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_cls_token = use_cls_token
        self.pool_type = pool_type
        
        # 3D Patch Embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm
        )
        
        # Calculate number of patches
        self.patches_t = (seq_len + patch_size[0] - 1) // patch_size[0]
        self.patches_h = (img_size[0] + patch_size[1] - 1) // patch_size[1]
        self.patches_w = (img_size[1] + patch_size[2] - 1) // patch_size[2]
        self.num_patches = self.patches_t * self.patches_h * self.patches_w
        
        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_patches += 1
        
        # Positional encoding (3D-aware)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            x: (B, D) - pooled features
        """
        B = x.shape[0]
        
        # 3D Patch Embedding: (B, C, T, H, W) -> (B, N, D)
        x = self.patch_embed(x)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Pooling
        if self.pool_type == 'cls' and self.use_cls_token:
            return x[:, 0]
        elif self.pool_type == 'mean':
            return x.mean(dim=1)
        elif self.pool_type == 'max':
            return x.max(dim=1)[0]
        elif self.pool_type == 'last':
            return x[:, -1]
        else:
            return x.mean(dim=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) or (B, T, C, H, W)
        Returns:
            x: (B, D)
        """
        # Handle input format: (B, T, C, H, W) -> (B, C, T, H, W)
        if x.dim() == 5 and x.shape[1] == self.seq_len:
            x = x.permute(0, 2, 1, 3, 4)
            
        return self.forward_features(x)


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.input_norm = nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
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

    def forward(self, x, action):
        xu = torch.cat([x, action], dim=-1)
        xu = self.input_norm(xu)
        return self.net(xu)
