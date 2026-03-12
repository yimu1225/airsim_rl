import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm import Mamba
from Vim.vim.models_mamba import create_block, RMSNorm, layer_norm_fn, rms_norm_fn


class MambaBlock(nn.Module):
    """
    Single Mamba layer wrapper (aligned with ST_SVimTD3 style).
    Input/Output: (B, N, D)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(x)


class TemporalMambaStack(nn.Module):
    """
    Stack of plain Mamba layers for post-Vim temporal enhancement.
    Input/Output: (B, N, D)
    """
    def __init__(self, dim, n_layers=1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_layers = nn.ModuleList([
            MambaBlock(dim=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(max(1, int(n_layers)))
        ])

    def forward(self, x):
        for layer in self.mamba_layers:
            x = layer(x)
        return x


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
        -> CLS token output
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
        temporal_depth=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_cls_token=True,
        if_bidirectional=True,
        if_abs_pos_embed=True,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=True,
        if_bimamba=True,
        bimamba_type="v2",
        if_divide_out=True,
        norm_epsilon=1e-5,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_cls_token = use_cls_token
        self.temporal_depth = max(1, int(temporal_depth))
        self.if_bidirectional = if_bidirectional
        self.if_abs_pos_embed = if_abs_pos_embed
        self.residual_in_fp32 = residual_in_fp32

        if not self.use_cls_token:
            raise ValueError("VisionMamba3D is configured to use CLS token output only. Set use_cls_token=True.")

        # If fused kernels are unavailable, gracefully fall back to unfused path.
        self.fused_add_norm = bool(fused_add_norm and (layer_norm_fn is not None) and ((not rms_norm) or (rms_norm_fn is not None)))
        
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
        if self.if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        # Build official Vim blocks and keep the same scan behavior.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                d_state=d_state,
                ssm_cfg={"d_conv": d_conv, "expand": expand},
                norm_epsilon=norm_epsilon,
                drop_path=inter_dpr[i],
                rms_norm=rms_norm,
                residual_in_fp32=self.residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                if_divide_out=if_divide_out,
            )
            for i in range(depth)
        ])

        # Extra plain Mamba stack after Vim features (same style as ST_SVimTD3).
        self.post_vim_mamba = TemporalMambaStack(
            dim=embed_dim,
            n_layers=self.temporal_depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        norm_cls = nn.LayerNorm if not rms_norm else RMSNorm
        self.norm_f = norm_cls(embed_dim, eps=norm_epsilon)
        
        # Initialize weights
        if self.if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
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
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        token_position = 0
        
        # Add positional encoding
        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # Official Vim forward scan / bidirectional scan logic.
        residual = None
        hidden_states = x

        if not self.if_bidirectional:
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual, inference_params=None)
        else:
            pair_count = len(self.layers) // 2
            for i in range(pair_count):
                hidden_states_f, residual_f = self.layers[i * 2](hidden_states, residual, inference_params=None)
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                    inference_params=None,
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

            # If depth is odd, run the last layer in forward mode.
            if len(self.layers) % 2 == 1:
                hidden_states, residual = self.layers[-1](hidden_states, residual, inference_params=None)

        # Apply additional plain Mamba layers for temporal enhancement.
        hidden_states = self.post_vim_mamba(hidden_states)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        # Fixed to CLS token to match Vim-style usage in this project.
        return hidden_states[:, token_position]
    
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
