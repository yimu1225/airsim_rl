import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn


class PatchEmbed(nn.Module):
    """VideoMamba-style 3D patch embedding (tubelet + spatial patch)."""

    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.proj(x)


class VisionMambaBlock(nn.Module):
    """
    Bi-directional selective-scan block used by both spatial and temporal branches.
    This keeps the forward/backward scanning + gating structure aligned with Vim style.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_path=0.0,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_inner = int(d_model * expand)

        # Keep residual_in_fp32 for numerical behavior parity with Vim.
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.fused_add_norm = True
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        # Always use mamba_ssm implementation (no handwritten SSM fallback).
        self.mamba_fwd = Mamba(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,
        )
        self.mamba_bwd = Mamba(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,
        )

        self.out_proj = nn.Linear(self.d_inner, self.d_model)

        self.norm = RMSNorm(self.d_model, eps=1e-5)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states):
        # hidden_states: (B, L, d_model)
        hidden_states, residual = rms_norm_fn(
            hidden_states,
            self.norm.weight,
            self.norm.bias,
            residual=None,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm.eps,
        )

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        y_forward = self.mamba_fwd(x)
        y_backward = torch.flip(self.mamba_bwd(torch.flip(x, dims=[1])), dims=[1])
        y = (y_forward + y_backward) * F.silu(z)

        y = self.out_proj(y)
        residual = residual.to(dtype=y.dtype)
        return residual + self.drop_path(y)


class HierarchicalDualMamba(nn.Module):
    """
    Macro level:
      - Spatial branch: per-frame sequence scan over spatial tokens.
      - Temporal branch: per-token sequence scan over frames.
      - Shared gate z applied to both branches, then summed.

    Micro level:
      - Each branch is a VisionMambaBlock with bidirectional scanning.
    """

    def __init__(
        self,
        d_model,
        n_frames,
        n_patches,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_path=0.0,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_inner = int(d_model * expand)
        self.n_frames = int(n_frames)
        self.n_patches = int(n_patches)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)

        self.spatial_branch = VisionMambaBlock(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,
            drop_path=drop_path,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
        )
        self.temporal_branch = VisionMambaBlock(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,
            drop_path=drop_path,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
        )

        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.cls_gate = nn.Linear(self.d_inner * 2, self.d_inner)
        self.cls_proj = nn.Linear(self.d_inner, self.d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states, cls_token):
        # hidden_states: (B, T, N, d_model)
        bsz, frames, tokens, dim = hidden_states.shape
        residual = hidden_states

        flat = hidden_states.reshape(bsz, frames * tokens, dim)
        xz = self.in_proj(flat)
        x, z = xz.chunk(2, dim=-1)
        cls_inner, _ = self.in_proj(cls_token).chunk(2, dim=-1)

        x = x.reshape(bsz, frames, tokens, self.d_inner)
        z = F.silu(z.reshape(bsz, frames, tokens, self.d_inner))

        # Branch A: spatial-priority order (t, n)
        # [CLS, F1P1..F1PN, F2P1..F2PN, ...]
        x_spatial_seq = x.reshape(bsz, frames * tokens, self.d_inner)
        spatial_seq = torch.cat([cls_inner, x_spatial_seq], dim=1)
        y_spatial_seq = self.spatial_branch(spatial_seq)
        cls_spatial = y_spatial_seq[:, :1, :]
        y_spatial = y_spatial_seq[:, 1:, :].reshape(bsz, frames, tokens, self.d_inner)

        # Branch B: temporal-priority order (n, t)
        # [CLS, F1P1,F2P1..FTP1, F1P2,F2P2.., ...]
        x_temporal_seq = x.permute(0, 2, 1, 3).reshape(bsz, tokens * frames, self.d_inner)
        temporal_seq = torch.cat([cls_inner, x_temporal_seq], dim=1)
        y_temporal_seq = self.temporal_branch(temporal_seq)
        cls_temporal = y_temporal_seq[:, :1, :]
        y_temporal = y_temporal_seq[:, 1:, :].reshape(bsz, tokens, frames, self.d_inner).permute(0, 2, 1, 3)

        y = (y_spatial * z) + (y_temporal * z)
        y = y.reshape(bsz, frames * tokens, self.d_inner)
        y = self.drop_path(self.out_proj(y))
        y = y.reshape(bsz, frames, tokens, dim)
        out = residual + y

        cls_pair = torch.cat([cls_spatial, cls_temporal], dim=-1)
        cls_gate = torch.sigmoid(self.cls_gate(cls_pair))
        cls_combined = cls_gate * cls_spatial + (1.0 - cls_gate) * cls_temporal
        cls_out = cls_token + self.drop_path(self.cls_proj(cls_combined))

        return out, cls_out


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


class DualBranchVideoMambaEncoder(nn.Module):
    """
    Hierarchical Dual VideoMamba Encoder (对应设计图):
    
    第一层分解 (Macro):
    - Forward Branch (Spatial Scan): 对每帧内部进行空间双向扫描 (VisionMambaBlock)
    - Backward Branch (Temporal Scan): 对每空间位置进行时间双向扫描 (VisionMambaBlock)
    - z Gate: 分别与两个分支输出相乘后相加
    
    第二层分解 (Micro, inside each branch):
    - 复用 Vision Mamba 结构: Forward SSM + Backward SSM (bi-directional)
    
    Patch Embedding 使用 Video Mamba 的方式 (3D Conv).
    """

    def __init__(
        self,
        input_height,
        input_width,
        input_channels=1,
        num_frames=None,
        embed_dim=48,
        depth=2,
        patch_size=8,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    ):
        super().__init__()

        if num_frames is None:
            num_frames = input_channels
        self.num_frames = max(1, int(num_frames))
        self.embed_dim = int(embed_dim)
        self.depth = max(1, int(depth))
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.fused_add_norm = True

        # Video Mamba style Patch Embedding (3D Conv)
        self.patch_embed = PatchEmbed(
            img_size=(input_height, input_width),
            patch_size=patch_size,
            kernel_size=1,  # tubelet_size = 1, process each frame independently
            in_chans=1,
            embed_dim=self.embed_dim,
        )

        self.tokens_per_frame = self.patch_embed.num_patches

        # Positional embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.tokens_per_frame, self.embed_dim)
        )
        self.temporal_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_frames, self.embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Hierarchical Dual Mamba layers
        # Each layer has: Spatial Branch + Temporal Branch + Gating
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        
        self.layers = nn.ModuleList([
            HierarchicalDualMamba(
                d_model=embed_dim,
                n_frames=self.num_frames,
                n_patches=self.tokens_per_frame,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[i],
                rms_norm=True,
                residual_in_fp32=self.residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
            )
            for i in range(self.depth)
        ])

        # Final normalization
        self.norm_f = RMSNorm(self.embed_dim, eps=1e-5)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.repr_dim = self.embed_dim

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.spatial_pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embedding, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        norm_types = (nn.LayerNorm, RMSNorm)
        for module_name, m in self.named_modules():
            # Preserve native Mamba initialization from mamba_ssm for stability/performance.
            if "mamba_fwd" in module_name or "mamba_bwd" in module_name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, norm_types):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: (B, T, H, W) where T is stacked frame dimension in TD3.
        返回: (B, embed_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Expected depth tensor shape (B, T, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {x.shape[1]}")

        bsz = x.shape[0]
        x = x.unsqueeze(1)  # (B, 1, T, H, W)

        # Video Mamba style Patch Embedding: (B, C, T, H', W')
        x = self.patch_embed(x)
        bsz, channels, frames, grid_h, grid_w = x.shape
        n_tokens = grid_h * grid_w

        # (B, T, N, C)
        token_4d = x.permute(0, 2, 3, 4, 1).reshape(bsz, frames, n_tokens, channels)

        # Add spatial + temporal positional embeddings
        spatial_pos = self.spatial_pos_embed.reshape(1, 1, n_tokens, channels)
        temporal_pos = self.temporal_pos_embedding[:, :frames, :].reshape(1, frames, 1, channels)
        token_4d = token_4d + spatial_pos + temporal_pos
        token_4d = self.pos_drop(token_4d)

        # Pass through all layers with single CLS token propagation.
        cls_token = self.cls_token.expand(bsz, -1, -1)
        for layer in self.layers:
            token_4d, cls_token = layer(token_4d, cls_token)

        # Vim-style readout from propagated global CLS token.
        fused = self.norm_f(cls_token.squeeze(1))
        return fused


class Encoder(DualBranchVideoMambaEncoder):
    """
    Compatibility alias for TD3 agent integration.
    """
   

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