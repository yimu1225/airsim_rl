import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from mamba_ssm import Mamba
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as mamba_selective_scan_fn
except Exception:
    mamba_selective_scan_fn = None



# Fused kernels are intentionally disabled in this merged module.
layer_norm_fn, rms_norm_fn = None, None


class RMSNorm(nn.Module):
    """Minimal, device-safe RMSNorm implementation."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        # Keep LayerNorm-like attribute for compatibility with generic init code.
        self.bias = None
        self.eps = eps

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(rms + self.eps)
        return x_norm.to(orig_dtype) * self.weight.to(orig_dtype)


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
        if mamba_selective_scan_fn is None:
            raise RuntimeError(
                "mamba_ssm.ops.selective_scan_interface.selective_scan_fn is unavailable. "
                "Please ensure mamba_ssm (and its CUDA ops) is installed correctly."
            )
        self.d_model = int(d_model)
        self.d_inner = int(d_model * expand)

        # Keep these flags for API compatibility with caller configs.
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.fused_add_norm = bool(fused_add_norm)
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

        use_rms = bool(rms_norm and (RMSNorm is not None))
        self.norm = RMSNorm(self.d_model) if use_rms else nn.LayerNorm(self.d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states):
        # hidden_states: (B, L, d_model)
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        y_forward = self.mamba_fwd(x)
        y_backward = torch.flip(self.mamba_bwd(torch.flip(x, dims=[1])), dims=[1])
        y = (y_forward + y_backward) * F.silu(z)

        y = self.out_proj(y)
        return residual + self.drop_path(y)


class HierarchicalDualMamba(nn.Module):
    """
    Macro level:
      - Spatial branch: per-frame sequence scan over spatial tokens.
      - Temporal branch: per-token sequence scan over frames.
      - Shared gate z applied to both branches, then summed.

    Micro level:
      - Keep two Vim-style branches (spatial + temporal) with independent CLS tokens.
      - Pack 4 scan routes into one grouped selective scan:
        spatial forward/backward + temporal forward/backward.
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
        dt_rank="auto",
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
        force_fp32=True,
    ):
        super().__init__()
        if mamba_selective_scan_fn is None:
            raise RuntimeError(
                "mamba_ssm.ops.selective_scan_interface.selective_scan_fn is unavailable. "
                "Please ensure mamba_ssm (and its CUDA ops) is installed correctly."
            )
        self.d_model = int(d_model)
        self.d_inner = int(d_model * expand)
        self.n_frames = int(n_frames)
        self.n_patches = int(n_patches)
        self.d_state = int(d_state)
        self.k_group = 4
        self.dt_rank = int(math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank)
        self.force_fp32 = bool(force_fp32)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        use_rms = bool(rms_norm and (RMSNorm is not None))
        norm_cls = RMSNorm if use_rms else nn.LayerNorm
        self.spatial_norm = norm_cls(self.d_inner)
        self.temporal_norm = norm_cls(self.d_inner)

        # Branch-specific CLS tokens:
        # spatial branch uses CLS_A; temporal branch uses CLS_B.
        self.cls_spatial = nn.Parameter(torch.zeros(1, 1, self.d_inner))
        self.cls_temporal = nn.Parameter(torch.zeros(1, 1, self.d_inner))
        trunc_normal_(self.cls_spatial, std=0.02)
        trunc_normal_(self.cls_temporal, std=0.02)

        # Per-route scan parameters (4 routes):
        # 0: spatial fwd, 1: spatial bwd, 2: temporal fwd, 3: temporal bwd
        self.x_proj_weight = nn.Parameter(
            torch.empty(self.k_group, self.dt_rank + 2 * self.d_state, self.d_inner)
        )
        self.dt_projs_weight = nn.Parameter(
            torch.empty(self.k_group, self.d_inner, self.dt_rank)
        )
        self.dt_projs_bias = nn.Parameter(torch.empty(self.k_group, self.d_inner))
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).view(1, 1, self.d_state)
        A = A.repeat(self.k_group, self.d_inner, 1).contiguous()
        self.A_logs = nn.Parameter(torch.log(A).view(self.k_group * self.d_inner, self.d_state))
        self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner, dtype=torch.float32))

        self.spatial_out_proj = nn.Linear(self.d_inner, self.d_inner)
        self.temporal_out_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        dt = torch.exp(
            torch.rand(self.k_group, self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt)

    def forward(self, hidden_states):
        # hidden_states: (B, T, N, d_model)
        bsz, frames, tokens, dim = hidden_states.shape
        residual = hidden_states

        flat = hidden_states.reshape(bsz, frames * tokens, dim)
        xz = self.in_proj(flat)
        x, z = xz.chunk(2, dim=-1)

        x = x.reshape(bsz, frames, tokens, self.d_inner)
        z = F.silu(z.reshape(bsz, frames, tokens, self.d_inner))

        # Branch A: spatial-priority order (t, n)
        # [CLS_A, F1P1..F1PN, F2P1..F2PN, ...]
        l = frames * tokens
        cls_spatial = self.cls_spatial.expand(bsz, -1, -1)
        spatial_seq = torch.cat([cls_spatial, x.reshape(bsz, l, self.d_inner)], dim=1)

        # Branch B: temporal-priority order (n, t)
        # [CLS_B, F1P1,F2P1..FTP1, F1P2,F2P2.., ...]
        cls_temporal = self.cls_temporal.expand(bsz, -1, -1)
        temporal_seq = torch.cat(
            [cls_temporal, x.permute(0, 2, 1, 3).reshape(bsz, l, self.d_inner)],
            dim=1,
        )

        spatial_seq_n = self.spatial_norm(spatial_seq)
        temporal_seq_n = self.temporal_norm(temporal_seq)

        # Pack 4 routes and run one grouped selective scan.
        # routes: [spatial fwd, spatial bwd, temporal fwd, temporal bwd]
        routes = torch.stack(
            [
                spatial_seq_n,
                spatial_seq_n.flip(dims=[1]),
                temporal_seq_n,
                temporal_seq_n.flip(dims=[1]),
            ],
            dim=1,
        )  # (B, 4, L+1, D)
        route_len = routes.shape[2]
        routes_scan = routes.permute(0, 1, 3, 2).contiguous()  # (B, 4, D, L+1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", routes_scan, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        u = routes_scan.reshape(bsz, -1, route_len)
        delta = dts.contiguous().reshape(bsz, -1, route_len)
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        delta_bias = self.dt_projs_bias.reshape(-1).float()

        if self.force_fp32:
            u = u.float()
            delta = delta.float()
            Bs = Bs.float()
            Cs = Cs.float()

        ys = mamba_selective_scan_fn(
            u,
            delta,
            As,
            Bs.contiguous(),
            Cs.contiguous(),
            Ds,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
        ).view(bsz, self.k_group, self.d_inner, route_len)

        spatial_out = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # (B, D, L+1)
        temporal_out = ys[:, 2] + ys[:, 3].flip(dims=[-1])  # (B, D, L+1)
        spatial_out = spatial_out.transpose(1, 2).contiguous()  # (B, L+1, D)
        temporal_out = temporal_out.transpose(1, 2).contiguous()  # (B, L+1, D)

        # Branch-local residual + projection (Vim-like branch identity kept).
        spatial_seq = spatial_seq + self.drop_path(self.spatial_out_proj(spatial_out))
        temporal_seq = temporal_seq + self.drop_path(self.temporal_out_proj(temporal_out))

        y_spatial = spatial_seq[:, 1:, :].reshape(bsz, frames, tokens, self.d_inner)
        y_temporal = temporal_seq[:, 1:, :].reshape(bsz, tokens, frames, self.d_inner).permute(0, 2, 1, 3)

        y = (y_spatial * z) + (y_temporal * z)
        y = y.reshape(bsz, frames * tokens, self.d_inner)
        y = self.drop_path(self.out_proj(y))
        y = y.reshape(bsz, frames, tokens, dim)
        return residual + y


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
    - Spatial Branch: 空间优先序列 (t, n)
    - Temporal Branch: 时间优先序列 (n, t)
    - z Gate: 两分支输出加权融合
    
    第二层分解 (Micro, inside each branch):
    - 在单次 grouped selective scan 中并行完成:
      空间前向/后向 + 时间前向/后向 四路扫描
    
    Patch Embedding 使用 Video Mamba 的方式 (3D Conv).
    """

    def __init__(
        self,
        input_height,
        input_width,
        input_channels=1,
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

        self.num_frames = max(1, int(input_channels))
        self.embed_dim = int(embed_dim)
        self.depth = max(1, int(depth))
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.use_rms_norm = bool(rms_norm and (RMSNorm is not None))

        # If fused kernels are unavailable, gracefully fall back.
        self.fused_add_norm = bool(
            fused_add_norm
            and (layer_norm_fn is not None)
            and ((not self.use_rms_norm) or (rms_norm_fn is not None))
        )

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
                rms_norm=self.use_rms_norm,
                residual_in_fp32=self.residual_in_fp32,
                fused_add_norm=self.fused_add_norm,
            )
            for i in range(self.depth)
        ])

        # Final normalization
        norm_cls = RMSNorm if self.use_rms_norm else nn.LayerNorm
        self.norm_f = norm_cls(self.embed_dim, eps=1e-5)

        self.repr_dim = self.embed_dim

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.spatial_pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embedding, std=0.02)
        norm_types = (nn.LayerNorm,) if RMSNorm is None else (nn.LayerNorm, RMSNorm)
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

        # Pass through Hierarchical Dual Mamba layers
        # Each layer: Spatial Scan Branch + Temporal Scan Branch + Gating
        for layer in self.layers:
            token_4d = layer(token_4d)

        # Final normalization
        token_4d = self.norm_f(token_4d)

        # Global average pooling over time and space dimensions
        # (B, T, N, C) -> (B, C)
        fused = token_4d.mean(dim=(1, 2))
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
