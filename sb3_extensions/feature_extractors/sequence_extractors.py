"""Legacy-accurate sequence backbones adapted to SB3 FeatureExtractor APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import NatureCNN
from torch import nn

try:
    from mamba_ssm import Mamba
except Exception as exc:  # pragma: no cover - runtime dependency check
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None

try:
    from Vim.vim.models_mamba import VisionMamba
except Exception as exc:  # pragma: no cover - runtime dependency check
    VisionMamba = None
    _VIM_IMPORT_ERROR = exc
else:
    _VIM_IMPORT_ERROR = None


@dataclass
class SequenceShape:
    seq_len: int
    channels: int
    height: int
    width: int


def _require_mamba():
    if Mamba is None:
        raise ImportError("mamba_ssm is required for this migrated feature extractor.") from _MAMBA_IMPORT_ERROR
    return Mamba


def _require_vim():
    if VisionMamba is None:
        raise ImportError("Vim.vim.models_mamba.VisionMamba is required for this migrated feature extractor.") from _VIM_IMPORT_ERROR
    return VisionMamba


def _param(params: dict[str, Any], key: str, default: Any = None) -> Any:
    return params[key] if key in params else default


class MambaBlock(nn.Module):
    """Temporal Mamba block copied from the legacy Vim/Mamba algorithms."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, pre_norm: bool = True) -> None:
        super().__init__()
        mamba_cls = _require_mamba()
        self.pre_norm = bool(pre_norm)
        self.norm = nn.LayerNorm(dim)
        self.mamba = mamba_cls(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.mamba(self.norm(x) if self.pre_norm else x)


class TemporalMambaStack(nn.Module):
    """Stacked temporal Mamba layers. Input/output shape: (B, T, D)."""

    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand, pre_norm=pre_norm)
                for _ in range(max(1, int(n_layers)))
            ]
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class NatureFrameCNN(nn.Module):
    """SB3 NatureCNN applied frame-wise to depth sequences."""

    def __init__(self, shape: SequenceShape, output_dim: int = 128) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        frame_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(shape.channels, shape.height, shape.width),
            dtype=np.float32,
        )
        self.cnn = NatureCNN(frame_space, features_dim=self.output_dim, normalized_image=True)
        self.repr_dim = self.output_dim

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        batch_size, seq_len, channels, height, width = depth_seq.shape
        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        return self.cnn(frames).view(batch_size, seq_len, self.output_dim)


class VimFrameEncoder(nn.Module):
    """Per-frame VisionMamba encoder from legacy Vim_TD3."""

    def __init__(self, shape: SequenceShape, params: dict[str, Any]) -> None:
        super().__init__()
        vim_cls = _require_vim()
        self.embed_dim = int(_param(params, "st_mamba_embed_dim", 64))
        self.depth = int(_param(params, "st_mamba_depth", 1))
        self.patch_size = int(_param(params, "st_mamba_patch_size", 16))
        self.d_state = int(_param(params, "st_mamba_d_state", 32))
        self.seq_len = int(shape.seq_len)
        self.flatten_all_tokens = bool(_param(params, "st_vim_flatten_all_tokens", True))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim
        self.vim = vim_cls(
            img_size=(shape.height, shape.width),
            patch_size=self.patch_size,
            stride=self.patch_size,
            depth=self.depth,
            embed_dim=self.embed_dim,
            d_state=self.d_state,
            channels=shape.channels,
            num_classes=0,
            if_bidirectional=False,
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            fused_add_norm=True,
            residual_in_fp32=True,
            if_cls_token=True,
            use_middle_cls_token=True,
            final_pool_type="none",
            if_bimamba=True,
            bimamba_type="v2",
            drop_rate=float(_param(params, "st_mamba_drop_rate", 0.0)),
            drop_path_rate=float(_param(params, "st_mamba_drop_path_rate", 0.0)),
        )

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        batch_size, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        frame_tokens = self.vim(frames, return_features=True).view(batch_size, seq_len, self.embed_dim)
        return frame_tokens.reshape(batch_size, seq_len * self.embed_dim) if self.flatten_all_tokens else frame_tokens[:, -1, :]


class STVimEncoder(VimFrameEncoder):
    """Frame-wise VisionMamba followed by temporal Mamba, matching ST_Vim_* algorithms."""

    def __init__(self, shape: SequenceShape, params: dict[str, Any], pre_norm: bool = True) -> None:
        super().__init__(shape, params)
        pre_norm = bool(_param(params, "pre_norm", pre_norm))
        self.d_conv = int(_param(params, "st_mamba_d_conv", 4))
        self.expand = int(_param(params, "st_mamba_expand", 2))
        self.temporal_layers = int(_param(params, "st_mamba_temporal_depth", 2))
        self.concat_cls_before_temporal_mamba = bool(_param(params, "st_vim_concat_cls_before_temporal_mamba", False))
        if self.concat_cls_before_temporal_mamba:
            self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else 1
        else:
            self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim
        temporal_dim = 1 if self.concat_cls_before_temporal_mamba else self.embed_dim
        self.temporal_mamba = TemporalMambaStack(
            dim=temporal_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            pre_norm=pre_norm,
        )

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        batch_size, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        frame_tokens = self.vim(frames, return_features=True).view(batch_size, seq_len, self.embed_dim)
        if self.concat_cls_before_temporal_mamba:
            temporal_tokens = self.temporal_mamba(frame_tokens.reshape(batch_size, seq_len * self.embed_dim, 1))
            return temporal_tokens.reshape(batch_size, seq_len * self.embed_dim) if self.flatten_all_tokens else temporal_tokens[:, -1, :]
        temporal_tokens = self.temporal_mamba(frame_tokens)
        return temporal_tokens.reshape(batch_size, seq_len * self.embed_dim) if self.flatten_all_tokens else temporal_tokens[:, -1, :]


class STSeqVimEncoder(STVimEncoder):
    """ST_Vim with base-state sequence fusion before temporal Mamba."""

    def __init__(self, shape: SequenceShape, params: dict[str, Any], base_dim: int) -> None:
        super().__init__(shape, params)
        if base_dim <= 0:
            raise ValueError("STSeqVimEncoder requires base_dim > 0")
        self.base_dim = int(base_dim)
        self.state_proj_dim = int(_param(params, "st_state_proj_dim", self.embed_dim))
        self.base_proj = nn.Sequential(
            nn.LayerNorm(self.base_dim),
            nn.Linear(self.base_dim, self.state_proj_dim),
            nn.GELU(),
            nn.Linear(self.state_proj_dim, self.embed_dim),
        )
        self.fuse_proj = nn.Sequential(nn.LayerNorm(self.embed_dim * 2), nn.Linear(self.embed_dim * 2, self.embed_dim))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim
        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

    def _prepare_base_seq(self, base: th.Tensor, batch_size: int, seq_len: int) -> th.Tensor:
        if base.dim() == 1:
            return base.view(1, 1, self.base_dim).expand(batch_size, seq_len, self.base_dim)
        if base.dim() == 2:
            return base.unsqueeze(1).expand(batch_size, seq_len, self.base_dim)
        if base.dim() == 3:
            return base
        raise ValueError(f"Unsupported base tensor dim={base.dim()}")

    def forward(self, depth_seq: th.Tensor, base: th.Tensor) -> th.Tensor:
        batch_size, seq_len, channels, height, width = depth_seq.shape
        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        vis_tokens = self.vim(frames, return_features=True).view(batch_size, seq_len, self.embed_dim)
        base_seq = self._prepare_base_seq(base, batch_size, seq_len)
        base_tokens = self.base_proj(base_seq.reshape(batch_size * seq_len, self.base_dim)).view(
            batch_size, seq_len, self.embed_dim
        )
        fused_tokens = self.fuse_proj(th.cat([vis_tokens, base_tokens], dim=-1))
        temporal_tokens = self.temporal_mamba(fused_tokens)
        return temporal_tokens.reshape(batch_size, seq_len * self.embed_dim) if self.flatten_all_tokens else temporal_tokens[:, -1, :]


class STVSeqVimEncoder(STSeqVimEncoder):
    """Vim-only visual encoder plus base-state sequence fusion; no temporal Mamba."""

    def forward(self, depth_seq: th.Tensor, base: th.Tensor) -> th.Tensor:
        batch_size, seq_len, channels, height, width = depth_seq.shape
        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        vis_tokens = self.vim(frames, return_features=True).view(batch_size, seq_len, self.embed_dim)
        base_seq = self._prepare_base_seq(base, batch_size, seq_len)
        base_tokens = self.base_proj(base_seq.reshape(batch_size * seq_len, self.base_dim)).view(
            batch_size, seq_len, self.embed_dim
        )
        fused_tokens = self.fuse_proj(th.cat([vis_tokens, base_tokens], dim=-1))
        return fused_tokens.reshape(batch_size, seq_len * self.embed_dim) if self.flatten_all_tokens else fused_tokens[:, -1, :]


class VideoPatchEmbed(nn.Module):
    """VideoMamba-style 3D patch embedding from STV_Patch_TD3."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_frames) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.in_chans = int(in_chans)
        self.num_frames = int(num_frames)
        self.grid_size = (
            (self.img_size[0] - self.patch_size[0]) // self.patch_size[0] + 1,
            (self.img_size[1] - self.patch_size[1]) // self.patch_size[1] + 1,
        )
        self.num_patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.num_patches = self.num_frames * self.num_patches_per_frame
        self.proj = nn.Conv3d(
            self.in_chans,
            int(embed_dim),
            kernel_size=(1, self.patch_size[0], self.patch_size[1]),
            stride=(1, self.patch_size[0], self.patch_size[1]),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.dim() != 5:
            raise ValueError(f"VideoPatchEmbed expects 5D input (B,T,C,H,W), got {tuple(x.shape)}")
        bsz, frames, channels, height, width = x.shape
        if frames != self.num_frames or channels != self.in_chans or (height, width) != self.img_size:
            raise ValueError("VideoPatchEmbed input shape does not match configured video shape.")
        x = self.proj(x.permute(0, 2, 1, 3, 4))
        bsz, dim, frames_out, grid_h, grid_w = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(bsz, frames_out * grid_h * grid_w, dim)


class STVimVideoPatchEncoder(STVimEncoder):
    """Sequence-level video patch embedding variant from STV_Patch_TD3."""

    def __init__(self, shape: SequenceShape, params: dict[str, Any]) -> None:
        super().__init__(shape, params)
        self.repr_dim = self.embed_dim
        self.vim.patch_embed = VideoPatchEmbed(
            img_size=(shape.height, shape.width),
            patch_size=self.patch_size,
            in_chans=shape.channels,
            embed_dim=self.embed_dim,
            num_frames=shape.seq_len,
        )
        total_tokens = int(self.vim.patch_embed.num_patches + self.vim.num_tokens)
        self.vim.pos_embed = nn.Parameter(th.zeros(1, total_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.vim.pos_embed, std=0.02)
        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        cls_feature = self.vim(depth_seq, return_features=True)
        temporal_tokens = self.temporal_mamba(cls_feature.unsqueeze(1))
        return temporal_tokens.reshape(temporal_tokens.shape[0], -1) if self.flatten_all_tokens else temporal_tokens[:, 0, :]


class MambaOnlyEncoder(nn.Module):
    """Migrated Mamba_TD3 encoder: SB3 NatureCNN per frame + temporal Mamba."""

    def __init__(self, shape: SequenceShape, params: dict[str, Any]) -> None:
        super().__init__()
        self.seq_len = int(shape.seq_len)
        nature_output_dim = int(_param(params, "mamba_cnn_output_dim", 128))
        self.spatial_encoder = NatureFrameCNN(shape, output_dim=nature_output_dim)
        self.embed_dim = self.spatial_encoder.repr_dim
        self.temporal_layers = int(_param(params, "mamba_td3_temporal_depth", 2))
        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=int(_param(params, "mamba_d_state", 16)),
            d_conv=int(_param(params, "mamba_d_conv", 4)),
            expand=int(_param(params, "mamba_expand", 2)),
        )
        self.flatten_all_tokens = bool(_param(params, "mamba_td3_flatten_all_tokens", True))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        batch_size, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        frame_tokens = self.spatial_encoder(depth_seq)
        temporal_tokens = self.temporal_mamba(frame_tokens)
        return temporal_tokens.reshape(batch_size, seq_len * self.embed_dim) if self.flatten_all_tokens else temporal_tokens[:, -1, :]


class DualBranchVideoMambaEncoder(nn.Module):
    """Hierarchical dual-branch VideoMamba encoder from `ST_DualVim_TD3`."""

    def __init__(self, shape: SequenceShape, params: dict[str, Any]) -> None:
        super().__init__()
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, rms_norm_fn
        from timm.models.layers import DropPath, trunc_normal_, to_2tuple

        mamba_cls = _require_mamba()
        self.RMSNorm = RMSNorm
        self.rms_norm_fn = rms_norm_fn
        self.DropPath = DropPath

        class PatchEmbed(nn.Module):
            def __init__(self, img_size, patch_size, kernel_size=1, in_chans=1, embed_dim=768):
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
                return self.proj(x)

        class VisionMambaBlock(nn.Module):
            def __init__(
                self,
                d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                drop_path=0.0,
                residual_in_fp32=True,
            ):
                super().__init__()
                self.d_model = int(d_model)
                self.d_inner = int(d_model * expand)
                self.residual_in_fp32 = bool(residual_in_fp32)
                self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
                self.mamba_fwd = mamba_cls(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
                self.mamba_bwd = mamba_cls(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
                self.out_proj = nn.Linear(self.d_inner, self.d_model)
                self.norm = RMSNorm(self.d_model, eps=1e-5)
                self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

            def forward(self, hidden_states):
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
                y_backward = th.flip(self.mamba_bwd(th.flip(x, dims=[1])), dims=[1])
                y = (y_forward + y_backward) * F.silu(z)
                y = self.out_proj(y)
                return residual.to(dtype=y.dtype) + self.drop_path(y)

        class HierarchicalDualMamba(nn.Module):
            def __init__(
                self,
                d_model,
                n_frames,
                n_patches,
                d_state=16,
                d_conv=4,
                expand=2,
                drop_path=0.0,
                residual_in_fp32=True,
            ):
                super().__init__()
                self.d_model = int(d_model)
                self.d_inner = int(d_model * expand)
                self.n_frames = int(n_frames)
                self.n_patches = int(n_patches)
                self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
                self.spatial_branch = VisionMambaBlock(
                    self.d_inner,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=1,
                    drop_path=drop_path,
                    residual_in_fp32=residual_in_fp32,
                )
                self.temporal_branch = VisionMambaBlock(
                    self.d_inner,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=1,
                    drop_path=drop_path,
                    residual_in_fp32=residual_in_fp32,
                )
                self.out_proj = nn.Linear(self.d_inner, self.d_model)
                self.cls_gate = nn.Linear(self.d_inner * 2, self.d_inner)
                self.cls_proj = nn.Linear(self.d_inner, self.d_model)
                self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

            def forward(self, hidden_states, cls_token):
                bsz, frames, tokens, dim = hidden_states.shape
                residual = hidden_states
                flat = hidden_states.reshape(bsz, frames * tokens, dim)
                xz = self.in_proj(flat)
                x, z = xz.chunk(2, dim=-1)
                cls_inner, _ = self.in_proj(cls_token).chunk(2, dim=-1)
                x = x.reshape(bsz, frames, tokens, self.d_inner)
                z = F.silu(z.reshape(bsz, frames, tokens, self.d_inner))

                spatial_seq = th.cat([cls_inner, x.reshape(bsz, frames * tokens, self.d_inner)], dim=1)
                y_spatial_seq = self.spatial_branch(spatial_seq)
                cls_spatial = y_spatial_seq[:, :1, :]
                y_spatial = y_spatial_seq[:, 1:, :].reshape(bsz, frames, tokens, self.d_inner)

                temporal_seq = th.cat(
                    [cls_inner, x.permute(0, 2, 1, 3).reshape(bsz, tokens * frames, self.d_inner)],
                    dim=1,
                )
                y_temporal_seq = self.temporal_branch(temporal_seq)
                cls_temporal = y_temporal_seq[:, :1, :]
                y_temporal = y_temporal_seq[:, 1:, :].reshape(bsz, tokens, frames, self.d_inner).permute(0, 2, 1, 3)

                y = ((y_spatial * z) + (y_temporal * z)).reshape(bsz, frames * tokens, self.d_inner)
                y = self.drop_path(self.out_proj(y)).reshape(bsz, frames, tokens, dim)
                out = residual + y

                cls_pair = th.cat([cls_spatial, cls_temporal], dim=-1)
                cls_gate = th.sigmoid(self.cls_gate(cls_pair))
                cls_combined = cls_gate * cls_spatial + (1.0 - cls_gate) * cls_temporal
                cls_out = cls_token + self.drop_path(self.cls_proj(cls_combined))
                return out, cls_out

        self.num_frames = int(shape.seq_len)
        self.embed_dim = int(_param(params, "st_mamba_embed_dim", 64))
        self.depth = max(1, int(_param(params, "st_mamba_depth", 1)))
        self.patch_size = int(_param(params, "st_mamba_patch_size", 16))
        self.d_state = int(_param(params, "st_mamba_d_state", 32))
        self.d_conv = int(_param(params, "st_mamba_d_conv", 4))
        self.expand = int(_param(params, "st_mamba_expand", 2))
        self.drop_rate = float(_param(params, "st_mamba_drop_rate", 0.0))
        self.drop_path_rate = float(_param(params, "st_mamba_drop_path_rate", 0.1))
        self.residual_in_fp32 = True

        self.patch_embed = PatchEmbed(
            img_size=(shape.height, shape.width),
            patch_size=self.patch_size,
            kernel_size=1,
            in_chans=shape.channels,
            embed_dim=self.embed_dim,
        )
        self.tokens_per_frame = self.patch_embed.num_patches
        self.spatial_pos_embed = nn.Parameter(th.zeros(1, self.tokens_per_frame, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(th.zeros(1, self.num_frames, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in th.linspace(0, self.drop_path_rate, self.depth)]
        self.layers = nn.ModuleList(
            [
                HierarchicalDualMamba(
                    d_model=self.embed_dim,
                    n_frames=self.num_frames,
                    n_patches=self.tokens_per_frame,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                    drop_path=dpr[i],
                    residual_in_fp32=self.residual_in_fp32,
                )
                for i in range(self.depth)
            ]
        )
        self.norm_f = RMSNorm(self.embed_dim, eps=1e-5)
        self.cls_token = nn.Parameter(th.zeros(1, 1, self.embed_dim))
        self.repr_dim = self.embed_dim

        trunc_normal_(self.spatial_pos_embed, std=0.02)
        trunc_normal_(self.temporal_pos_embedding, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        if depth_seq.dim() != 5:
            raise ValueError(f"Expected depth sequence (B,T,C,H,W), got {tuple(depth_seq.shape)}")
        bsz, frames_in, channels_in, height, width = depth_seq.shape
        if frames_in != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {frames_in}")
        x = self.patch_embed(depth_seq.permute(0, 2, 1, 3, 4))
        bsz, channels, frames, grid_h, grid_w = x.shape
        n_tokens = grid_h * grid_w
        token_4d = x.permute(0, 2, 3, 4, 1).reshape(bsz, frames, n_tokens, channels)
        spatial_pos = self.spatial_pos_embed.reshape(1, 1, n_tokens, channels)
        temporal_pos = self.temporal_pos_embedding[:, :frames, :].reshape(1, frames, 1, channels)
        token_4d = self.pos_drop(token_4d + spatial_pos + temporal_pos)
        cls_token = self.cls_token.expand(bsz, -1, -1)
        for layer in self.layers:
            token_4d, cls_token = layer(token_4d, cls_token)
        return self.norm_f(cls_token.squeeze(1))


class PaperFeatureExtractor(nn.Module):
    """Liu Bokai et al. self-supervised attention feature extractor for LSTM_SAC."""

    def __init__(self, input_height: int, input_width: int, feature_dim: int = 64) -> None:
        super().__init__()
        if int(feature_dim) != 64:
            raise ValueError("The LSTM_SAC paper feature extractor uses a fixed 64-channel image feature.")
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.feature_dim = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_reduce = nn.Conv2d(64, 32, kernel_size=1)
        self.attn_expand = nn.Conv2d(32, 64, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def preprocess_depth(depth: th.Tensor) -> th.Tensor:
        depth = depth.float()
        if depth.numel() > 0 and float(depth.detach().max().item()) > 1.5:
            depth = depth / 255.0
        return depth.clamp(0.0, 1.0)

    def encode_frames(self, frames: th.Tensor) -> th.Tensor:
        frames = self.preprocess_depth(frames)
        en = F.relu(self.conv1(frames), inplace=True)
        en = F.relu(self.conv2(en), inplace=True)
        en = F.relu(self.conv3(en), inplace=True)
        e1 = self.avg_pool(en)
        e2 = F.relu(self.attn_reduce(e1), inplace=True)
        e3 = th.sigmoid(self.attn_expand(e2))
        return self.avg_pool(en * (e1 + e3)).flatten(1)

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        batch_size, seq_len, channels, height, width = depth_seq.shape
        if channels != 1:
            raise ValueError(f"Expected single-channel depth frames, got C={channels}")
        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        omega = self.encode_frames(frames)
        return omega.view(batch_size, seq_len, self.feature_dim)


class LSTMPaperEncoder(nn.Module):
    """Paper image extractor followed by the FC/LSTM/FC stack used by LSTM_SAC."""

    def __init__(self, shape: SequenceShape, params: dict[str, Any]) -> None:
        super().__init__()
        self.seq_len = int(shape.seq_len)
        feature_dim = int(_param(params, "lstm_sac_feature_dim", 64))
        hidden_dim = int(_param(params, "lstm_sac_hidden_dim", 512))
        self.image = PaperFeatureExtractor(shape.height, shape.width, feature_dim=feature_dim)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.repr_dim = hidden_dim

    def forward(self, depth_seq: th.Tensor) -> th.Tensor:
        features = self.image(depth_seq)
        x = self.input_norm(features)
        x = F.relu(self.fc1(x), inplace=True)
        x, _ = self.lstm(x)
        return F.relu(self.fc2(x[:, -1, :]), inplace=True)


class AirSimSequenceExtractor(BaseFeaturesExtractor):
    """SB3 adapter that wraps a migrated legacy visual/temporal encoder."""

    encoder_cls: type[nn.Module] | None = None
    uses_base_sequence: bool = False

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        base_feature_dim: int = 32,
        algorithm_params: dict[str, Any] | None = None,
        append_base: bool = True,
        **encoder_kwargs,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(f"{self.__class__.__name__} expects Dict observation space.")
        if "depth" not in observation_space.spaces or "base" not in observation_space.spaces:
            raise KeyError("AirSim observation space must contain 'depth' and 'base'.")
        super().__init__(observation_space, features_dim=features_dim)
        self.depth_space = observation_space.spaces["depth"]
        self.base_space = observation_space.spaces["base"]
        self.depth_shape = tuple(int(dim) for dim in self.depth_space.shape)
        self.shape = self._infer_sequence_shape(self.depth_shape)
        self.base_dim = int(self.base_space.shape[0])
        self.params = dict(algorithm_params or {})
        self.params.update(encoder_kwargs)
        self.append_base = bool(append_base and not self.uses_base_sequence)
        self.depth_scale = self._infer_depth_scale(self.depth_space)

        if self.encoder_cls is None:
            raise NotImplementedError("encoder_cls must be set by subclasses.")
        if self.uses_base_sequence:
            self.encoder = self.encoder_cls(self.shape, self.params, self.base_dim)
        else:
            self.encoder = self.encoder_cls(self.shape, self.params)

        if self.append_base:
            vision_output_dim = int(features_dim) - int(base_feature_dim)
            if vision_output_dim <= 0:
                raise ValueError("features_dim must be greater than base_feature_dim when appending base features.")
            self.vision_proj = nn.Sequential(nn.Linear(self.encoder.repr_dim, vision_output_dim), nn.ReLU())
            self.base_net = nn.Sequential(nn.Linear(self.base_dim, int(base_feature_dim)), nn.ReLU())
        else:
            self.vision_proj = nn.Sequential(nn.Linear(self.encoder.repr_dim, int(features_dim)), nn.ReLU())
            self.base_net = nn.Identity()

    @staticmethod
    def _infer_sequence_shape(depth_shape: tuple[int, ...]) -> SequenceShape:
        if len(depth_shape) == 3:
            seq_len, height, width = depth_shape
            return SequenceShape(seq_len=seq_len, channels=1, height=height, width=width)
        if len(depth_shape) >= 4:
            seq_len = depth_shape[0]
            channels = int(np.prod(depth_shape[1:-2]))
            height, width = depth_shape[-2:]
            return SequenceShape(seq_len=seq_len, channels=channels, height=height, width=width)
        raise ValueError(f"Depth observation must be (T,H,W) or (T,C,H,W), got {depth_shape}")

    @staticmethod
    def _infer_depth_scale(depth_space: spaces.Box) -> float:
        high = np.asarray(depth_space.high, dtype=np.float32)
        finite = high[np.isfinite(high)]
        if finite.size == 0:
            return 255.0
        value = float(np.max(finite))
        return value if value > 1.0 else 1.0

    def _prepare_depth(self, depth: th.Tensor) -> th.Tensor:
        depth = depth.float()
        if depth.dim() == len(self.depth_shape):
            depth = depth.unsqueeze(0)
        batch_size = depth.shape[0]
        depth = depth.reshape(batch_size, self.shape.seq_len, self.shape.channels, self.shape.height, self.shape.width)
        if self.depth_scale > 1.0:
            depth = depth / self.depth_scale
        return depth

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        depth = self._prepare_depth(observations["depth"])
        base = observations["base"].float()
        if base.dim() == 1:
            base = base.unsqueeze(0)
        if self.uses_base_sequence:
            vision_features = self.encoder(depth, base)
        else:
            vision_features = self.encoder(depth)
        vision_features = self.vision_proj(vision_features)
        if self.append_base:
            return th.cat([vision_features, self.base_net(base)], dim=1)
        return vision_features
