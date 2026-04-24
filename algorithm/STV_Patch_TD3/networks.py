import torch
import torch.nn as nn

from Vim.vim.models_mamba import VisionMamba

from ..ST_Vim_TD3.networks import TemporalMambaStack


class VideoPatchEmbed(nn.Module):
    """
    VideoMamba-style patch embedding.
    Input:  (B, T, C, H, W)
    Output: (B, T*N, D), N = patches per frame
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_frames):
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
            in_channels=self.in_chans,
            out_channels=int(embed_dim),
            kernel_size=(1, self.patch_size[0], self.patch_size[1]),
            stride=(1, self.patch_size[0], self.patch_size[1]),
        )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"VideoPatchEmbed expects 5D input (B,T,C,H,W), got {tuple(x.shape)}")

        bsz, frames, channels, height, width = x.shape
        if frames != self.num_frames:
            raise ValueError(f"Expected num_frames={self.num_frames}, got {frames}")
        if channels != self.in_chans:
            raise ValueError(f"Expected in_chans={self.in_chans}, got {channels}")
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size ({height}, {width}) doesn't match model {self.img_size}."
            )

        # Conv3d expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)  # (B, D, T, H', W')
        bsz, dim, frames_out, grid_h, grid_w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(bsz, frames_out * grid_h * grid_w, dim)
        return x


class STVimVideoPatchEncoder(nn.Module):
    """
    Keep ST_VimTD3 pipeline unchanged, only replace frame-wise 2D patch embedding
    with sequence-level video patch embedding.
    """

    def __init__(
        self,
        input_height,
        input_width,
        input_channels=1,
        num_frames=4,
        embed_dim=48,
        depth=2,
        patch_size=8,
        d_state=16,
        d_conv=4,
        expand=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        temporal_layers=2,
        flatten_all_tokens=True,
    ):
        super().__init__()
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.input_channels = int(input_channels)
        self.seq_len = max(1, int(num_frames))
        self.embed_dim = int(embed_dim)
        self.temporal_layers = int(temporal_layers)
        self.flatten_all_tokens = bool(flatten_all_tokens)
        # CLS-readout path keeps representation size fixed to embed_dim.
        self.repr_dim = self.embed_dim

        self.vim = VisionMamba(
            img_size=(self.input_height, self.input_width),
            patch_size=patch_size,
            stride=patch_size,
            depth=depth,
            embed_dim=self.embed_dim,
            d_state=d_state,
            channels=self.input_channels,
            num_classes=0,
            if_bidirectional=False,
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            fused_add_norm=True,
            residual_in_fp32=True,
            if_cls_token=True,
            use_middle_cls_token=True,
            final_pool_type='none',
            if_bimamba=True,
            bimamba_type="v2",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.vim.patch_embed = VideoPatchEmbed(
            img_size=(self.input_height, self.input_width),
            patch_size=patch_size,
            in_chans=self.input_channels,
            embed_dim=self.embed_dim,
            num_frames=self.seq_len,
        )
        total_tokens = int(self.vim.patch_embed.num_patches + self.vim.num_tokens)
        self.vim.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.vim.pos_embed, std=0.02)

        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def _normalize_depth(self, depth_seq: torch.Tensor) -> torch.Tensor:
        # Accept:
        # - (T, H, W)
        # - (B, T, H, W)
        # - (T, C, H, W)
        # - (B, T, C, H, W)
        if depth_seq.dim() == 3:
            depth_seq = depth_seq.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
        elif depth_seq.dim() == 4:
            if depth_seq.shape[-2:] != (self.input_height, self.input_width):
                raise ValueError(f"Unexpected spatial shape: {tuple(depth_seq.shape)}")

            # (B, T, H, W) -> (B, T, 1, H, W)
            if depth_seq.shape[-3] == self.seq_len:
                depth_seq = depth_seq.unsqueeze(2)
            # (T, C, H, W) -> (1, T, C, H, W)
            else:
                depth_seq = depth_seq.unsqueeze(0)
        elif depth_seq.dim() != 5:
            raise ValueError(f"Expected depth with 3/4/5 dims, got {tuple(depth_seq.shape)}")

        if depth_seq.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {depth_seq.shape[1]}")
        if depth_seq.shape[2] != self.input_channels:
            raise ValueError(f"Expected channels={self.input_channels}, got {depth_seq.shape[2]}")
        return depth_seq

    def forward(self, depth_seq, action=None):
        del action
        depth_seq = self._normalize_depth(depth_seq)

        # Vim returns a single global CLS feature: (B, D).
        cls_feature = self.vim(depth_seq, return_features=True)
        if cls_feature.dim() != 2:
            raise ValueError(f"Expected CLS feature shape (B, D), got {tuple(cls_feature.shape)}")

        # Feed CLS to TemporalMamba as a length-1 sequence.
        temporal_tokens = self.temporal_mamba(cls_feature.unsqueeze(1))  # (B, 1, D)
        if self.flatten_all_tokens:
            return temporal_tokens.reshape(temporal_tokens.shape[0], -1)  # still (B, D)
        return temporal_tokens[:, 0, :]


class Encoder(STVimVideoPatchEncoder):
    """
    Compatibility alias for agent import.
    """
