import torch
import torch.nn as nn

from Vim.vim.models_mamba import VisionMamba
from mamba_ssm import Mamba
from ..config_loader import get_algo_param




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
        return self.mamba(self.norm(x))


class TemporalMambaStack(nn.Module):
    """
    1D Mamba stack for temporal processing.
    Input/Output: (Batch, SeqLen, Dim)
    """
    def __init__(self, dim, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
      
        self.mamba_layers = nn.ModuleList([
            MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        for layer in self.mamba_layers:
            x = layer(x)
        return x


class STVimEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = get_algo_param(args, "st_mamba_embed_dim")
        self.depth = get_algo_param(args, "st_mamba_depth")
        self.patch_size = get_algo_param(args, "st_mamba_patch_size")
        self.d_state = get_algo_param(args, "st_mamba_d_state")
        self.d_conv = get_algo_param(args, "st_mamba_d_conv")
        self.expand = get_algo_param(args, "st_mamba_expand")
        self.temporal_layers = get_algo_param(args, "st_mamba_temporal_depth", 2)

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        self.seq_len = args.n_frames
        # Default to flatten all temporal tokens for downstream policy/value heads.
        # Set args.st_vim_flatten_all_tokens = False to recover old behavior (last token only).
        self.flatten_all_tokens = bool(get_algo_param(args, "st_vim_flatten_all_tokens", True))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim

        height = depth_shape[1]
        width = depth_shape[2]

        self.vim = VisionMamba(
            img_size=(height, width),
            patch_size=self.patch_size,
            stride=self.patch_size,
            depth=self.depth,
            embed_dim=self.embed_dim,
            d_state=self.d_state,
            channels=in_chans,
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
            drop_rate=get_algo_param(args, "st_mamba_drop_rate"),
            drop_path_rate=get_algo_param(args, "st_mamba_drop_path_rate"),
        )

        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )

    def forward(self, depth_seq, action=None):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)

        B, T, C, H, W = depth_seq.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")

        frames = depth_seq.reshape(B * T, C, H, W)
        frame_tokens = self.vim(frames, return_features=True)
        frame_tokens = frame_tokens.view(B, T, self.embed_dim)

        temporal_tokens = self.temporal_mamba(frame_tokens)
        if self.flatten_all_tokens:
            # Use all temporal tokens: (B, T, D) -> (B, T*D)
            vis_tokens = temporal_tokens.reshape(B, T * self.embed_dim)
        else:
            # Old behavior: use only the last temporal token.
            vis_tokens = temporal_tokens[:, -1, :]

        return vis_tokens


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
