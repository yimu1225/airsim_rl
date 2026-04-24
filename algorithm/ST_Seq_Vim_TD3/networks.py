import torch
import torch.nn as nn

from Vim.vim.models_mamba import VisionMamba
from mamba_ssm import Mamba
from ..config_loader import get_algo_param


class MambaBlock(nn.Module):
    """Temporal Mamba block with pre-normalization."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(self.norm(x))


class TemporalMambaStack(nn.Module):
    """Stacked temporal Mamba layers for sequence modeling."""

    def __init__(self, dim, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_layers = nn.ModuleList(
            [MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
        )

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

        self.base_dim = int(getattr(args, "base_dim", 0))
        if self.base_dim <= 0:
            raise ValueError("STVimEncoder requires args.base_dim > 0 for base-state sequence fusion")
        self.state_proj_dim = int(get_algo_param(args, "st_state_proj_dim", self.embed_dim))

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        self.seq_len = args.n_frames
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
            final_pool_type="none",
            if_bimamba=True,
            bimamba_type="v2",
            drop_rate=get_algo_param(args, "st_mamba_drop_rate"),
            drop_path_rate=get_algo_param(args, "st_mamba_drop_path_rate"),
        )

        self.base_proj = nn.Sequential(
            nn.LayerNorm(self.base_dim),
            nn.Linear(self.base_dim, self.state_proj_dim),
            nn.GELU(),
            nn.Linear(self.state_proj_dim, self.embed_dim),
        )
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )

        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

    def _prepare_base_seq(self, base_seq, batch_size, seq_len):
        if base_seq is None:
            raise ValueError("base_seq cannot be None for STVimEncoder")

        if base_seq.dim() == 1:
            if base_seq.shape[0] != self.base_dim:
                raise ValueError(f"Expected base dim {self.base_dim}, got {base_seq.shape[0]}")
            base_seq = base_seq.view(1, 1, self.base_dim).expand(batch_size, seq_len, self.base_dim)
        elif base_seq.dim() == 2:
            if base_seq.shape == (seq_len, self.base_dim):
                if batch_size != 1:
                    raise ValueError(
                        f"Ambiguous base_seq shape {tuple(base_seq.shape)} for batch_size={batch_size}; expected (B, base_dim)."
                    )
                base_seq = base_seq.unsqueeze(0)
            elif base_seq.shape == (batch_size, self.base_dim):
                base_seq = base_seq.unsqueeze(1).expand(batch_size, seq_len, self.base_dim)
            else:
                raise ValueError(
                    f"Invalid base_seq shape {tuple(base_seq.shape)}. "
                    f"Expected (T,{self.base_dim}) or (B,{self.base_dim}) with T={seq_len}, B={batch_size}."
                )
        elif base_seq.dim() == 3:
            if base_seq.shape[0] != batch_size or base_seq.shape[1] != seq_len or base_seq.shape[2] != self.base_dim:
                raise ValueError(
                    f"Invalid base_seq shape {tuple(base_seq.shape)}. "
                    f"Expected (B,T,{self.base_dim}) with B={batch_size}, T={seq_len}."
                )
        else:
            raise ValueError(f"Unsupported base_seq dim={base_seq.dim()}, expected 1/2/3")

        return base_seq

    def forward(self, depth_seq, base_seq):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)

        if depth_seq.dim() != 5:
            raise ValueError(f"Expected depth_seq rank 5 (B,T,C,H,W), got shape={tuple(depth_seq.shape)}")

        batch_size, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        base_seq = self._prepare_base_seq(base_seq, batch_size, seq_len)

        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        frame_tokens = self.vim(frames, return_features=True)
        frame_tokens = frame_tokens.view(batch_size, seq_len, self.embed_dim)

        base_tokens = self.base_proj(base_seq.reshape(batch_size * seq_len, self.base_dim))
        base_tokens = base_tokens.view(batch_size, seq_len, self.embed_dim)

        fused_tokens = torch.cat([frame_tokens, base_tokens], dim=-1)
        fused_tokens = self.fuse_proj(fused_tokens)

        temporal_tokens = self.temporal_mamba(fused_tokens)
        if self.flatten_all_tokens:
            return temporal_tokens.reshape(batch_size, seq_len * self.embed_dim)
        return temporal_tokens[:, -1, :]


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
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, features, action):
        xu = torch.cat([features, action], dim=-1)
        xu = self.input_norm(xu)
        return self.net(xu)
