import torch
import torch.nn as nn

from Vim.vim.models_mamba import VisionMamba
from ..config_loader import get_algo_param


class VimStateSeqEncoder(nn.Module):
    """
    Vim-only visual encoder + base-state sequence fusion.
    Difference from ST variant:
    - No extra temporal Mamba stack after fusion.
    - Keep Vim-only ablation spirit while still using base sequence per timestep.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = get_algo_param(args, "st_mamba_embed_dim")
        self.depth = get_algo_param(args, "st_mamba_depth")
        self.patch_size = get_algo_param(args, "st_mamba_patch_size")
        self.d_state = get_algo_param(args, "st_mamba_d_state")

        self.base_dim = int(getattr(args, "base_dim", 0))
        if self.base_dim <= 0:
            raise ValueError("VimStateSeqEncoder requires args.base_dim > 0")
        self.state_proj_dim = int(get_algo_param(args, "st_state_proj_dim", self.embed_dim))

        self.seq_len = int(args.n_frames)
        self.flatten_all_tokens = bool(get_algo_param(args, "st_vim_flatten_all_tokens", True))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
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

    def _prepare_base_seq(self, base_seq, batch_size):
        if base_seq is None:
            raise ValueError("base_seq cannot be None")

        if base_seq.dim() == 1:
            if base_seq.shape[0] != self.base_dim:
                raise ValueError(f"Expected base dim {self.base_dim}, got {base_seq.shape[0]}")
            return base_seq.view(1, 1, self.base_dim).expand(batch_size, self.seq_len, self.base_dim)

        if base_seq.dim() == 2:
            if base_seq.shape == (self.seq_len, self.base_dim):
                if batch_size != 1:
                    raise ValueError(
                        f"Ambiguous base shape {tuple(base_seq.shape)} for batch_size={batch_size}; expected (B, base_dim)."
                    )
                return base_seq.unsqueeze(0)
            if base_seq.shape == (batch_size, self.base_dim):
                return base_seq.unsqueeze(1).expand(batch_size, self.seq_len, self.base_dim)
            raise ValueError(
                f"Invalid base shape {tuple(base_seq.shape)}; expected (T,{self.base_dim}) or (B,{self.base_dim})."
            )

        if base_seq.dim() == 3:
            expected = (batch_size, self.seq_len, self.base_dim)
            if tuple(base_seq.shape) != expected:
                raise ValueError(f"Invalid base sequence shape {tuple(base_seq.shape)}; expected {expected}.")
            return base_seq

        raise ValueError(f"Unsupported base_seq dim={base_seq.dim()}")

    def forward(self, depth_seq, base_seq):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)

        if depth_seq.dim() != 5:
            raise ValueError(f"Expected depth_seq rank 5 (B,T,C,H,W), got shape={tuple(depth_seq.shape)}")

        batch_size, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        base_seq = self._prepare_base_seq(base_seq, batch_size)

        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        vis_tokens = self.vim(frames, return_features=True).view(batch_size, seq_len, self.embed_dim)

        base_tokens = self.base_proj(base_seq.reshape(batch_size * seq_len, self.base_dim))
        base_tokens = base_tokens.view(batch_size, seq_len, self.embed_dim)

        fused = self.fuse_proj(torch.cat([vis_tokens, base_tokens], dim=-1))

        if self.flatten_all_tokens:
            return fused.reshape(batch_size, seq_len * self.embed_dim)
        return fused[:, -1, :]


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
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

    def forward(self, features):
        return self.net(self.input_norm(features))


class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim + action_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
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
        x = torch.cat([features, action], dim=-1)
        return self.net(self.input_norm(x))
