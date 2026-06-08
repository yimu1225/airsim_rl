import torch
import torch.nn as nn

from Vim.vim.models_mamba import VisionMamba
from ..config_loader import get_algo_param


class VimEncoder(nn.Module):
    """
    Vim-only encoder:
    - Per-frame VisionMamba feature extraction
    - No temporal Mamba stack
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = get_algo_param(args, "st_mamba_embed_dim")
        self.depth = get_algo_param(args, "st_mamba_depth")
        self.patch_size = get_algo_param(args, "st_mamba_patch_size")
        self.d_state = get_algo_param(args, "st_mamba_d_state")
        self.seq_len = args.n_frames

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        height = depth_shape[1]
        width = depth_shape[2]

        # Keep token usage behavior aligned with ST-VimTD3 for fair ablation.
        self.flatten_all_tokens = bool(get_algo_param(args, "vmflatten_all_tokens", True))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim

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

    def forward(self, depth_seq):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)

        bsz, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        frames = depth_seq.reshape(bsz * seq_len, channels, height, width)
        frame_tokens = self.vim(frames, return_features=True).view(bsz, seq_len, self.embed_dim)

        if self.flatten_all_tokens:
            return frame_tokens.reshape(bsz, seq_len * self.embed_dim)
        return frame_tokens[:, -1, :]


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
