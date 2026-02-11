import torch
import torch.nn as nn

from Vim.vim.models_mamba import VisionMamba
from mamba_ssm import Mamba


MAMBA_AVAILABLE = True


class TemporalMambaStack(nn.Module):
    """
    1D Mamba stack for temporal processing.
    Input/Output: (Batch, SeqLen, Dim)
    """
    def __init__(self, dim, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
        else:
            self.gru = nn.GRU(dim, dim, batch_first=True, num_layers=n_layers)

    def forward(self, x):
        if MAMBA_AVAILABLE:
            for layer in self.mamba_layers:
                x = layer(x)
            return x
        out, _ = self.gru(x)
        return out


class STVimTokenMambaEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super().__init__()
        self.embed_dim = args.st_mamba_embed_dim
        self.depth = args.st_mamba_depth
        self.patch_size = args.st_mamba_patch_size
        self.d_state = args.st_mamba_d_state
        self.d_conv = args.st_mamba_d_conv
        self.expand = args.st_mamba_expand
        self.temporal_layers = getattr(args, "st_mamba_temporal_depth", 2)

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        self.seq_len = args.seq_len

        height = depth_shape[1]
        width = depth_shape[2]

        cmd_in_dim = state_dim + (action_dim if action_dim is not None else 0)
        self.cmd_mlp = nn.Sequential(
            nn.Linear(cmd_in_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

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
            drop_rate=args.st_mamba_drop_rate,
            drop_path_rate=args.st_mamba_drop_path_rate,
        )

        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )

    def forward(self, depth_seq, state_vec, action=None):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)
        if state_vec.dim() == 1:
            state_vec = state_vec.unsqueeze(0)
        if action is not None and action.dim() == 1:
            action = action.unsqueeze(0)

        B, T, C, H, W = depth_seq.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")

        frames = depth_seq.reshape(B * T, C, H, W)
        frame_tokens = self.vim(frames, return_features=True)
        frame_tokens = frame_tokens.view(B, T, self.embed_dim)

        temporal_tokens = self.temporal_mamba(frame_tokens)
        vis_token = temporal_tokens[:, -1, :]

        if action is not None:
            cmd_input = torch.cat([state_vec, action], dim=-1)
        else:
            cmd_input = state_vec
        cmd_token = self.cmd_mlp(cmd_input)

        return vis_token + cmd_token


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.q1_net(x), self.q2_net(x)

    def q1(self, x):
        return self.q1_net(x)
