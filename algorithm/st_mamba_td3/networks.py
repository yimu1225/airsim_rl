import torch
import torch.nn as nn

from Vim.vim.models_mamba import VisionMamba


class STMambaEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, args):
        super().__init__()
        self.embed_dim = args.st_mamba_embed_dim
        self.depth = args.st_mamba_depth
        self.patch_size = args.st_mamba_patch_size
        self.d_state = args.st_mamba_d_state
        self.d_conv = args.st_mamba_d_conv
        self.expand = args.st_mamba_expand

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        self.seq_len = args.seq_len

        height = depth_shape[1]
        width = depth_shape[2]
        self.grid_h = height // self.patch_size
        self.grid_w = width // self.patch_size
        self.tokens_per_frame = self.grid_h * self.grid_w
        self.num_tokens = self.tokens_per_frame * self.seq_len

        img_size = (height * self.seq_len, width)
        self.vim = VisionMamba(
            img_size=img_size,
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

    def forward(self, depth_seq, state_vec, action=None):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)

        B, T, C, H, W = depth_seq.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")

        merged = depth_seq.reshape(B, C, T * H, W)
        vis_token = self.vim(merged, return_features=True)

        return vis_token


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
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
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim + action_dim)
        self.q1_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_net = nn.Sequential(
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
        return self.q1_net(xu), self.q2_net(xu)

    def q1(self, x, action):
        xu = torch.cat([x, action], dim=-1)
        xu = self.input_norm(xu)
        return self.q1_net(xu)
