import torch
import torch.nn as nn
from torch.distributions import Normal

from Vim.vim.models_mamba import VisionMamba
from mamba_ssm import Mamba
from ..config_loader import get_algo_param


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(self.norm(x))


class TemporalMambaStack(nn.Module):
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
    """Frame-wise VisionMamba followed by temporal Mamba."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = get_algo_param(args, "st_mamba_embed_dim")
        self.depth = get_algo_param(args, "st_mamba_depth")
        self.patch_size = get_algo_param(args, "st_mamba_patch_size")
        self.d_state = get_algo_param(args, "st_mamba_d_state")
        self.d_conv = get_algo_param(args, "st_mamba_d_conv")
        self.expand = get_algo_param(args, "st_mamba_expand")
        self.temporal_layers = get_algo_param(args, "st_mamba_temporal_depth", 2)
        self.seq_len = args.n_frames

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        height = depth_shape[1]
        width = depth_shape[2]

        self.concat_cls_before_temporal_mamba = bool(
            get_algo_param(args, "st_vim_concat_cls_before_temporal_mamba", False)
        )
        self.flatten_all_tokens = bool(get_algo_param(args, "st_vim_flatten_all_tokens", True))
        if self.concat_cls_before_temporal_mamba:
            self.repr_dim = (self.embed_dim * self.seq_len) if self.flatten_all_tokens else 1
        else:
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

        temporal_dim = 1 if self.concat_cls_before_temporal_mamba else self.embed_dim
        self.temporal_mamba = TemporalMambaStack(
            dim=temporal_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

    def forward(self, depth_seq):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)

        batch_size, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        frame_tokens = self.vim(frames, return_features=True)
        frame_tokens = frame_tokens.view(batch_size, seq_len, self.embed_dim)

        if self.concat_cls_before_temporal_mamba:
            temporal_input = frame_tokens.reshape(batch_size, seq_len * self.embed_dim, 1)
            temporal_tokens = self.temporal_mamba(temporal_input)
            if self.flatten_all_tokens:
                return temporal_tokens.reshape(batch_size, seq_len * self.embed_dim)
            return temporal_tokens[:, -1, :]

        temporal_tokens = self.temporal_mamba(frame_tokens)
        if self.flatten_all_tokens:
            return temporal_tokens.reshape(batch_size, seq_len * self.embed_dim)
        return temporal_tokens[:, -1, :]


class Actor(nn.Module):
    """Unsquashed diagonal Gaussian actor, matching SB3 PPO for Box actions."""

    def __init__(self, repr_dim, action_dim, hidden_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(repr_dim)
        self.mean_net = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def distribution(self, obs):
        obs = self.input_norm(obs)
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs, deterministic=False):
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_log_prob(self, obs, action):
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class Critic(nn.Module):
    """State value function."""

    def __init__(self, repr_dim, hidden_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(repr_dim)
        self.value_net = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
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

    def forward(self, obs):
        obs = self.input_norm(obs)
        return self.value_net(obs)
