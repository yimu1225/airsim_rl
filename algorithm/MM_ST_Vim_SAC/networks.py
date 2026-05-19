import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

from Vim.vim.models_mamba import VisionMamba
from mamba_ssm import Mamba
from ..config_loader import get_algo_param


LOG_STD_MIN = -20
LOG_STD_MAX = 2


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
    """Per-frame visual/base fusion followed by temporal Mamba."""

    def __init__(self, args):
        super().__init__()
        legacy_embed_dim = get_algo_param(args, "st_mamba_embed_dim")
        self.vim_embed_dim = int(get_algo_param(args, "st_vim_embed_dim", legacy_embed_dim))
        self.depth = get_algo_param(args, "st_mamba_depth")
        self.patch_size = get_algo_param(args, "st_mamba_patch_size")
        self.d_state = get_algo_param(args, "st_mamba_d_state")
        self.d_conv = get_algo_param(args, "st_mamba_d_conv")
        self.expand = get_algo_param(args, "st_mamba_expand")
        self.temporal_layers = get_algo_param(args, "st_mamba_temporal_depth", 2)
        self.seq_len = args.n_frames
        self.base_dim = int(getattr(args, "base_dim", 0))
        if self.base_dim <= 0:
            raise ValueError("MM_ST_Vim_SAC requires args.base_dim > 0")
        self.base_proj_dim = int(get_algo_param(args, "st_base_proj_dim", self.vim_embed_dim))
        self.temporal_dim = self.vim_embed_dim + self.base_proj_dim
        configured_temporal_dim = get_algo_param(args, "st_temporal_mamba_dim", self.temporal_dim)
        if int(configured_temporal_dim) != self.temporal_dim:
            raise ValueError(
                "MM_ST_Vim_SAC concatenates visual and base tokens directly, so "
                f"st_temporal_mamba_dim must equal st_vim_embed_dim + st_base_proj_dim "
                f"({self.vim_embed_dim} + {self.base_proj_dim} = {self.temporal_dim}), "
                f"got {configured_temporal_dim}"
            )

        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        height = depth_shape[1]
        width = depth_shape[2]

        self.flatten_all_tokens = bool(get_algo_param(args, "st_vim_flatten_all_tokens", True))
        self.repr_dim = self.temporal_dim * self.seq_len if self.flatten_all_tokens else self.temporal_dim

        self.vim = VisionMamba(
            img_size=(height, width),
            patch_size=self.patch_size,
            stride=self.patch_size,
            depth=self.depth,
            embed_dim=self.vim_embed_dim,
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
            nn.Linear(self.base_dim, self.base_proj_dim),
            nn.GELU(),
            nn.Linear(self.base_proj_dim, self.base_proj_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.temporal_dim),
            nn.Linear(self.temporal_dim, self.temporal_dim * 2),
            nn.GELU(),
            nn.Linear(self.temporal_dim * 2, self.temporal_dim),
        )
        self.temporal_mamba = TemporalMambaStack(
            dim=self.temporal_dim,
            n_layers=self.temporal_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

    def _prepare_base_sequence(self, base_seq, batch_size, seq_len):
        if base_seq.dim() == 2:
            expected = (seq_len, self.base_dim)
            if batch_size != 1 or tuple(base_seq.shape) != expected:
                raise ValueError(
                    f"Expected base_seq shape {expected} for a single sample, got {tuple(base_seq.shape)}"
                )
            base_seq = base_seq.unsqueeze(0)
        elif base_seq.dim() == 3:
            expected = (batch_size, seq_len, self.base_dim)
            if tuple(base_seq.shape) != expected:
                raise ValueError(f"Expected base_seq shape {expected}, got {tuple(base_seq.shape)}")
        else:
            raise ValueError(f"Expected base_seq rank 2 or 3, got shape={tuple(base_seq.shape)}")
        return base_seq

    def forward(self, depth_seq, base_seq):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)
        if depth_seq.dim() != 5:
            raise ValueError(f"Expected depth_seq shape (B,T,C,H,W), got {tuple(depth_seq.shape)}")

        batch_size, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        base_seq = self._prepare_base_sequence(base_seq, batch_size, seq_len)

        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        frame_tokens = self.vim(frames, return_features=True)
        frame_tokens = frame_tokens.view(batch_size, seq_len, self.vim_embed_dim)

        base_tokens = self.base_proj(base_seq.reshape(batch_size * seq_len, self.base_dim))
        base_tokens = base_tokens.view(batch_size, seq_len, self.base_proj_dim)
        fused_tokens = torch.cat([frame_tokens, base_tokens], dim=-1)
        fused_tokens = self.fusion(fused_tokens)

        temporal_tokens = self.temporal_mamba(fused_tokens)
        if self.flatten_all_tokens:
            return temporal_tokens.reshape(batch_size, seq_len * self.temporal_dim)
        return temporal_tokens[:, -1, :]


class Actor(nn.Module):
    """SAC squashed Gaussian actor."""

    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        self.action_dim = action_shape[0]
        self.input_norm = nn.LayerNorm(repr_dim)
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def distribution_params(self, obs):
        obs = self.input_norm(obs)
        latent = self.trunk(obs)
        mean = self.mean_linear(latent)
        log_std = self.log_std_linear(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def action_log_prob(self, obs):
        mean, log_std = self.distribution_params(obs)
        std = log_std.exp()
        dist = pyd.Normal(mean, std)
        gaussian_action = dist.rsample()
        action = torch.tanh(gaussian_action)
        log_prob = dist.log_prob(gaussian_action).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - gaussian_action - F.softplus(-2 * gaussian_action))).sum(
            dim=-1,
            keepdim=True,
        )
        return action, log_prob

    def forward(self, obs, deterministic=False):
        mean, log_std = self.distribution_params(obs)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = pyd.Normal(mean, std)
        return torch.tanh(dist.sample())


class Critic(nn.Module):
    """Twin Q-network critic."""

    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        input_dim = repr_dim + action_shape[0]
        self.input_norm = nn.LayerNorm(input_dim)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

    def forward(self, obs, action):
        q_input = torch.cat([obs, action], dim=-1)
        q_input = self.input_norm(q_input)
        return self.q1(q_input), self.q2(q_input)
