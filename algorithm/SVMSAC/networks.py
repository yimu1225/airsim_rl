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
            get_algo_param(args, "vmconcat_cls_before_temporal_mamba", False)
        )
        self.flatten_all_tokens = bool(get_algo_param(args, "vmflatten_all_tokens", True))
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


# ================================================================
#  VisualSubNetwork - 视觉分支
#  深度序列 → STVimEncoder → so_repr (视觉特征)
#  MLP(so_repr) → Tanh → ao_visual (视觉子动作)
# ================================================================

class VisualSubNetwork(nn.Module):
    """Visual subnetwork: depth_seq → STVimEncoder → (ao_visual, so_repr)."""
    def __init__(self, args, hidden_dims, out_dim):
        super().__init__()
        self.encoder = STVimEncoder(args)
        self.repr_dim = self.encoder.repr_dim

        h1, h2 = hidden_dims
        self.input_norm = nn.LayerNorm(self.repr_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.repr_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, out_dim),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, depth_seq):
        """Returns: ao_visual (B, out_dim), so_repr (B, repr_dim)."""
        so_repr = self.encoder(depth_seq)
        ao_visual = self.mlp(self.input_norm(so_repr))
        return ao_visual, so_repr


# ================================================================
#  BaseSubNetwork - 基础状态分支
#  基础状态 (11维) → MLP → Tanh → ao_base (基础子动作)
# ================================================================

class BaseSubNetwork(nn.Module):
    """Base state subnetwork: full base_state → MLP → ao_base."""
    def __init__(self, base_dim, hidden_dims, out_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(base_dim)
        h1, h2 = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, out_dim),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, base_state):
        """Returns: ao_base (B, out_dim)."""
        return self.mlp(self.input_norm(base_state))


# ================================================================
#  GlobalActor - 融合 ao_visual, ao_base 与完整状态表征
#  输入: cat[ao_visual, ao_base, so_repr, base_state]
#  输出: action, log_prob (SAC squashed Gaussian)
# ================================================================

class GlobalActor(nn.Module):
    """Global Actor: fuses sub-actions + full state → SAC Gaussian action."""

    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

    def distribution_params(self, ao_visual, ao_base, so_repr, base_state):
        x = torch.cat([ao_visual, ao_base, so_repr, base_state], dim=-1)
        x = self.input_norm(x)
        latent = self.trunk(x)
        mean = self.mean_linear(latent)
        log_std = self.log_std_linear(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def action_log_prob(self, ao_visual, ao_base, so_repr, base_state):
        mean, log_std = self.distribution_params(ao_visual, ao_base, so_repr, base_state)
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

    def forward(self, ao_visual, ao_base, so_repr, base_state, deterministic=False):
        mean, log_std = self.distribution_params(ao_visual, ao_base, so_repr, base_state)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = pyd.Normal(mean, std)
        return torch.tanh(dist.sample())


# ================================================================
#  Critic - 独立的视觉编码器 + 双Q网络
#  输入: depth_seq, base_state, action
#  独立 STVimEncoder(depth) → so_repr
#  cat[so_repr, base_state, action] → Q1, Q2
# ================================================================

class Critic(nn.Module):
    """Twin Q-network critic with independent STVimEncoder."""

    def __init__(self, args, base_dim, action_dim):
        super().__init__()
        self.encoder = STVimEncoder(args)
        self.repr_dim = self.encoder.repr_dim

        input_dim = self.repr_dim + base_dim + action_dim
        self.input_norm = nn.LayerNorm(input_dim)
        hidden_dim = args.hidden_dim
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

    def forward(self, depth_seq, base_state, action):
        so_repr = self.encoder(depth_seq)
        x = torch.cat([so_repr, base_state, action], dim=-1)
        x = self.input_norm(x)
        return self.q1(x), self.q2(x)
