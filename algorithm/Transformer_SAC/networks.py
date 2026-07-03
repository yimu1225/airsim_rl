import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd

from ..cnn_modules import CNN
from ..config_loader import get_algo_param


class TransformerTemporalEncoder(nn.Module):
    """Transformer encoder over a sequence of per-frame visual tokens."""

    def __init__(
        self,
        dim,
        seq_len,
        n_layers=2,
        n_heads=4,
        dim_feedforward=None,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        use_cls_token=False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.use_cls_token = bool(use_cls_token)
        self.token_count = self.seq_len + int(self.use_cls_token)

        if dim % n_heads != 0:
            raise ValueError(f"Transformer embedding dim {dim} must be divisible by n_heads={n_heads}.")

        if dim_feedforward is None:
            dim_feedforward = dim * 4

        self.cls_token = None
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_count, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(n_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(n_layers)),
            norm=nn.LayerNorm(dim),
        )
        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        return self.encoder(x)


class TransformerEncoder(nn.Module):
    """
    Transformer-SAC visual encoder:
    CNN extracts per-frame spatial features, Transformer models temporal context.
    """

    def __init__(self, args):
        super().__init__()
        depth_shape = args.depth_shape
        height = depth_shape[-2]
        width = depth_shape[-1]

        self.seq_len = int(getattr(args, "n_frames", 4))
        self.spatial_encoder = CNN(
            input_height=height,
            input_width=width,
            input_channels=1,
        )
        self.embed_dim = self.spatial_encoder.repr_dim

        self.use_cls_token = bool(get_algo_param(args, "transformer_sac_use_cls_token", False))
        self.flatten_all_tokens = bool(get_algo_param(args, "transformer_sac_flatten_all_tokens", True))
        self.temporal_transformer = TransformerTemporalEncoder(
            dim=self.embed_dim,
            seq_len=self.seq_len,
            n_layers=get_algo_param(args, "transformer_sac_temporal_depth", 2),
            n_heads=get_algo_param(args, "transformer_sac_n_heads", 4),
            dim_feedforward=get_algo_param(args, "transformer_sac_ff_dim", self.embed_dim * 4),
            dropout=get_algo_param(args, "transformer_sac_dropout", 0.0),
            activation=get_algo_param(args, "transformer_sac_activation", "gelu"),
            norm_first=get_algo_param(args, "transformer_sac_norm_first", True),
            use_cls_token=self.use_cls_token,
        )

        if self.use_cls_token:
            self.repr_dim = self.embed_dim if not self.flatten_all_tokens else self.embed_dim * (self.seq_len + 1)
        else:
            self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim

    def _format_depth_sequence(self, depth_seq):
        if depth_seq.dim() == 2:
            depth_seq = depth_seq.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif depth_seq.dim() == 3:
            depth_seq = depth_seq.unsqueeze(0).unsqueeze(2)
        elif depth_seq.dim() == 4:
            if depth_seq.size(0) == self.seq_len and depth_seq.size(1) == 1:
                depth_seq = depth_seq.unsqueeze(0)
            else:
                depth_seq = depth_seq.unsqueeze(2)
        elif depth_seq.dim() != 5:
            raise ValueError(f"Unsupported depth_seq shape: {tuple(depth_seq.shape)}")

        if depth_seq.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {depth_seq.size(1)}")
        if depth_seq.size(2) != 1:
            raise ValueError(f"Expected single-channel depth frames, got {tuple(depth_seq.shape)}")
        return depth_seq

    def forward(self, depth_seq):
        depth_seq = self._format_depth_sequence(depth_seq)
        batch_size, seq_len, channels, height, width = depth_seq.shape

        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        frame_tokens = self.spatial_encoder(frames).view(batch_size, seq_len, self.embed_dim)
        temporal_tokens = self.temporal_transformer(frame_tokens)

        if self.flatten_all_tokens:
            return temporal_tokens.reshape(batch_size, temporal_tokens.size(1) * self.embed_dim)
        if self.use_cls_token:
            return temporal_tokens[:, 0, :]
        return temporal_tokens[:, -1, :]


class Actor(nn.Module):
    """SAC actor with a tanh-squashed Gaussian policy."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        self.action_dim = action_shape[0]
        self.input_norm = nn.LayerNorm(repr_dim)
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        ) 
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="linear")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, with_log_prob=False):
        obs = self.input_norm(obs)
        latent = self.trunk(obs)
        mean = self.mean_linear(latent)
        log_std = self.log_std_linear(latent).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

        if not compute_pi:
            return mean, log_std

        std = log_std.exp()
        dist = pyd.Normal(mean, std)
        gaussian_action = dist.rsample()
        action = torch.tanh(gaussian_action)

        if compute_log_pi:
            log_prob = dist.log_prob(gaussian_action).sum(dim=-1, keepdim=True)
            log_prob -= (2 * (math.log(2) - gaussian_action - F.softplus(-2 * gaussian_action))).sum(
                dim=-1,
                keepdim=True,
            )
            if with_log_prob:
                return action, log_prob, mean, log_std
            return action, log_prob, mean, log_std
        return action, mean, log_std

    def get_action(self, obs, deterministic=False):
        mean, log_std = self.forward(obs, compute_pi=False)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = pyd.Normal(mean, std)
        gaussian_action = dist.rsample()
        return torch.tanh(gaussian_action)


class Critic(nn.Module):
    """Twin Q-network critic for SAC."""

    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        input_dim = repr_dim + action_shape[0]
        self.input_norm = nn.LayerNorm(input_dim)
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="linear")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, obs, action):
        q_input = torch.cat([obs, action], dim=-1)
        q_input = self.input_norm(q_input)
        return self.q1(q_input), self.q2(q_input)
