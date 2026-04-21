import torch
import torch.nn as nn

from mamba_ssm import Mamba

from ..cnn_modules import CNN


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
        self.layers = nn.ModuleList(
            [MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(max(1, int(n_layers)))]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MambaEncoder(nn.Module):
    """
    Mamba-only encoder:
    - CNN for per-frame spatial features
    - Temporal Mamba stack for sequence modeling
    """

    def __init__(self, args):
        super().__init__()
        depth_shape = args.depth_shape
        in_chans = depth_shape[0]
        height = depth_shape[1]
        width = depth_shape[2]

        self.seq_len = args.n_frames
        self.spatial_encoder = CNN(
            input_height=height,
            input_width=width,
            input_channels=in_chans,
        )
        self.embed_dim = self.spatial_encoder.repr_dim

        self.temporal_layers = int(getattr(args, "mamba_td3_temporal_depth", 2))
        self.temporal_mamba = TemporalMambaStack(
            dim=self.embed_dim,
            n_layers=self.temporal_layers,
            d_state=getattr(args, "mamba_d_state", 16),
            d_conv=getattr(args, "mamba_d_conv", 4),
            expand=getattr(args, "mamba_expand", 2),
        )

        self.flatten_all_tokens = bool(getattr(args, "mamba_td3_flatten_all_tokens", True))
        self.repr_dim = self.embed_dim * self.seq_len if self.flatten_all_tokens else self.embed_dim

    def forward(self, depth_seq):
        if depth_seq.dim() == 4:
            depth_seq = depth_seq.unsqueeze(0)

        bsz, seq_len, channels, height, width = depth_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        frames = depth_seq.reshape(bsz * seq_len, channels, height, width)
        frame_tokens = self.spatial_encoder(frames).view(bsz, seq_len, self.embed_dim)
        temporal_tokens = self.temporal_mamba(frame_tokens)

        if self.flatten_all_tokens:
            return temporal_tokens.reshape(bsz, seq_len * self.embed_dim)
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

    def forward(self, features):
        return self.net(self.input_norm(features))


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
        x = torch.cat([features, action], dim=-1)
        return self.net(self.input_norm(x))
