import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class PaperFeatureExtractor(nn.Module):
    """
    Self-supervised attention image feature extractor from Liu Bokai et al.

    The encoder follows Fig. 3:
    depth -> 3x3 conv x3 -> global average E1 -> 1x1 conv 64->32 -> 1x1 conv
    32->64 sigmoid E3 -> residual E1 + E3 -> channel reweight -> global average
    image feature omega (64x1x1).

    The decoder follows the paper's VAE branch:
    omega -> FC mean/std -> z -> four transposed convolutions. For 480x720
    input this exactly matches the paper's spatial path:
    64x1x1 -> 32x5x7 -> 16x23x35 -> 8x119x179 -> 1x480x720.
    For other project resolutions, the same four deconvolution stages are
    parameterized so the final transposed convolution reconstructs that run's
    input resolution directly.
    """

    def __init__(self, input_height: int, input_width: int, feature_dim: int = 64):
        super().__init__()
        if int(feature_dim) != 64:
            raise ValueError("The paper feature extractor uses a fixed 64-channel image feature.")

        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.feature_dim = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attn_reduce = nn.Conv2d(64, 32, kernel_size=1)
        self.attn_expand = nn.Conv2d(32, 64, kernel_size=1)

        self.fc_mean = nn.Linear(64, 64)
        self.fc_log_std = nn.Linear(64, 64)

        h3, h4_pad, h4_out_pad = self._solve_deconv_axis(self.input_height, kernel=7, stride=4)
        w3, w4_pad, w4_out_pad = self._solve_deconv_axis(self.input_width, kernel=7, stride=4)
        h2, h3_pad, h3_out_pad = self._solve_deconv_axis(h3, kernel=9, stride=5)
        w2, w3_pad, w3_out_pad = self._solve_deconv_axis(w3, kernel=9, stride=5)
        h1, h2_pad, h2_out_pad = self._solve_deconv_axis(h2, kernel=7, stride=4)
        w1, w2_pad, w2_out_pad = self._solve_deconv_axis(w2, kernel=11, stride=4)

        self.decoder_shapes = {
            "z": (1, 1),
            "deconv1": (h1, w1),
            "deconv2": (h2, w2),
            "deconv3": (h3, w3),
            "deconv4": (self.input_height, self.input_width),
        }

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(h1, w1), stride=1)
        self.deconv2 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=(7, 11),
            stride=4,
            padding=(h2_pad, w2_pad),
            output_padding=(h2_out_pad, w2_out_pad),
        )
        self.deconv3 = nn.ConvTranspose2d(
            16,
            8,
            kernel_size=(9, 9),
            stride=5,
            padding=(h3_pad, w3_pad),
            output_padding=(h3_out_pad, w3_out_pad),
        )
        self.deconv4 = nn.ConvTranspose2d(
            8,
            1,
            kernel_size=(7, 7),
            stride=4,
            padding=(h4_pad, w4_pad),
            output_padding=(h4_out_pad, w4_out_pad),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _solve_deconv_axis(target_size: int, kernel: int, stride: int):
        """Find input size, padding, and output_padding for exact deconv output."""
        if target_size <= 0:
            raise ValueError(f"Decoder target size must be positive, got {target_size}")

        start = max(1, int(np.floor((target_size - kernel) / stride)) + 1)
        best = None
        for input_size in range(start, start + stride + kernel + 8):
            base = (input_size - 1) * stride + kernel
            for output_padding in range(stride):
                diff = base + output_padding - target_size
                if diff < 0 or diff % 2 != 0:
                    continue
                padding = diff // 2
                candidate = (padding, input_size, output_padding)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            raise ValueError(
                f"Cannot solve ConvTranspose axis for target={target_size}, "
                f"kernel={kernel}, stride={stride}"
            )
        padding, input_size, output_padding = best
        return input_size, padding, output_padding

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def preprocess_depth(depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float()
        if depth.numel() > 0 and float(depth.detach().max().item()) > 1.5:
            depth = depth / 255.0
        return depth.clamp(0.0, 1.0)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self.preprocess_depth(frames)
        en = F.relu(self.conv1(frames), inplace=True)
        en = F.relu(self.conv2(en), inplace=True)
        en = F.relu(self.conv3(en), inplace=True)

        e1 = self.avg_pool(en)
        e2 = F.relu(self.attn_reduce(e1), inplace=True)
        e3 = torch.sigmoid(self.attn_expand(e2))
        ew = e1 + e3
        weighted = en * ew
        omega = self.avg_pool(weighted).flatten(1)
        return omega

    def decode_feature(self, omega: torch.Tensor, sample: bool = True):
        mean = self.fc_mean(omega)
        log_std = self.fc_log_std(omega).clamp(LOG_STD_MIN, LOG_STD_MAX)
        if sample:
            noise = torch.randn_like(mean)
            z = torch.exp(log_std) * noise + mean
        else:
            z = mean

        x = z.view(z.size(0), 64, 1, 1)
        x = F.relu(self.deconv1(x), inplace=True)
        x = F.relu(self.deconv2(x), inplace=True)
        x = F.relu(self.deconv3(x), inplace=True)
        reconstruction = self.deconv4(x)
        return reconstruction, mean, log_std

    def reconstruction_loss(self, frames: torch.Tensor, kl_weight: float = 0.5):
        target = self.preprocess_depth(frames)
        omega = self.encode_frames(target)
        reconstruction, mean, log_std = self.decode_feature(omega, sample=True)
        recon_loss = F.mse_loss(reconstruction, target)
        variance = torch.exp(2.0 * log_std)
        kl_loss = 0.5 * torch.mean(mean.pow(2) + variance - 2.0 * log_std - 1.0)
        feature_loss = recon_loss + float(kl_weight) * kl_loss
        return feature_loss, recon_loss, kl_loss

    def forward(self, depth_seq: torch.Tensor) -> torch.Tensor:
        if depth_seq.dim() != 5:
            raise ValueError(f"Expected depth sequence (B,T,C,H,W), got {tuple(depth_seq.shape)}")
        batch_size, seq_len, channels, height, width = depth_seq.shape
        if channels != 1:
            raise ValueError(f"Expected single-channel depth frames, got C={channels}")
        frames = depth_seq.reshape(batch_size * seq_len, channels, height, width)
        omega = self.encode_frames(frames)
        return omega.view(batch_size, seq_len, self.feature_dim)


class LSTMSACActor(nn.Module):
    """Actor from Fig. 4: FC512 -> LSTM512 -> FC512 -> mean/std -> tanh action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.input_norm = nn.LayerNorm(state_dim)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def distribution_params(self, state_seq: torch.Tensor):
        x = self.input_norm(state_seq)
        x = F.relu(self.fc1(x), inplace=True)
        x, _ = self.lstm(x)
        x = F.relu(self.fc2(x[:, -1, :]), inplace=True)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def action_log_prob(self, state_seq: torch.Tensor):
        mean, log_std = self.distribution_params(state_seq)
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

    def forward(self, state_seq: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.distribution_params(state_seq)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = pyd.Normal(mean, std)
        return torch.tanh(dist.sample())


class LSTMSACCritic(nn.Module):
    """Critic from Fig. 4: concat(s_t,a_t) -> FC512 -> LSTM512 -> FC512 -> FC512 -> Q."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        input_dim = state_dim + action_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state_seq: torch.Tensor, action: torch.Tensor):
        action_seq = action.unsqueeze(1).expand(-1, state_seq.size(1), -1)
        x = torch.cat([state_seq, action_seq], dim=-1)
        x = self.input_norm(x)
        x = F.relu(self.fc1(x), inplace=True)
        x, _ = self.lstm(x)
        x = F.relu(self.fc2(x[:, -1, :]), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return self.q(x)
