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
    omega -> FC mean/std -> z -> four transposed convolutions with kernels
    5x7, 7x11, 9x9, 7x7.
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

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 7), stride=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=(7, 11), stride=4)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=(9, 9), stride=5)
        self.deconv4 = nn.ConvTranspose2d(
            8,
            1,
            kernel_size=(7, 7),
            stride=4,
            output_padding=(1, 1),
        )

        self.apply(self._init_weights)

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
        if reconstruction.shape[-2:] != (self.input_height, self.input_width):
            reconstruction = F.interpolate(
                reconstruction,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False,
            )
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
