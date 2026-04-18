import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cnn_modules import CNN


class Encoder(CNN):
    """Encoder that uses the unified CNN from cnn_modules."""

    def __init__(self, input_height, input_width, input_channels=1):
        super().__init__(input_height, input_width, input_channels=input_channels)


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet linear layer.

    Note:
        This layer does not auto-resample noise in ``forward``.
        Call ``reset_noise()`` explicitly from the agent to control when
        a sampled noisy policy is held fixed (per environment step / update).
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_init = float(sigma_init)

        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.register_buffer("weight_epsilon", torch.zeros(self.out_features, self.in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(self.out_features))
            self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
            self.register_buffer("bias_epsilon", torch.zeros(self.out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_sigma", None)
            self.register_buffer("bias_epsilon", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x = torch.randn(size, device=device, dtype=dtype)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device, self.weight_mu.dtype)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device, self.weight_mu.dtype)

        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))

        if self.bias_epsilon is not None:
            self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor, use_noise: bool = False) -> torch.Tensor:
        if use_noise:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = None
            if self.bias_mu is not None:
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim, noisy_sigma_init: float = 0.5):
        super().__init__()

        self.input_norm = nn.LayerNorm(repr_dim)

        self.fc1 = NoisyLinear(repr_dim, hidden_dim, sigma_init=noisy_sigma_init)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = NoisyLinear(hidden_dim, hidden_dim, sigma_init=noisy_sigma_init)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = NoisyLinear(hidden_dim, action_shape[0], sigma_init=noisy_sigma_init)

    def forward(self, obs, std=None, use_noise: bool = False):
        obs = self.input_norm(obs)

        x = self.fc1(obs, use_noise=use_noise)
        x = self.ln1(x)
        x = F.relu(x, inplace=True)

        x = self.fc2(x, use_noise=use_noise)
        x = self.ln2(x)
        x = F.relu(x, inplace=True)

        x = self.fc3(x, use_noise=use_noise)
        x = torch.tanh(x)
        return x

    def reset_noise(self) -> None:
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.input_norm = nn.LayerNorm(repr_dim + action_shape[0])

        self.Q1 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        h_action = self.input_norm(h_action)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2
