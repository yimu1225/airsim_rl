import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from ..cnn_modules import CNN


class TruncatedNormal(pyd.Normal):
    """Truncated Normal distribution for SAC action sampling."""
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x
 
    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Encoder(CNN):
    """
    Encoder that uses the unified CNN from cnn_modules.
    """
    def __init__(self, input_height, input_width, input_channels=1):
        super().__init__(input_height, input_width, input_channels)


class Actor(nn.Module):
    """SAC Actor with stochastic policy (Gaussian)."""
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        
        self.action_dim = action_shape[0]
        
        # LayerNorm on input representation
        self.input_norm = nn.LayerNorm(repr_dim)
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Mean and log_std heads
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, with_log_prob=False):
        """
        Args:
            obs: encoded state representation
            compute_pi: whether to sample action
            compute_log_pi: whether to compute log probability
            with_log_prob: if True, return log_prob along with action
        Returns:
            action (detached if not with_log_prob), mean, log_std
        """
        obs = self.input_norm(obs)
        h = self.trunk(obs)
        
        mean = self.mean_linear(h)
        log_std = self.log_std_linear(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        if not compute_pi:
            return mean, log_std
        
        # Reparameterization trick
        std = log_std.exp()
        dist = pyd.Normal(mean, std)
        
        # Sample action
        z = dist.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(z)
        
        if compute_log_pi:
            # Compute log probability with tanh correction
            log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
            # Tanh correction: log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
            log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(dim=-1, keepdim=True)
        else:
            log_prob = None
            
        if with_log_prob:
            return action, log_prob, mean, log_std
        return action, mean, log_std

    def get_action(self, obs, deterministic=False):
        """Get action for inference (no gradient)."""
        with torch.no_grad():
            mean, log_std = self.forward(obs, compute_pi=False)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                dist = pyd.Normal(mean, std)
                z = dist.sample()
                action = torch.tanh(z)
        return action


class Critic(nn.Module):
    """Twin Q-networks for SAC."""
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()
        
        input_dim = repr_dim + action_shape[0]
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Q1 network
        self.Q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.Q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, action):
        """Return Q-values from both networks."""
        h_action = torch.cat([obs, action], dim=-1)
        h_action = self.input_norm(h_action)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2
    
    def forward_q1(self, obs, action):
        """Return only Q1 value (for policy update efficiency)."""
        h_action = torch.cat([obs, action], dim=-1)
        h_action = self.input_norm(h_action)
        return self.Q1(h_action)
