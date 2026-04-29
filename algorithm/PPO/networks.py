import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from ..cnn_modules import CNN


class Encoder(CNN):
    """
    Encoder that uses the unified CNN from cnn_modules.
    Same as TD3 for fair comparison.
    """
    def __init__(self, input_height, input_width, input_channels=1):
        super().__init__(input_height, input_width, input_channels)


class Actor(nn.Module):
    """
    PPO Actor with Gaussian policy for continuous actions.
    Outputs mean and log_std for action distribution.
    """
    def __init__(self, repr_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(repr_dim)
        
        # Mean network
        self.mean_net = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Learnable log standard deviation (state-independent)
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass.
        
        Args:
            obs: encoded observation
            deterministic: if True, return mean action; otherwise sample
            
        Returns:
            action: sampled or deterministic action (normalized to [-1, 1])
            log_prob: log probability of the action
        """
        obs = self.input_norm(obs)
        mean = self.mean_net(obs)
        
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            # SB3 PPO uses an unsquashed diagonal Gaussian and clips the action
            # only when interacting with Box environments.
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_log_prob(self, obs, action):
        """
        Compute log probability of a given action.
        Used for PPO updates.
        
        Args:
            obs: encoded observation
            action: action (already normalized to [-1, 1])
            
        Returns:
            log_prob: log probability of the action
            entropy: entropy of the distribution
        """
        obs = self.input_norm(obs)
        mean = self.mean_net(obs)
        
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class Critic(nn.Module):
    """
    PPO Critic (Value Function).
    Estimates state value V(s).
    """
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
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs):
        """
        Forward pass.
        
        Args:
            obs: encoded observation
            
        Returns:
            value: state value estimate
        """
        obs = self.input_norm(obs)
        value = self.value_net(obs)
        return value
