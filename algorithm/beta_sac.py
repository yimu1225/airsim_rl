from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.optim import Adam

from .config_loader import get_algo_param
from .SAC.agent import SACAgent
from .PL_SAC.agent import PLSACAgent
from .VMSAC.agent import STVimSACAgent
from .DPER_VMSAC.agent import DPERSTVimSACAgent
from .PL_DPER_VMSAC.agent import PLDPERSTVimSACAgent


class BetaFeedForwardActor(nn.Module):
    """SAC actor that samples bounded actions from independent Beta distributions."""

    def __init__(self, repr_dim, action_shape, hidden_dim, concentration_offset=1.0, eps=1e-6):
        super().__init__()
        self.action_dim = action_shape[0]
        self.beta_param_offset = float(concentration_offset)
        self.eps = float(eps)

        self.input_norm = nn.LayerNorm(repr_dim)
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.alpha_linear = nn.Linear(hidden_dim, self.action_dim)
        self.beta_linear = nn.Linear(hidden_dim, self.action_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def distribution_params(self, obs):
        obs = self.input_norm(obs)
        latent = self.trunk(obs)
        alpha = F.softplus(self.alpha_linear(latent)) + self.beta_param_offset + self.eps
        beta = F.softplus(self.beta_linear(latent)) + self.beta_param_offset + self.eps
        return alpha, beta

    def _distribution(self, obs):
        alpha, beta = self.distribution_params(obs)
        return pyd.Beta(alpha, beta), alpha, beta

    def _sample_action_log_prob(self, obs, compute_log_pi=True):
        dist, alpha, beta = self._distribution(obs)
        unit_action = dist.rsample().clamp(self.eps, 1.0 - self.eps)
        action = 2.0 * unit_action - 1.0
        if compute_log_pi:
            log_prob = dist.log_prob(unit_action).sum(dim=-1, keepdim=True)
            log_prob -= self.action_dim * math.log(2.0)
        else:
            log_prob = None
        return action, log_prob, alpha, beta

    def action_log_prob(self, obs):
        action, log_prob, _, _ = self._sample_action_log_prob(obs, compute_log_pi=True)
        return action, log_prob

    def forward(self, obs, deterministic=False, compute_pi=True, compute_log_pi=True, with_log_prob=False):
        if not compute_pi:
            return self.distribution_params(obs)

        if deterministic:
            alpha, beta = self.distribution_params(obs)
            unit_action = alpha / (alpha + beta)
            action = 2.0 * unit_action - 1.0
            if with_log_prob:
                return action, None, alpha, beta
            return action

        action, log_prob, alpha, beta = self._sample_action_log_prob(
            obs,
            compute_log_pi=compute_log_pi,
        )
        if with_log_prob:
            return action, log_prob, alpha, beta
        return action

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            if deterministic:
                alpha, beta = self.distribution_params(obs)
                unit_action = alpha / (alpha + beta)
                return 2.0 * unit_action - 1.0
            action, _, _, _ = self._sample_action_log_prob(obs, compute_log_pi=False)
            return action


def _beta_kwargs(args):
    return {
        "concentration_offset": float(get_algo_param(args, "beta_concentration_offset", 1.0)),
        "eps": float(get_algo_param(args, "beta_concentration_eps", 1e-6)),
    }


def _replace_base_actor(agent, action_shape, state_dim_attr="state_dim"):
    kwargs = _beta_kwargs(agent.args)
    state_dim = getattr(agent, state_dim_attr)
    agent.actor = BetaFeedForwardActor(state_dim, action_shape, agent.args.hidden_dim, **kwargs).to(agent.device)
    agent.actor_params = (
        list(agent.actor.parameters())
        + list(agent.actor_encoder.parameters())
        + list(agent.actor_base_adapter.parameters())
    )
    agent.actor_optimizer = Adam(agent.actor_params, lr=agent.args.actor_lr)


class SACBetaAgent(SACAgent):
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)
        _replace_base_actor(self, action_space.shape)


class PLSACBetaAgent(PLSACAgent):
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)
        _replace_base_actor(self, action_space.shape)


class STVimSACBetaAgent(STVimSACAgent):
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)
        _replace_base_actor(self, action_space.shape)


class DPERSTVimSACBetaAgent(DPERSTVimSACAgent):
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)
        _replace_base_actor(self, action_space.shape)


class PLDPERSTVimSACBetaAgent(PLDPERSTVimSACAgent):
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)
        _replace_base_actor(self, action_space.shape, state_dim_attr="actor_state_dim")


__all__ = [
    "SACBetaAgent",
    "PLSACBetaAgent",
    "STVimSACBetaAgent",
    "DPERSTVimSACBetaAgent",
    "PLDPERSTVimSACBetaAgent",
]
