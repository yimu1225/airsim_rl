"""Asymmetric critic policies for TD3."""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.td3.policies import TD3Policy
from torch import nn


class AsymContinuousCritic(ContinuousCritic):
    """Critic that appends raw privileged observation features to encoded state features."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        privileged_key: str = "base",
    ) -> None:
        self.privileged_key = privileged_key
        self.privileged_dim = 0
        if isinstance(observation_space, spaces.Dict) and privileged_key in observation_space.spaces:
            self.privileged_dim = int(spaces.utils.flatdim(observation_space.spaces[privileged_key]))
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        input_dim = features_dim + self.privileged_dim + action_dim
        for idx in range(n_critics):
            q_net = nn.Sequential(*create_mlp(input_dim, 1, net_arch, activation_fn))
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def _privileged_features(self, obs) -> th.Tensor | None:
        if self.privileged_dim <= 0 or not isinstance(obs, dict):
            return None
        value = obs[self.privileged_key].float()
        if value.dim() == 1:
            value = value.unsqueeze(0)
        return value.reshape(value.shape[0], -1)

    def _q_input(self, obs, actions: th.Tensor, detach_features: bool = False) -> th.Tensor:
        grad_enabled = (not self.share_features_extractor) and (not detach_features)
        with th.set_grad_enabled(grad_enabled):
            features = self.extract_features(obs, self.features_extractor)
        privileged = self._privileged_features(obs)
        if privileged is not None:
            features = th.cat([features, privileged], dim=1)
        return th.cat([features, actions], dim=1)

    def forward(self, obs, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        qvalue_input = self._q_input(obs, actions)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs, actions: th.Tensor) -> th.Tensor:
        qvalue_input = self._q_input(obs, actions, detach_features=True)
        return self.q_networks[0](qvalue_input)


class AsymTD3Policy(TD3Policy):
    """TD3 policy with an asymmetric critic."""

    def __init__(self, *args, privileged_key: str = "base", **kwargs) -> None:
        self.privileged_key = privileged_key
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> AsymContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(privileged_key=self.privileged_key)
        return AsymContinuousCritic(**critic_kwargs).to(self.device)

