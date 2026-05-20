"""Privileged-learning critic/value policies for AirSim control."""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BaseModel, ContinuousCritic
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy
from torch import nn


def _as_privileged_keys(privileged_key) -> tuple[str, ...]:
    if privileged_key is None:
        return tuple()
    if isinstance(privileged_key, str):
        return (privileged_key,)
    return tuple(str(key) for key in privileged_key)


class PLContinuousCritic(ContinuousCritic):
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
        privileged_key: str | tuple[str, ...] = ("clean_base", "clean_depth"),
    ) -> None:
        self.privileged_key = privileged_key
        self.privileged_keys = _as_privileged_keys(privileged_key)
        self.privileged_dim = 0
        if isinstance(observation_space, spaces.Dict):
            self.privileged_dim = sum(
                int(spaces.utils.flatdim(observation_space.spaces[key]))
                for key in self.privileged_keys
                if key in observation_space.spaces
            )
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
        features = []
        for key in self.privileged_keys:
            if key not in obs:
                continue
            value = obs[key].float()
            if value.dim() == 1:
                value = value.unsqueeze(0)
            features.append(value.reshape(value.shape[0], -1))
        if not features:
            return None
        return th.cat(features, dim=1)

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


class PLActorCriticPolicy(MultiInputActorCriticPolicy):
    """PPO actor-critic policy with privileged inputs for the value branch only."""

    def __init__(self, *args, privileged_key: str | tuple[str, ...] = ("clean_base", "clean_depth"), **kwargs) -> None:
        observation_space = args[0] if args else kwargs.get("observation_space")
        self.privileged_key = privileged_key
        self.privileged_keys = _as_privileged_keys(privileged_key)
        self.privileged_dim = self._infer_privileged_dim(observation_space, self.privileged_keys)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _infer_privileged_dim(observation_space: spaces.Space | None, privileged_keys: tuple[str, ...]) -> int:
        if not isinstance(observation_space, spaces.Dict):
            return 0
        return sum(
            int(spaces.utils.flatdim(observation_space.spaces[key]))
            for key in privileged_keys
            if key in observation_space.spaces
        )

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(privileged_key=self.privileged_key)
        return data

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim + self.privileged_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _privileged_features(self, obs: PyTorchObs, batch_size: int, dtype: th.dtype, device: th.device) -> th.Tensor:
        if self.privileged_dim <= 0:
            return th.empty((batch_size, 0), dtype=dtype, device=device)
        if isinstance(obs, dict):
            pieces = []
            for key in self.privileged_keys:
                if key not in obs:
                    continue
                privileged = obs[key].float()
                if privileged.dim() == 1:
                    privileged = privileged.unsqueeze(0)
                pieces.append(privileged.reshape(privileged.shape[0], -1))
            if not pieces:
                return th.zeros((batch_size, self.privileged_dim), dtype=dtype, device=device)
            privileged = th.cat(pieces, dim=1)
            if privileged.dim() == 1:
                privileged = privileged.unsqueeze(0)
            privileged = privileged.reshape(privileged.shape[0], -1).to(device=device, dtype=dtype)
            if privileged.shape[1] > self.privileged_dim:
                privileged = privileged[:, : self.privileged_dim]
            elif privileged.shape[1] < self.privileged_dim:
                pad = th.zeros(
                    (privileged.shape[0], self.privileged_dim - privileged.shape[1]),
                    dtype=dtype,
                    device=device,
                )
                privileged = th.cat([privileged, pad], dim=1)
            return privileged
        return th.zeros((batch_size, self.privileged_dim), dtype=dtype, device=device)

    def _actor_features(self, features: th.Tensor) -> th.Tensor:
        if self.privileged_dim <= 0:
            return features
        zeros = th.zeros((features.shape[0], self.privileged_dim), dtype=features.dtype, device=features.device)
        return th.cat([features, zeros], dim=1)

    def _critic_features(self, features: th.Tensor, obs: PyTorchObs) -> th.Tensor:
        privileged = self._privileged_features(obs, features.shape[0], features.dtype, features.device)
        return th.cat([features, privileged], dim=1)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi = self.mlp_extractor.forward_actor(self._actor_features(features))
            latent_vf = self.mlp_extractor.forward_critic(self._critic_features(features, obs))
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(self._actor_features(pi_features))
            latent_vf = self.mlp_extractor.forward_critic(self._critic_features(vf_features, obs))
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi = self.mlp_extractor.forward_actor(self._actor_features(features))
            latent_vf = self.mlp_extractor.forward_critic(self._critic_features(features, obs))
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(self._actor_features(pi_features))
            latent_vf = self.mlp_extractor.forward_critic(self._critic_features(vf_features, obs))
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs):
        features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(self._actor_features(features))
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(self._critic_features(features, obs))
        return self.value_net(latent_vf)


class PLTD3Policy(TD3Policy):
    """TD3 policy with a privileged-learning critic."""

    def __init__(self, *args, privileged_key: str | tuple[str, ...] = ("clean_base", "clean_depth"), **kwargs) -> None:
        self.privileged_key = privileged_key
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> PLContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(privileged_key=self.privileged_key)
        return PLContinuousCritic(**critic_kwargs).to(self.device)


class PLSACPolicy(SACPolicy):
    """SAC policy with a privileged-learning critic."""

    def __init__(self, *args, privileged_key: str | tuple[str, ...] = ("clean_base", "clean_depth"), **kwargs) -> None:
        self.privileged_key = privileged_key
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> PLContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(privileged_key=self.privileged_key)
        return PLContinuousCritic(**critic_kwargs).to(self.device)
