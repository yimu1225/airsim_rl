"""PPO wrappers whose differences are provided by feature extractors."""

from __future__ import annotations

from stable_baselines3 import PPO

from sb3_extensions.policies import PLActorCriticPolicy


class STVimPPO(PPO):
    pass


class PLSTVimPPO(PPO):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = PLActorCriticPolicy
        super().__init__(policy, *args, **kwargs)
