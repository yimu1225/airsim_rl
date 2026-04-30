"""Privileged-learning TD3."""

from __future__ import annotations

from stable_baselines3 import TD3

from sb3_extensions.policies import PLTD3Policy


class PLTD3(TD3):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = PLTD3Policy
        super().__init__(policy, *args, **kwargs)
