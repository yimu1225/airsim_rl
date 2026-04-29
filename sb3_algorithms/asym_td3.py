"""Asymmetric TD3."""

from __future__ import annotations

from stable_baselines3 import TD3

from sb3_extensions.policies import AsymTD3Policy


class AsymTD3(TD3):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = AsymTD3Policy
        super().__init__(policy, *args, **kwargs)

