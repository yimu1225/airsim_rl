"""PER + privileged-learning TD3."""

from __future__ import annotations

from sb3_algorithms.per_td3 import PERTD3
from sb3_extensions.policies import PLTD3Policy


class PLPERTD3(PERTD3):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = PLTD3Policy
        super().__init__(policy, *args, **kwargs)
