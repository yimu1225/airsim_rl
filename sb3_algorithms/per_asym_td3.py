"""PER + asymmetric critic TD3."""

from __future__ import annotations

from sb3_algorithms.per_td3 import PERTD3
from sb3_extensions.policies import AsymTD3Policy


class PERAsymTD3(PERTD3):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = AsymTD3Policy
        super().__init__(policy, *args, **kwargs)

