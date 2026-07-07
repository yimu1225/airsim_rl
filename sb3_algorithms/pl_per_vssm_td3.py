"""PER + ST-Vim TD3 with a privileged-learning critic."""

from __future__ import annotations

from sb3_algorithms.per_vssm_td3 import PERVSSM_TD3
from sb3_extensions.policies import PLTD3Policy


class PLPERVSSM_TD3(PERVSSM_TD3):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = PLTD3Policy
        super().__init__(policy, *args, **kwargs)
