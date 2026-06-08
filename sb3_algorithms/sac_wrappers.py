"""SAC wrappers whose differences are provided by feature extractors."""

from __future__ import annotations

from stable_baselines3 import SAC

from sb3_extensions.policies import PLSACPolicy


class LSTMSAC(SAC):
    pass


class VMSAC(SAC):
    pass


class PLSAC(SAC):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = PLSACPolicy
        super().__init__(policy, *args, **kwargs)
