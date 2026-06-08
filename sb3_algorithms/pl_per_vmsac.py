"""PER + ST-Vim SAC with a privileged-learning critic."""

from __future__ import annotations

from sb3_algorithms.per_sac import PERSAC
from sb3_extensions.policies import PLSACPolicy


class PLPERSTVimSAC(PERSAC):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        if policy == "MultiInputPolicy":
            policy = PLSACPolicy
        super().__init__(policy, *args, **kwargs)
