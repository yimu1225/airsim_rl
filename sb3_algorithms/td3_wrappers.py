"""TD3 wrappers whose differences are provided by feature extractors."""

from __future__ import annotations

from stable_baselines3 import TD3


class VimTD3(TD3):
    pass


class STVimTD3(TD3):
    pass


class STSeqVimTD3(TD3):
    pass


class STVSeqVimTD3(TD3):
    pass


class VimPatchTD3(TD3):
    pass


class STSVimTD3(TD3):
    pass


class MambaTD3(TD3):
    pass


class DualVimTD3(TD3):
    pass


class STVimAsymTD3(TD3):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        from sb3_extensions.policies import AsymTD3Policy

        if policy == "MultiInputPolicy":
            policy = AsymTD3Policy
        super().__init__(policy, *args, **kwargs)
