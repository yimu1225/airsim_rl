"""TD3 wrappers whose differences are provided by feature extractors."""

from __future__ import annotations

from stable_baselines3 import TD3


class VimTD3(TD3):
    pass


class VMTD3(TD3):
    pass


class STSeqVimTD3(TD3):
    pass


class STVSeqVimTD3(TD3):
    pass


class VimPatchTD3(TD3):
    pass


class SAFEVMTD3(TD3):
    pass


class MambaTD3(TD3):
    pass


class DualVimTD3(TD3):
    pass


class PLVMTD3(TD3):
    def __init__(self, policy="MultiInputPolicy", *args, **kwargs) -> None:
        from sb3_extensions.policies import PLTD3Policy

        if policy == "MultiInputPolicy":
            policy = PLTD3Policy
        super().__init__(policy, *args, **kwargs)
