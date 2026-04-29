"""SB3 algorithm extensions for AirSim RL."""

from sb3_algorithms.asym_td3 import AsymTD3
from sb3_algorithms.per_asym_td3 import PERAsymTD3
from sb3_algorithms.per_sac import PERSAC
from sb3_algorithms.per_st_vim_sac import PERSTVimSAC
from sb3_algorithms.per_st_vim_td3 import PERSTVimTD3
from sb3_algorithms.per_td3 import PERTD3
from sb3_algorithms.ppo_wrappers import STVimPPO
from sb3_algorithms.sac_wrappers import LSTMSAC, STVimSAC
from sb3_algorithms.td3_wrappers import (
    DualVimTD3,
    MambaTD3,
    STSeqVimTD3,
    STSVimTD3,
    STVSeqVimTD3,
    STVimAsymTD3,
    STVimTD3,
    VimPatchTD3,
    VimTD3,
)

__all__ = [
    "AsymTD3",
    "DualVimTD3",
    "LSTMSAC",
    "MambaTD3",
    "PERAsymTD3",
    "PERSAC",
    "PERSTVimSAC",
    "PERSTVimTD3",
    "PERTD3",
    "STSeqVimTD3",
    "STSVimTD3",
    "STVSeqVimTD3",
    "STVimAsymTD3",
    "STVimPPO",
    "STVimSAC",
    "STVimTD3",
    "VimPatchTD3",
    "VimTD3",
]
