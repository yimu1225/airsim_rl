"""SB3 algorithm extensions for AirSim RL."""

from sb3_algorithms.pl_td3 import PLTD3
from sb3_algorithms.pl_per_td3 import PLPERTD3
from sb3_algorithms.pl_per_st_vim_sac import PLPERSTVimSAC
from sb3_algorithms.per_sac import PERSAC
from sb3_algorithms.per_st_vim_sac import PERSTVimSAC
from sb3_algorithms.per_st_vim_td3 import PERSTVimTD3
from sb3_algorithms.per_td3 import PERTD3
from sb3_algorithms.ppo_wrappers import STVimPPO
from sb3_algorithms.sac_wrappers import LSTMSAC, PLSAC, STVimSAC
from sb3_algorithms.td3_wrappers import (
    DualVimTD3,
    MambaTD3,
    STSeqVimTD3,
    STSVimTD3,
    STVSeqVimTD3,
    PLSTVimTD3,
    STVimTD3,
    VimPatchTD3,
    VimTD3,
)

__all__ = [
    "DualVimTD3",
    "LSTMSAC",
    "MambaTD3",
    "PLPERTD3",
    "PLPERSTVimSAC",
    "PLSAC",
    "PLSTVimTD3",
    "PLTD3",
    "PERSAC",
    "PERSTVimSAC",
    "PERSTVimTD3",
    "PERTD3",
    "STSeqVimTD3",
    "STSVimTD3",
    "STVSeqVimTD3",
    "STVimPPO",
    "STVimSAC",
    "STVimTD3",
    "VimPatchTD3",
    "VimTD3",
]
