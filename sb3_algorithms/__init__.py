"""SB3 algorithm extensions for AirSim RL."""

from sb3_algorithms.pl_td3 import PLTD3
from sb3_algorithms.pl_per_td3 import PLPERTD3
from sb3_algorithms.pl_per_vmsac import PLPERVMSAC
from sb3_algorithms.pl_per_vmtd3 import PLPERVMTD3
from sb3_algorithms.per_sac import PERSAC
from sb3_algorithms.per_vmsac import PERVMSAC
from sb3_algorithms.per_vmtd3 import PERVMTD3
from sb3_algorithms.per_td3 import PERTD3
from sb3_algorithms.ppo_wrappers import PLSTVimPPO, STVimPPO
from sb3_algorithms.sac_wrappers import LSTMSAC, PLSAC, VMSAC
from sb3_algorithms.td3_wrappers import (
    DualVimTD3,
    MambaTD3,
    STSeqVimTD3,
    SAFEVMTD3,
    STVSeqVimTD3,
    PLVMTD3,
    VMTD3,
    VimPatchTD3,
    VimTD3,
)

__all__ = [
    "DualVimTD3",
    "LSTMSAC",
    "MambaTD3",
    "PLPERTD3",
    "PLPERVMSAC",
    "PLPERVMTD3",
    "PLSAC",
    "PLSTVimPPO",
    "PLVMTD3",
    "PLTD3",
    "PERSAC",
    "PERVMSAC",
    "PERVMTD3",
    "PERTD3",
    "STSeqVimTD3",
    "SAFEVMTD3",
    "STVSeqVimTD3",
    "STVimPPO",
    "VMSAC",
    "VMTD3",
    "VimPatchTD3",
    "VimTD3",
]
