"""SB3 algorithm extensions for AirSim RL."""

from sb3_algorithms.pl_td3 import PLTD3
from sb3_algorithms.pl_per_td3 import PLPERTD3
from sb3_algorithms.pl_per_vssm_sac import PLPERVSSM_SAC
from sb3_algorithms.pl_per_vssm_td3 import PLPERVSSM_TD3
from sb3_algorithms.per_sac import PERSAC
from sb3_algorithms.per_vssm_sac import PERVSSM_SAC
from sb3_algorithms.per_vssm_td3 import PERVSSM_TD3
from sb3_algorithms.per_td3 import PERTD3
from sb3_algorithms.ppo_wrappers import PLSTVimPPO, STVimPPO
from sb3_algorithms.sac_wrappers import LSTMSAC, PLSAC, VSSM_SAC
from sb3_algorithms.td3_wrappers import (
    DualVimTD3,
    MambaTD3,
    STSeqVimTD3,
    SAFE_VSSM_TD3,
    STVSeqVimTD3,
    PLVSSM_TD3,
    VSSM_TD3,
    VimPatchTD3,
    VimTD3,
)

__all__ = [
    "DualVimTD3",
    "LSTMSAC",
    "MambaTD3",
    "PLPERTD3",
    "PLPERVSSM_SAC",
    "PLPERVSSM_TD3",
    "PLSAC",
    "PLSTVimPPO",
    "PLVSSM_TD3",
    "PLTD3",
    "PERSAC",
    "PERVSSM_SAC",
    "PERVSSM_TD3",
    "PERTD3",
    "STSeqVimTD3",
    "SAFE_VSSM_TD3",
    "STVSeqVimTD3",
    "STVimPPO",
    "VSSM_SAC",
    "VSSM_TD3",
    "VimPatchTD3",
    "VimTD3",
]
