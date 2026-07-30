"""Input-level MambaLRP support for CL-VSSM-SAC."""

from .attribution import (
    MambaLRPResult,
    compute_mambalrp,
    evaluate_patch_flipping,
    select_spaced_top_indices,
)

__all__ = [
    "MambaLRPResult",
    "compute_mambalrp",
    "evaluate_patch_flipping",
    "select_spaced_top_indices",
]
