#!/usr/bin/env python3
"""Compatibility entry point for the modular ``eval.mambalrp`` package."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mambalrp import (
    MambaLRPResult,
    compute_mambalrp,
    evaluate_patch_flipping,
    select_spaced_top_indices,
)
from eval.mambalrp.runner import main, run_visualization
from eval.mambalrp.self_test import run_self_tests


__all__ = [
    "MambaLRPResult",
    "compute_mambalrp",
    "evaluate_patch_flipping",
    "run_self_tests",
    "run_visualization",
    "select_spaced_top_indices",
]


if __name__ == "__main__":
    main()
