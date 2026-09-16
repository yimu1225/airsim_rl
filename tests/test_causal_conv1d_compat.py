from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def test_vendored_mamba_uses_causal_conv1d_public_cpp_wrappers():
    interface_path = (
        Path(__file__).parents[1]
        / "Vim"
        / "mamba-1p1p1"
        / "mamba_ssm"
        / "ops"
        / "selective_scan_interface.py"
    )
    module_name = "_test_selective_scan_interface"
    spec = importlib.util.spec_from_file_location(module_name, interface_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    from causal_conv1d.cpp_functions import (
        causal_conv1d_bwd_function,
        causal_conv1d_fwd_function,
    )

    assert module.causal_conv1d_fwd_function is causal_conv1d_fwd_function
    assert module.causal_conv1d_bwd_function is causal_conv1d_bwd_function
