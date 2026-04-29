"""YAML parameter loader for the SB3 migration layer.

The Vim-family folders intentionally keep their original mixed/upper-case names
(`ST_Seq_Vim_TD3`, `STV_Seq_Vim_TD3`, ...), matching the existing project
convention while staying self-contained in the SB3 migration layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from algo_name_utils import to_internal_core_algorithm_name


_PARAMS_DIR = Path(__file__).resolve().parent / "params"

def _resolve_algorithm_dir(algorithm_name: str) -> Path:
    core_name = to_internal_core_algorithm_name(algorithm_name)
    folder_name = core_name
    folder_path = _PARAMS_DIR / folder_name
    if not folder_path.is_dir():
        raise ValueError(
            f"Unknown SB3 algorithm '{algorithm_name}'. Expected params folder '{folder_name}' under '{_PARAMS_DIR}'."
        )
    return folder_path


def load_algorithm_params(algorithm_name: str) -> tuple[dict[str, Any], Path]:
    """Load algorithm-specific parameters from `sb3_algorithms/params/<Algo>/params.yaml`."""

    config_path = _resolve_algorithm_dir(algorithm_name) / "params.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing SB3 algorithm parameter file: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if raw is None:
        return {}, config_path
    if not isinstance(raw, dict):
        raise ValueError(f"Algorithm params file must be a YAML mapping: {config_path}")

    params = raw.get("params", raw)
    if not isinstance(params, dict):
        raise ValueError(f"Algorithm params file must contain a mapping: {config_path}")
    return params, config_path


def apply_algorithm_params(args: argparse.Namespace, algorithm_name: str) -> tuple[Path, list[str]]:
    """Apply YAML parameters to the argparse namespace for a single algorithm run."""

    params, config_path = load_algorithm_params(algorithm_name)
    setattr(args, "algorithm_params", dict(params))
    for key, value in params.items():
        setattr(args, key, value)
    return config_path, sorted(params.keys())


def get_algo_param(args: argparse.Namespace, key: str, default: Any = None) -> Any:
    """Read a parameter from YAML-backed `args.algorithm_params`, then args, then default."""

    algo_params = getattr(args, "algorithm_params", None)
    if isinstance(algo_params, dict) and key in algo_params:
        return algo_params[key]
    if hasattr(args, key):
        return getattr(args, key)
    return default


__all__ = ["apply_algorithm_params", "get_algo_param", "load_algorithm_params"]
