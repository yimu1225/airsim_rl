from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from algo_name_utils import to_internal_core_algorithm_name


_ALGO_DIR = Path(__file__).resolve().parent

def _resolve_algorithm_dir(algorithm_name: str) -> Path:
    core_name = to_internal_core_algorithm_name(algorithm_name)
    folder_name = core_name
    folder_path = _ALGO_DIR / folder_name
    if not folder_path.is_dir():
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. Expected folder '{folder_name}' under '{_ALGO_DIR}'."
        )
    return folder_path


def load_algorithm_params(algorithm_name: str) -> Tuple[Dict[str, Any], Path]:
    """
    Load algorithm-specific parameters from:
      algorithm/<algorithm_folder>/params.yaml

    Returns:
      params: dict of algorithm-specific parameters
      config_path: resolved YAML path
    """
    algo_dir = _resolve_algorithm_dir(algorithm_name)
    config_path = algo_dir / "params.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing algorithm parameter file: {config_path}. "
            f"Please add params.yaml in '{algo_dir.name}'."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}, config_path

    if not isinstance(raw, dict):
        raise ValueError(f"Algorithm params file must be a YAML mapping: {config_path}")

    params = raw.get("params", raw)
    if not isinstance(params, dict):
        raise ValueError(
            f"Algorithm params file must contain a mapping (or top-level 'params' mapping): {config_path}"
        )

    return params, config_path


def apply_algorithm_params(args: argparse.Namespace, algorithm_name: str) -> Tuple[Path, list[str]]:
    """
    Apply algorithm-specific params onto argparse namespace.

    Later-loaded algorithm params intentionally override same-name attributes to
    ensure each algorithm run gets its own defaults.
    """
    params, config_path = load_algorithm_params(algorithm_name)
    # Keep an explicit algorithm-local view for in-algorithm access.
    setattr(args, "algorithm_params", dict(params))
    for key, value in params.items():
        setattr(args, key, value)
    return config_path, sorted(params.keys())


def get_algo_param(args: argparse.Namespace, key: str, default: Any = None) -> Any:
    """
    Read algorithm-specific parameter with precedence:
      1) args.algorithm_params[key]
      2) args.<key>
      3) provided default
    """
    algo_params = getattr(args, "algorithm_params", None)
    if isinstance(algo_params, dict) and key in algo_params:
        return algo_params[key]
    if hasattr(args, key):
        return getattr(args, key)
    return default


__all__ = ["load_algorithm_params", "apply_algorithm_params", "get_algo_param"]
