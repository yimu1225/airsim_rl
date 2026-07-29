"""Shared paths for algorithm-specific training results."""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
from typing import Union


PathLike = Union[str, Path]
DEFAULT_RESULTS_ROOT = Path("results")
TRAINING_LOG_COLUMNS = (
    "episode",
    "total_timesteps",
    "reward",
    "episode_length",
    "success_rate",
)


def algorithm_results_dir(
    algorithm_name: str,
    *,
    results_root: PathLike = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Return ``results/<algorithm>`` after validating the directory name."""
    algorithm = str(algorithm_name).strip()
    if not algorithm or Path(algorithm).name != algorithm or algorithm in {".", ".."}:
        raise ValueError(
            f"algorithm_name must be one directory name, got {algorithm_name!r}"
        )
    return Path(results_root) / algorithm


def training_run_dir(
    algorithm_name: str,
    seed: int,
    *,
    results_root: PathLike = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Return ``results/<algorithm>/seed<seed>`` for one training run."""
    return algorithm_results_dir(
        algorithm_name,
        results_root=results_root,
    ) / f"seed{int(seed)}"


def training_log_path(
    algorithm_name: str,
    seed: int,
    *,
    results_root: PathLike = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Return the standard CSV path for one training run."""
    algorithm = str(algorithm_name).strip()
    return training_run_dir(
        algorithm,
        seed,
        results_root=results_root,
    ) / f"{algorithm}_seed{int(seed)}_log.csv"


def reset_training_run_dir(
    algorithm_name: str,
    seed: int,
    *,
    results_root: PathLike = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Recreate one run directory and return its path."""
    run_dir = training_run_dir(
        algorithm_name,
        seed,
        results_root=results_root,
    )
    if run_dir.exists():
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Training result path is not a directory: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def initialize_training_log_csv(
    algorithm_name: str,
    seed: int,
    *,
    results_root: PathLike = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Create the standard training CSV and return its path."""
    csv_path = training_log_path(
        algorithm_name,
        seed,
        results_root=results_root,
    )
    with csv_path.open(mode="w", newline="") as file:
        csv.writer(file).writerow(TRAINING_LOG_COLUMNS)
    return csv_path


def discover_training_logs(
    algorithm_name: str,
    *,
    results_root: PathLike = DEFAULT_RESULTS_ROOT,
) -> list[Path]:
    """Return standard CSV logs under one algorithm's seed directories."""
    algorithm = str(algorithm_name).strip()
    algorithm_dir = algorithm_results_dir(
        algorithm,
        results_root=results_root,
    )
    return sorted(
        path
        for path in algorithm_dir.glob(
            f"seed*/{algorithm}_seed*_log.csv"
        )
        if path.is_file()
    )
