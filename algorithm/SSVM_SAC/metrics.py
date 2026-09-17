"""Training metrics compatible with the project's shared curve plotter."""

from __future__ import annotations

import csv
from pathlib import Path

from result_paths import (
    DEFAULT_RESULTS_ROOT,
    initialize_training_log_csv,
    reset_training_run_dir,
    training_log_path,
    training_run_dir,
)


class FinalStageCurveLogger:
    """Write final-stage episode metrics using the shared training CSV format."""

    def __init__(
        self,
        algorithm_name: str,
        seed: int,
        *,
        results_root: str | Path = DEFAULT_RESULTS_ROOT,
        overwrite: bool = False,
    ) -> None:
        self.algorithm_name = str(algorithm_name)
        self.seed = int(seed)
        self.results_root = Path(results_root)
        self.run_dir = training_run_dir(
            self.algorithm_name,
            self.seed,
            results_root=self.results_root,
        )
        self.csv_path = training_log_path(
            self.algorithm_name,
            self.seed,
            results_root=self.results_root,
        )

        if self.csv_path.exists() and not overwrite:
            raise FileExistsError(
                f"Training curve already exists: {self.csv_path}. "
                "Pass --overwrite-results to start a new final-stage run."
            )

        if overwrite:
            reset_training_run_dir(
                self.algorithm_name,
                self.seed,
                results_root=self.results_root,
            )
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = initialize_training_log_csv(
            self.algorithm_name,
            self.seed,
            results_root=self.results_root,
        )

    def record(
        self,
        *,
        episode: int,
        total_timesteps: int,
        reward: float,
        episode_length: int,
        success_rate: float,
    ) -> None:
        """Append one completed episode to the shared training log."""
        with self.csv_path.open(mode="a", newline="") as stream:
            csv.writer(stream).writerow(
                [
                    int(episode),
                    int(total_timesteps),
                    float(reward),
                    int(episode_length),
                    float(success_rate),
                ]
            )
