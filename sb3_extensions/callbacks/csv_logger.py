"""Small CSV logger callback for SB3 training."""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_path: str | Path, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.log_freq = max(1, int(log_freq))
        self._file = None
        self._writer = None

    def _init_callback(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=["timesteps", "episodes"])
        if self.log_path.stat().st_size == 0:
            self._writer.writeheader()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 and self._writer is not None:
            self._writer.writerow(
                {
                    "timesteps": int(self.num_timesteps),
                    "episodes": int(getattr(self.model, "_episode_num", 0)),
                }
            )
            self._file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

