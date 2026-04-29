"""Curriculum progression callback."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq: int = 5000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.check_freq = max(1, int(check_freq))

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        try:
            rates = self.training_env.env_method("get_success_rate")
            levels_before = self.training_env.get_attr("level")
            changed = self.training_env.env_method("increase_level")
            levels_after = self.training_env.get_attr("level")
            if rates:
                self.logger.record("curriculum/success_rate", float(rates[0]))
            if levels_after:
                self.logger.record("curriculum/level", int(levels_after[0]))
            if any(bool(item) for item in changed) and self.verbose:
                print(f"Curriculum level changed: {levels_before} -> {levels_after}")
        except Exception as exc:
            if self.verbose:
                print(f"CurriculumCallback skipped progression check: {exc}")
        return True
