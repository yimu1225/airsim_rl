"""AirSim/UE health recovery callback."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class AirSimHealthCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.check_freq = max(1, int(check_freq))

    def _check(self) -> None:
        try:
            results = self.training_env.env_method("check_ue4_status")
            if any(bool(item) for item in results):
                self.logger.record("airsim/recovered", 1)
        except Exception as exc:
            if self.verbose:
                print(f"AirSimHealthCallback skipped health check: {exc}")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._check()
        return True

    def _on_rollout_end(self) -> None:
        self._check()

