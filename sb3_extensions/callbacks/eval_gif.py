"""Evaluation callback that can optionally save GIF rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class EvalAndSaveGifCallback(EvalCallback):
    def __init__(self, *args, gif_path: str | Path | None = None, save_gifs: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gif_path = Path(gif_path) if gif_path is not None else None
        self.save_gifs = bool(save_gifs)

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.save_gifs and self.gif_path is not None and self.n_calls % self.eval_freq == 0:
            self._save_depth_gif()
        return continue_training

    def _save_depth_gif(self) -> None:
        try:
            import imageio.v2 as imageio

            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            frames = []
            for _ in range(300):
                action, _ = self.model.predict(obs, deterministic=True)
                step_out = self.eval_env.step(action)
                if len(step_out) == 5:
                    obs, _, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, _, done, _ = step_out
                depth = obs["depth"] if isinstance(obs, dict) else None
                if depth is not None:
                    frame = np.asarray(depth)
                    if frame.ndim >= 3:
                        frame = frame[-1]
                    frames.append(np.clip(frame, 0, 255).astype(np.uint8))
                if done:
                    break
            if frames:
                self.gif_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(self.gif_path, frames)
        except Exception as exc:
            if self.verbose:
                print(f"EvalAndSaveGifCallback skipped GIF save: {exc}")

