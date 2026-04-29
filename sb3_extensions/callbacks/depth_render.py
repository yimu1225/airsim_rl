"""Optional depth observation preview callback."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DepthRenderCallback(BaseCallback):
    def __init__(self, render_freq: int = 100, window_name: str = "AirSim depth", scale: float = 2.5, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.render_freq = max(1, int(render_freq))
        self.window_name = window_name
        self.scale = float(scale)

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq != 0:
            return True
        try:
            import cv2

            obs = self.locals.get("new_obs")
            if isinstance(obs, dict):
                depth = obs.get("depth")
            else:
                depth = None
            if depth is None:
                return True
            frame = np.asarray(depth)
            if frame.ndim >= 4:
                frame = frame[0, -1]
            elif frame.ndim == 3:
                frame = frame[-1]
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            if self.scale != 1.0:
                frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
        except Exception as exc:
            if self.verbose:
                print(f"DepthRenderCallback skipped render: {exc}")
        return True

