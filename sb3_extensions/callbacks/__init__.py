"""Callbacks for SB3-based AirSim training."""

from sb3_extensions.callbacks.airsim_health import AirSimHealthCallback
from sb3_extensions.callbacks.curriculum import CurriculumCallback
from sb3_extensions.callbacks.depth_render import DepthRenderCallback
from sb3_extensions.callbacks.eval_gif import EvalAndSaveGifCallback
from sb3_extensions.callbacks.csv_logger import CSVLoggerCallback

__all__ = [
    "AirSimHealthCallback",
    "CSVLoggerCallback",
    "CurriculumCallback",
    "DepthRenderCallback",
    "EvalAndSaveGifCallback",
]
