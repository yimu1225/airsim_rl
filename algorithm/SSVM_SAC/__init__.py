"""SSVM-SAC: self-supervised visual memory-enhanced soft actor-critic."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "SSVMConfig": (".config", "SSVMConfig"),
    "EpisodeArchive": (".dataset", "EpisodeArchive"),
    "EpisodeSequenceDataset": (".dataset", "EpisodeSequenceDataset"),
    "DepthFrameDataset": (".dataset", "DepthFrameDataset"),
    "MambaPerceptionMemory": (".networks", "MambaPerceptionMemory"),
    "VisionMambaEncoder": (".networks", "VisionMambaEncoder"),
    "VisionMambaDecoder": (".networks", "VisionMambaDecoder"),
    "MultiFrameMambaReconstructor": (".networks", "MultiFrameMambaReconstructor"),
    "TemporalMambaMemory": (".networks", "TemporalMambaMemory"),
    "SSVMSACAgent": (".agent", "SSVMSACAgent"),
    "DualPrioritizedFeatureReplayBuffer": (
        ".buffer",
        "DualPrioritizedFeatureReplayBuffer",
    ),
    "FinalStageCurveLogger": (".metrics", "FinalStageCurveLogger"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
