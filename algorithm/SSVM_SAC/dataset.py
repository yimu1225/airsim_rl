"""Episode storage and causal sequence datasets for SSVM-SAC."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_FIELDS = ("depth", "base", "action", "reward", "terminated", "truncated")


class EpisodeArchive:
    """One compressed file per episode, so sampled windows never cross resets."""

    @staticmethod
    def path(root: str | Path, episode_id: int) -> Path:
        return Path(root) / f"episode_{episode_id:08d}.npz"

    @classmethod
    def save(
        cls,
        root: str | Path,
        *,
        episode_id: int,
        depth: np.ndarray,
        base: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        clean_depth: np.ndarray | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        arrays = {
            "depth": np.asarray(depth),
            "base": np.asarray(base),
            "action": np.asarray(action),
            "reward": np.asarray(reward),
            "terminated": np.asarray(terminated, dtype=bool),
            "truncated": np.asarray(truncated, dtype=bool),
        }
        lengths = {name: len(value) for name, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"episode fields have different lengths: {lengths}")
        if not lengths["depth"]:
            raise ValueError("cannot save an empty episode")
        if clean_depth is not None:
            clean_depth = np.asarray(clean_depth)
            if len(clean_depth) != lengths["depth"]:
                raise ValueError("clean_depth must have the same length as depth")
            arrays["clean_depth"] = clean_depth
        arrays["metadata"] = np.asarray(json.dumps(dict(metadata or {})))

        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        destination = cls.path(root_path, episode_id)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
        return destination

    @staticmethod
    def load(path: str | Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        missing = set(REQUIRED_FIELDS) - result.keys()
        if missing:
            raise ValueError(f"episode {path} is missing fields: {sorted(missing)}")
        return result

    @staticmethod
    def list(root: str | Path) -> list[Path]:
        return sorted(Path(root).glob("episode_*.npz"))

    @classmethod
    def next_episode_id(cls, root: str | Path) -> int:
        files = cls.list(root)
        if not files:
            return 0
        return max(int(path.stem.rsplit("_", 1)[1]) for path in files) + 1


def _split_files(
    files: Sequence[Path], split: str, validation_fraction: float, seed: int
) -> list[Path]:
    if split == "all":
        return list(files)
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train', 'validation', or 'all'")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    order = np.random.default_rng(seed).permutation(len(files))
    validation_count = int(round(len(files) * validation_fraction))
    validation_indices = set(order[:validation_count].tolist())
    return [path for index, path in enumerate(files)
            if (index in validation_indices) == (split == "validation")]


@dataclass(frozen=True)
class _Window:
    path: Path
    start: int
    length: int


class EpisodeSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Preloaded causal windows, always contained inside one episode."""

    def __init__(
        self,
        root: str | Path,
        *,
        sequence_length: int | None,
        split: str = "train",
        stride: int | None = None,
        validation_fraction: float = 0.2,
        seed: int = 0,
        use_clean_targets: bool = True,
    ) -> None:
        if sequence_length is not None and sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        self.sequence_length = sequence_length
        self.use_clean_targets = use_clean_targets
        files = _split_files(EpisodeArchive.list(root), split, validation_fraction, seed)
        self._episodes: dict[Path, dict[str, np.ndarray]] = {}
        self.windows: list[_Window] = []
        for path in files:
            episode = EpisodeArchive.load(path)
            self._episodes[path] = episode
            length = len(episode["depth"])
            if sequence_length is None:
                self.windows.append(_Window(path, 0, length))
            else:
                step = stride or sequence_length
                self.windows.extend(
                    _Window(path, start, min(sequence_length, length - start))
                    for start in range(0, length, step)
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        episode = self._episodes[window.path]
        start, stop = window.start, window.start + window.length
        padded_length = self.sequence_length or window.length
        result: dict[str, torch.Tensor] = {}
        for name in REQUIRED_FIELDS:
            source = episode[name][start:stop]
            padded = np.zeros((padded_length, *source.shape[1:]), dtype=source.dtype)
            padded[: window.length] = source
            result[name] = torch.from_numpy(padded)

        target_name = (
            "clean_depth"
            if self.use_clean_targets and "clean_depth" in episode
            else "depth"
        )
        target_source = episode[target_name][start:stop]
        target_depth = np.zeros(
            (padded_length, *target_source.shape[1:]), dtype=target_source.dtype
        )
        target_depth[: window.length] = target_source
        result["target_depth"] = torch.from_numpy(target_depth)
        valid = torch.zeros(padded_length, dtype=torch.bool)
        valid[: window.length] = True
        result["valid"] = valid
        result["episode_start"] = torch.zeros(padded_length, dtype=torch.bool)
        if window.start == 0:
            result["episode_start"][0] = True
        return result


def pad_episode_batch(
    samples: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Pad variable-length complete episodes for a batched causal forward."""
    if not samples:
        raise ValueError("cannot collate an empty episode batch")
    maximum_length = max(sample["valid"].shape[0] for sample in samples)
    result: dict[str, torch.Tensor] = {}
    for name in samples[0]:
        padded_values = []
        for sample in samples:
            value = sample[name]
            padding = maximum_length - value.shape[0]
            if padding:
                value = torch.cat(
                    (value, torch.zeros((padding, *value.shape[1:]), dtype=value.dtype)),
                    dim=0,
                )
            padded_values.append(value)
        result[name] = torch.stack(padded_values)
    return result


class DepthFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """Preload episode depth arrays and expose individual encoder frames."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        validation_fraction: float = 0.2,
        seed: int = 0,
        use_clean_targets: bool = True,
    ) -> None:
        self.use_clean_targets = use_clean_targets
        files = _split_files(EpisodeArchive.list(root), split, validation_fraction, seed)
        self._episodes: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self.frames: list[tuple[Path, int]] = []
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                depth = archive["depth"]
                target_name = (
                    "clean_depth"
                    if self.use_clean_targets and "clean_depth" in archive
                    else "depth"
                )
                target_depth = depth if target_name == "depth" else archive[target_name]
            self._episodes[path] = (depth, target_depth)
            length = len(depth)
            self.frames.extend((path, index) for index in range(length))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, frame_index = self.frames[index]
        depth, target_depth = self._episodes[path]
        return {
            "depth": torch.from_numpy(depth[frame_index]),
            "target_depth": torch.from_numpy(target_depth[frame_index]),
        }


def build_reconstruction_targets(
    depth: torch.Tensor,
    valid: torch.Tensor,
    offsets: Iterable[int] = (-5, -10, 5),
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Align target frame ``t + offset`` with each causal memory state ``m_t``."""

    if depth.ndim < 3 or valid.shape != depth.shape[:2]:
        raise ValueError("depth must be [B,T,...] and valid must be [B,T]")
    time = depth.shape[1]
    targets: dict[int, torch.Tensor] = {}
    masks: dict[int, torch.Tensor] = {}
    for offset in offsets:
        target = torch.zeros_like(depth)
        mask = torch.zeros_like(valid)
        if offset == 0:
            target.copy_(depth)
            mask.copy_(valid)
        elif abs(offset) < time:
            if offset > 0:
                target[:, : time - offset] = depth[:, offset:]
                mask[:, : time - offset] = valid[:, : time - offset] & valid[:, offset:]
            else:
                shift = -offset
                target[:, shift:] = depth[:, : time - shift]
                mask[:, shift:] = valid[:, shift:] & valid[:, : time - shift]
        targets[offset] = target
        masks[offset] = mask
    return targets, masks
