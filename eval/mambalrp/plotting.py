"""Rendering and serialization for input-level MambaLRP results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .attribution import (
    ACTION_KEYS,
    CaptureRecord,
    _normalize_signed_maps,
    _prepare_depth,
)


def _render_overlay(
    axis,
    depth: np.ndarray,
    relevance: np.ndarray,
    *,
    alpha: float,
) -> None:
    axis.imshow(
        depth,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    axis.imshow(
        relevance,
        cmap="jet",
        norm=Normalize(vmin=-1.0, vmax=1.0),
        alpha=float(alpha),
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _render_four_frames(
    record: CaptureRecord,
    output_path: Path,
    *,
    alpha: float,
    dpi: int,
) -> None:
    depth = record.sample.depth
    normalized = _normalize_signed_maps(record.result.pixel_relevance)
    frames = depth.shape[0]
    figure, axes = plt.subplots(
        2,
        frames,
        figsize=(3.05 * frames, 5.7),
        squeeze=False,
        constrained_layout=True,
    )
    for frame in range(frames):
        lag = frames - frame - 1
        axes[0, frame].imshow(
            depth[frame],
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axes[0, frame].set_title("t" if lag == 0 else f"t-{lag}")
        axes[0, frame].set_xticks([])
        axes[0, frame].set_yticks([])
        _render_overlay(
            axes[1, frame],
            depth[frame],
            normalized[frame],
            alpha=alpha,
        )
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("MambaLRP")
    scalar = ScalarMappable(
        norm=Normalize(vmin=-1.0, vmax=1.0), cmap="jet"
    )
    scalar.set_array([])
    figure.colorbar(
        scalar,
        ax=axes[1, :],
        shrink=0.78,
        label="Signed relevance (normalized for display)",
    )
    resolution = "x".join(
        map(str, record.result.pixel_relevance.shape[-2:])
    )
    figure.suptitle(
        "CL-VSSM-SAC input-level MambaLRP "
        f"— step {record.sample.step} — raw input {resolution}"
    )
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _render_action_frames(
    record: CaptureRecord,
    output_path: Path,
    *,
    alpha: float,
    dpi: int,
) -> None:
    depth = record.sample.depth
    action_maps = record.result.action_pixel_relevance
    frames = depth.shape[0]
    figure, axes = plt.subplots(
        len(ACTION_LABELS) + 1,
        frames,
        figsize=(3.05 * frames, 2.65 * (len(ACTION_LABELS) + 1)),
        squeeze=False,
        constrained_layout=True,
    )
    for frame in range(frames):
        lag = frames - frame - 1
        axes[0, frame].imshow(
            depth[frame],
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axes[0, frame].set_title("t" if lag == 0 else f"t-{lag}")
        axes[0, frame].set_xticks([])
        axes[0, frame].set_yticks([])
    axes[0, 0].set_ylabel("Original")
    for action_index, label in enumerate(ACTION_LABELS):
        normalized = _normalize_signed_maps(action_maps[action_index])
        row = action_index + 1
        for frame in range(frames):
            _render_overlay(
                axes[row, frame],
                depth[frame],
                normalized[frame],
                alpha=alpha,
            )
        axes[row, 0].set_ylabel(label)
    scalar = ScalarMappable(
        norm=Normalize(vmin=-1.0, vmax=1.0), cmap="jet"
    )
    scalar.set_array([])
    figure.colorbar(
        scalar,
        ax=axes[1:, :],
        shrink=0.72,
        label="Signed relevance (normalized per target)",
    )
    figure.suptitle(
        f"CL-VSSM-SAC per-action MambaLRP — step {record.sample.step}"
    )
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _render_summary(
    records: Sequence[CaptureRecord],
    output_path: Path,
    *,
    alpha: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        2,
        len(records),
        figsize=(3.05 * len(records), 5.7),
        squeeze=False,
        constrained_layout=True,
    )
    for column, record in enumerate(records):
        depth = record.sample.depth[-1]
        relevance = _normalize_signed_maps(
            record.result.pixel_relevance
        )[-1]
        axes[0, column].imshow(
            depth,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axes[0, column].set_title(
            f"Step {record.sample.step}\n"
            f"proximity={record.sample.obstacle_proximity:.1f}"
        )
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        _render_overlay(
            axes[1, column], depth, relevance, alpha=alpha
        )
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("MambaLRP")
    scalar = ScalarMappable(
        norm=Normalize(vmin=-1.0, vmax=1.0), cmap="jet"
    )
    scalar.set_array([])
    figure.colorbar(
        scalar,
        ax=axes[1, :],
        shrink=0.78,
        label="Signed relevance (normalized per sample)",
    )
    figure.suptitle("CL-VSSM-SAC input-level MambaLRP summary")
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _save_record(record: CaptureRecord, output_dir: Path) -> None:
    sample = record.sample
    result = record.result
    np.savez_compressed(
        output_dir / f"step_{sample.step:04d}_mambalrp.npz",
        step=np.int32(sample.step),
        base_state=sample.base_state.astype(np.float32),
        depth=sample.depth.astype(np.float32),
        original_physical_action=sample.physical_action.astype(np.float32),
        obstacle_proximity=np.float32(sample.obstacle_proximity),
        pixel_relevance=result.pixel_relevance.astype(np.float32),
        patch_relevance=result.patch_relevance.astype(np.float32),
        display_relevance=result.pixel_relevance.astype(np.float32),
        action_pixel_relevance=(
            result.action_pixel_relevance.astype(np.float32)
        ),
        action_patch_relevance=(
            result.action_patch_relevance.astype(np.float32)
        ),
        action_display_relevance=(
            result.action_pixel_relevance.astype(np.float32)
        ),
    )


def _write_metadata(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            metadata, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
