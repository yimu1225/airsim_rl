#!/usr/bin/env python3
"""Paired AOSA-style explanations for CL-VSSM-SAC and CL-SAC.

The script reads depth sequences and base states from a saved successful
CL-VSSM-SAC trajectory.  It does not launch AirSim and does not alter either
model.  Both actors receive exactly the same paired samples and exactly the
same spatiotemporal occlusion units.  Saved trajectories may come from the
legacy MambaLRP collector or from the online AOSA training-environment
collector.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aosa import (  # noqa: E402
    OcclusionConfig,
    OcclusionResult,
    SpatiotemporalOcclusionExplainer,
    describe_mask_plan,
)
from config import get_config  # noqa: E402
from eval.eval_common import resolve_checkpoint, set_agent_eval_mode  # noqa: E402


ACTION_KEYS = ("forward_velocity", "yaw_rate", "vertical_velocity")
ACTION_LABELS = (r"$v_x$", r"$\omega$", r"$v_z$")
MODEL_LABELS = ("VSSM-SAC", "SAC")
DEFAULT_MODEL_SEED = 25
DEFAULT_DPI = 300


matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 44,
        "font.weight": "bold",
        "axes.titlesize": 52,
        "axes.titleweight": "bold",
        "axes.labelsize": 44,
        "axes.labelweight": "bold",
        "xtick.labelsize": 44,
        "ytick.labelsize": 44,
        "legend.fontsize": 44,
    }
)


@dataclass(frozen=True)
class TrajectorySample:
    step: int
    base_state: np.ndarray
    depth: np.ndarray
    reference_physical_action: np.ndarray
    obstacle_proximity: float


@dataclass
class PairedRecord:
    sample: TrajectorySample
    vssm: OcclusionResult
    sac: OcclusionResult
    vssm_physical_action: np.ndarray
    sac_physical_action: np.ndarray


class ActorMeanAdapter:
    """Expose an agent's pre-tanh Actor mean as a batched black box."""

    def __init__(self, agent, *, kind: str) -> None:
        self.agent = agent
        self.kind = str(kind)
        if self.kind not in {"vssm", "sac"}:
            raise ValueError("kind must be 'vssm' or 'sac'")

    def __call__(
        self, base_state: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        base = torch.as_tensor(
            base_state, dtype=torch.float32, device=self.agent.device
        )
        depth_tensor = torch.as_tensor(
            depth, dtype=torch.float32, device=self.agent.device
        )
        with torch.inference_mode():
            if self.kind == "vssm":
                state = self.agent._encode_state(
                    base, depth_tensor, self.agent.actor_encoder
                )
                mean, _ = self.agent.actor.distribution_params(state)
            else:
                state = self.agent._concat_state(
                    base, depth_tensor, self.agent.actor_encoder
                )
                mean, _ = self.agent.actor(
                    state, compute_pi=False, compute_log_pi=False
                )
        return mean.detach().cpu().numpy().astype(np.float32)

    def physical_action(self, mean: np.ndarray) -> np.ndarray:
        normalized = np.tanh(np.asarray(mean, dtype=np.float32))
        scale = self.agent.action_scale.detach().cpu().numpy()
        bias = self.agent.action_bias.detach().cpu().numpy()
        return (scale * normalized + bias).astype(np.float32)


def _prepare_depth(depth: np.ndarray) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"Expected depth (T,H,W), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Reference depth contains non-finite values")
    return value


def _latest_successful_reference_run() -> Path:
    root = (
        REPO_ROOT
        / "results"
        / "explainability"
        / "mambalrp"
        / "test_scene"
        / "CL-VSSM-SAC"
    )
    candidates: list[Path] = []
    for metadata_path in root.glob("seed*/episode*/run_*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("termination") == "success":
            candidates.append(metadata_path.parent)
    if not candidates:
        raise FileNotFoundError(
            "No successful MambaLRP trajectory was found; pass "
            "--reference_run explicitly."
        )
    return sorted(candidates)[-1]


def _load_reference_samples(
    reference_run: Path,
    requested_steps: Sequence[int] | None,
) -> tuple[list[TrajectorySample], dict]:
    metadata_path = reference_run / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing reference metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("termination") != "success":
        raise ValueError(
            "The reference trajectory must be successful; got "
            f"{metadata.get('termination')!r}"
        )
    if metadata.get("algorithm") != "CL-VSSM-SAC":
        raise ValueError(
            "Expected a CL-VSSM-SAC reference, got "
            f"{metadata.get('algorithm')!r}"
        )
    completed = [
        int(value) for value in metadata.get("capture_steps_completed", [])
    ]
    steps = (
        sorted(set(map(int, requested_steps)))
        if requested_steps is not None
        else completed
    )
    if not steps:
        raise ValueError("Reference trajectory contains no captured samples")
    missing = sorted(set(steps) - set(completed))
    if missing:
        raise ValueError(
            "Requested steps are absent from the reference run: "
            + ", ".join(map(str, missing))
        )

    sample_file_pattern = str(
        metadata.get(
            "sample_file_pattern", "step_{step:04d}_mambalrp.npz"
        )
    )
    reference_action_key = str(
        metadata.get(
            "reference_action_key", "original_physical_action"
        )
    )
    samples: list[TrajectorySample] = []
    for step in steps:
        try:
            relative_path = sample_file_pattern.format(step=step)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "Invalid sample_file_pattern in reference metadata: "
                f"{sample_file_pattern!r}"
            ) from exc
        path = reference_run / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference sample: {path}")
        with np.load(path, allow_pickle=False) as data:
            if reference_action_key not in data:
                raise KeyError(
                    f"Reference sample {path} does not contain action key "
                    f"{reference_action_key!r}"
                )
            samples.append(
                TrajectorySample(
                    step=int(data["step"]),
                    base_state=np.asarray(
                        data["base_state"], dtype=np.float32
                    ).copy(),
                    depth=_prepare_depth(data["depth"]).copy(),
                    reference_physical_action=np.asarray(
                        data[reference_action_key], dtype=np.float32
                    ).copy(),
                    obstacle_proximity=float(data["obstacle_proximity"]),
                )
            )
    return samples, metadata


def _action_space(args) -> spaces.Box:
    return spaces.Box(
        low=np.array(
            [
                args.min_forward_speed,
                -args.max_yaw_rate,
                -args.max_vertical_speed,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                args.max_forward_speed,
                args.max_yaw_rate,
                args.max_vertical_speed,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )


def _load_actor(agent, checkpoint_path: str) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    actor_encoder = checkpoint.get("actor_encoder")
    actor = checkpoint.get("actor")
    if not isinstance(actor_encoder, dict) or not isinstance(actor, dict):
        raise ValueError(
            "Checkpoint must contain actor_encoder and actor state dictionaries"
        )
    agent.actor_encoder.load_state_dict(actor_encoder, strict=True)
    agent.actor.load_state_dict(actor, strict=True)
    del checkpoint


def _make_agents(
    common_args,
    first: TrajectorySample,
    *,
    model_seed: int,
    device: torch.device,
    vssm_checkpoint: str,
    sac_checkpoint: str,
):
    # Delay CUDA/Triton-dependent imports so --dry_run remains usable on a
    # machine where no CUDA driver is visible.
    from algorithm.SAC.agent import SACAgent
    from algorithm.SB_PER_VSSM_SAC.agent import SB_PERVSSM_SACAgent
    from algorithm.config_loader import apply_algorithm_params

    frames, height, width = map(int, first.depth.shape)
    action_space = _action_space(common_args)

    vssm_args = copy.deepcopy(common_args)
    vssm_args.algorithm_name = "CL-VSSM-SAC"
    vssm_args.seed = model_seed
    vssm_args.n_frames = frames
    apply_algorithm_params(vssm_args, "CL-VSSM-SAC")
    vssm = SB_PERVSSM_SACAgent(
        first.base_state.size,
        (1, height, width),
        action_space,
        vssm_args,
        device=device,
        seed=model_seed,
    )
    _load_actor(vssm, vssm_checkpoint)
    set_agent_eval_mode(vssm)

    sac_args = copy.deepcopy(common_args)
    sac_args.algorithm_name = "CL-SAC"
    sac_args.seed = model_seed
    sac_args.n_frames = frames
    apply_algorithm_params(sac_args, "CL-SAC")
    sac = SACAgent(
        first.base_state.size,
        (frames, height, width),
        action_space,
        sac_args,
        device=device,
        seed=model_seed,
    )
    _load_actor(sac, sac_checkpoint)
    set_agent_eval_mode(sac)
    return vssm, sac


def _shared_scale(first: np.ndarray, second: np.ndarray) -> float:
    values = np.concatenate(
        [np.abs(first).reshape(-1), np.abs(second).reshape(-1)]
    )
    scale = float(np.percentile(values, 99.0)) if values.size else 0.0
    return max(scale, float(np.finfo(np.float32).eps))


def _show_depth(axis, depth: np.ndarray) -> None:
    axis.imshow(
        depth,
        cmap="gray",
        vmin=0.0,
        vmax=255.0,
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _depth_sha256(depth: np.ndarray) -> str:
    """Hash the exact float32 array saved in the paired result."""

    value = np.ascontiguousarray(depth, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _render_original_frames(
    sample: TrajectorySample, output_path: Path, *, dpi: int
) -> None:
    """Render only the unmodified depth frames loaded from the source NPZ."""

    frames = sample.depth.shape[0]
    figure, axes = plt.subplots(
        1,
        frames,
        figsize=(2.65 * frames, 2.75),
        squeeze=False,
        constrained_layout=True,
    )
    for frame in range(frames):
        lag = frames - frame - 1
        _show_depth(axes[0, frame], sample.depth[frame])
        axes[0, frame].set_title("t" if lag == 0 else f"t-{lag}")
    axes[0, 0].set_ylabel("Saved depth")
    figure.suptitle(f"Step {sample.step}: original trajectory frames")
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _show_absolute_overlay(
    axis,
    depth: np.ndarray,
    influence: np.ndarray,
    *,
    scale: float,
    max_alpha: float,
) -> None:
    _show_depth(axis, depth)
    normalized = np.clip(influence / scale, 0.0, 1.0)
    alpha = float(max_alpha) * np.power(normalized, 0.65)
    alpha[normalized < 0.03] = 0.0
    axis.imshow(
        normalized,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        alpha=alpha,
        interpolation="bilinear",
    )


def _show_signed_overlay(
    axis,
    depth: np.ndarray,
    influence: np.ndarray,
    *,
    scale: float,
    max_alpha: float,
) -> None:
    _show_depth(axis, depth)
    normalized = np.clip(influence / scale, -1.0, 1.0)
    strength = np.abs(normalized)
    alpha = float(max_alpha) * np.power(strength, 0.65)
    alpha[strength < 0.03] = 0.0
    axis.imshow(
        normalized,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        alpha=alpha,
        interpolation="bilinear",
    )


def _render_action_map(
    record: PairedRecord,
    action_index: int,
    output_path: Path,
    *,
    signed: bool,
    overlay_alpha: float,
    dpi: int,
) -> None:
    depth = record.sample.depth
    first = (
        record.vssm.signed_influence[action_index]
        if signed
        else record.vssm.absolute_influence[action_index]
    )
    second = (
        record.sac.signed_influence[action_index]
        if signed
        else record.sac.absolute_influence[action_index]
    )
    scale = _shared_scale(first, second)
    frames = depth.shape[0]
    figure, axes = plt.subplots(
        3,
        frames,
        figsize=(2.65 * frames, 7.5),
        squeeze=False,
        constrained_layout=True,
    )
    for frame in range(frames):
        lag = frames - frame - 1
        _show_depth(axes[0, frame], depth[frame])
        axes[0, frame].set_title("t" if lag == 0 else f"t-{lag}")
        if signed:
            _show_signed_overlay(
                axes[1, frame],
                depth[frame],
                first[frame],
                scale=scale,
                max_alpha=overlay_alpha,
            )
            _show_signed_overlay(
                axes[2, frame],
                depth[frame],
                second[frame],
                scale=scale,
                max_alpha=overlay_alpha,
            )
        else:
            _show_absolute_overlay(
                axes[1, frame],
                depth[frame],
                first[frame],
                scale=scale,
                max_alpha=overlay_alpha,
            )
            _show_absolute_overlay(
                axes[2, frame],
                depth[frame],
                second[frame],
                scale=scale,
                max_alpha=overlay_alpha,
            )
    axes[0, 0].set_ylabel("Original\n(saved depth)")
    axes[1, 0].set_ylabel(MODEL_LABELS[0])
    axes[2, 0].set_ylabel(MODEL_LABELS[1])
    figure.suptitle(
        f"Step {record.sample.step}: {ACTION_LABELS[action_index]}\n"
        + (
            "Signed pre-tanh action effect"
            if signed
            else "Absolute pre-tanh action influence"
        )
    )
    scalar = ScalarMappable(
        norm=Normalize(vmin=-1.0 if signed else 0.0, vmax=1.0),
        cmap="coolwarm" if signed else "inferno",
    )
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        location="bottom",
        shrink=0.58,
        pad=0.03,
        aspect=40,
    )
    colorbar.set_label(
        "Normalized signed effect" if signed else "Normalized influence"
    )
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _render_all_actions_signed(
    record: PairedRecord,
    output_path: Path,
    *,
    overlay_alpha: float,
    dpi: int,
) -> None:
    """Render both models and all signed action effects in one figure."""

    depth = record.sample.depth
    frames = depth.shape[0]
    action_scales = [
        _shared_scale(
            record.vssm.signed_influence[action_index],
            record.sac.signed_influence[action_index],
        )
        for action_index in range(len(ACTION_KEYS))
    ]
    row_specs = [
        (MODEL_LABELS[0], action_index, record.vssm.signed_influence)
        for action_index in range(len(ACTION_KEYS))
    ] + [
        (MODEL_LABELS[1], action_index, record.sac.signed_influence)
        for action_index in range(len(ACTION_KEYS))
    ]
    figure, axes = plt.subplots(
        1 + len(row_specs),
        frames,
        figsize=(2.65 * frames, 16.2),
        squeeze=False,
        constrained_layout=True,
    )
    for frame in range(frames):
        lag = frames - frame - 1
        _show_depth(axes[0, frame], depth[frame])
        axes[0, frame].set_title("t" if lag == 0 else f"t-{lag}")
    axes[0, 0].set_ylabel("Original\n(saved depth)")

    for row, (model_label, action_index, influences) in enumerate(
        row_specs, start=1
    ):
        for frame in range(frames):
            _show_signed_overlay(
                axes[row, frame],
                depth[frame],
                influences[action_index, frame],
                scale=action_scales[action_index],
                max_alpha=overlay_alpha,
            )
        axes[row, 0].set_ylabel(
            f"{model_label}\n{ACTION_LABELS[action_index]}"
        )

    figure.suptitle(
        f"Step {record.sample.step}: signed effects for all actions\n"
        "Per-action normalization; each action shares one scale across both models"
    )
    scalar = ScalarMappable(
        norm=Normalize(vmin=-1.0, vmax=1.0), cmap="coolwarm"
    )
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        location="bottom",
        shrink=0.58,
        pad=0.02,
        aspect=40,
    )
    colorbar.set_label("Normalized signed effect")
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _select_summary_records(
    records: Sequence[PairedRecord], max_steps: int
) -> list[PairedRecord]:
    """Select evenly spaced trajectory records for a readable summary."""

    if len(records) <= max_steps:
        return list(records)
    indices = np.linspace(0, len(records) - 1, max_steps)
    indices = np.rint(indices).astype(int)
    return [records[index] for index in np.unique(indices)]


def _trajectory_action_scales(
    records: Sequence[PairedRecord],
) -> list[float]:
    """Return per-action scales shared by both models and every shown step."""

    action_scales: list[float] = []
    for action_index in range(len(ACTION_KEYS)):
        values = np.concatenate(
            [
                result.absolute_influence[action_index].reshape(-1)
                for record in records
                for result in (record.vssm, record.sac)
            ]
        )
        scale = float(np.percentile(values, 99.0))
        action_scales.append(
            max(scale, float(np.finfo(np.float32).eps))
        )
    return action_scales


def _render_paired_trajectory_heatmap(
    records: Sequence[PairedRecord],
    output_path: Path,
    *,
    action_scales: Sequence[float],
    dpi: int,
) -> None:
    """Tile paired multi-step four-frame blocks without a figure title."""

    if not records:
        raise ValueError("At least one record is required for a summary")

    frames = records[0].sample.depth.shape[0]
    row_specs = [
        (MODEL_LABELS[0], action_index, "vssm")
        for action_index in range(len(ACTION_KEYS))
    ] + [
        (MODEL_LABELS[1], action_index, "sac")
        for action_index in range(len(ACTION_KEYS))
    ]
    block_rows = 1 + len(row_specs)
    block_columns = frames
    grid_rows = block_rows
    grid_columns = len(records) * block_columns
    figure, axes = plt.subplots(
        grid_rows,
        grid_columns,
        figsize=(3.5 * grid_columns, 3.5 * grid_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.035, "hspace": 0.035},
    )
    for axis in axes.ravel():
        axis.set_visible(False)

    time_labels = [
        "t" if frames - frame - 1 == 0 else f"t-{frames-frame-1}"
        for frame in range(frames)
    ]
    for index, record in enumerate(records):
        row_start = 0
        column_start = index * block_columns
        for frame in range(frames):
            axis = axes[row_start, column_start + frame]
            axis.set_visible(True)
            _show_depth(axis, record.sample.depth[frame])
            axis.set_title(
                time_labels[frame], fontsize=40, fontweight="bold", pad=2
            )
        if index == 0:
            axes[row_start, column_start].set_ylabel(
                "Depth image",
                fontsize=40,
                fontweight="bold",
                rotation=0,
                ha="right",
                va="center",
                labelpad=10,
            )

        for row_offset, (model_label, action_index, result_name) in enumerate(
            row_specs, start=1
        ):
            row = row_start + row_offset
            scale = action_scales[action_index]
            result = getattr(record, result_name)
            for frame in range(frames):
                axis = axes[row, column_start + frame]
                axis.set_visible(True)
                normalized = np.clip(
                    result.absolute_influence[action_index, frame] / scale,
                    0.0,
                    1.0,
                )
                axis.imshow(
                    normalized,
                    cmap="turbo",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="bilinear",
                )
                axis.set_xticks([])
                axis.set_yticks([])
            if index == 0:
                axes[row, column_start].set_ylabel(
                    f"{model_label}: {ACTION_LABELS[action_index]}",
                    fontsize=36,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                )

    figure.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.01,
        top=0.93,
        wspace=0.035,
        hspace=0.035,
    )
    for index, record in enumerate(records):
        column_start = index * block_columns
        left = axes[0, column_start].get_position().x0
        right = axes[0, column_start + block_columns - 1].get_position().x1
        figure.text(
            (left + right) / 2.0,
            0.985,
            f"step={record.sample.step}",
            ha="center",
            va="top",
            fontsize=40,
            fontweight="bold",
        )
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _render_frame_deletion(
    record: PairedRecord, output_path: Path, *, dpi: int
) -> None:
    frames = record.sample.depth.shape[0]
    labels = [
        "t" if frames - index - 1 == 0 else f"t-{frames-index-1}"
        for index in range(frames)
    ]
    x = np.arange(frames)
    width = 0.36
    figure, axes = plt.subplots(
        1, len(ACTION_LABELS), figsize=(12.0, 3.8), constrained_layout=True
    )
    for action_index, axis in enumerate(np.atleast_1d(axes)):
        axis.bar(
            x - width / 2,
            np.abs(record.vssm.frame_deletion_delta[action_index]),
            width,
            label=MODEL_LABELS[0],
        )
        axis.bar(
            x + width / 2,
            np.abs(record.sac.frame_deletion_delta[action_index]),
            width,
            label=MODEL_LABELS[1],
        )
        axis.set_title(ACTION_LABELS[action_index])
        axis.set_xticks(x, labels)
        axis.set_ylabel("|change in pre-tanh mean|")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle(f"Step {record.sample.step}: whole-frame deletion effect")
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _save_record(record: PairedRecord, output_dir: Path) -> None:
    sample = record.sample
    np.savez_compressed(
        output_dir / f"step_{sample.step:04d}_paired_aosa.npz",
        step=np.int32(sample.step),
        base_state=sample.base_state.astype(np.float32),
        depth=sample.depth.astype(np.float32),
        reference_physical_action=sample.reference_physical_action.astype(
            np.float32
        ),
        obstacle_proximity=np.float32(sample.obstacle_proximity),
        vssm_pre_tanh_mean=record.vssm.original_output.astype(np.float32),
        sac_pre_tanh_mean=record.sac.original_output.astype(np.float32),
        vssm_physical_action=record.vssm_physical_action.astype(np.float32),
        sac_physical_action=record.sac_physical_action.astype(np.float32),
        vssm_absolute_influence=record.vssm.absolute_influence.astype(
            np.float32
        ),
        sac_absolute_influence=record.sac.absolute_influence.astype(
            np.float32
        ),
        vssm_signed_influence=record.vssm.signed_influence.astype(np.float32),
        sac_signed_influence=record.sac.signed_influence.astype(np.float32),
        vssm_frame_deletion_delta=record.vssm.frame_deletion_delta.astype(
            np.float32
        ),
        sac_frame_deletion_delta=record.sac.frame_deletion_delta.astype(
            np.float32
        ),
        occlusion_coverage=record.vssm.coverage.astype(np.float32),
        forward_flow=record.vssm.forward_flow.astype(np.float32),
        backward_flow=record.vssm.backward_flow.astype(np.float32),
    )


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _default_output_dir(model_seed: int, reference_run: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    return (
        REPO_ROOT
        / "results"
        / "explainability"
        / "aosa"
        / "paired_reference"
        / f"seed{model_seed}"
        / reference_run.name
        / stamp
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. VSSM inference normally "
            "requires the AirSim CUDA environment."
        )
    return device


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired AOSA-style spatiotemporal policy explanations "
            "from a saved successful CL-VSSM-SAC trajectory."
        )
    )
    parser.add_argument("--model_seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--reference_run", type=str, default=None)
    parser.add_argument("--capture_steps", type=int, nargs="+", default=None)
    parser.add_argument("--vssm_checkpoint", type=str, default=None)
    parser.add_argument("--sac_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--window_height", type=int, default=16)
    parser.add_argument("--window_width", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--temporal_radius", type=int, default=1)
    parser.add_argument("--reference_depth", type=float, default=255.0)
    parser.add_argument("--occlusion_batch_size", type=int, default=32)
    parser.add_argument("--motion", choices=("flow", "fixed"), default="flow")
    parser.add_argument(
        "--max_masks",
        type=int,
        default=None,
        help="Debug-only cap on occlusion candidates per sample.",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.82)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--skip_signed", action="store_true", help="Do not render signed maps."
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help=(
            "Save raw arrays for every selected step but render only one "
            "trajectory-level absolute heatmap figure."
        ),
    )
    parser.add_argument(
        "--summary_max_steps",
        type=int,
        default=10,
        help="Maximum evenly spaced step blocks in each trajectory summary.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate the saved trajectory and mask plan without loading models.",
    )
    script_args, remaining = parser.parse_known_args(argv)
    if script_args.capture_steps is not None:
        script_args.capture_steps = sorted(set(script_args.capture_steps))
        if script_args.capture_steps[0] < 0:
            parser.error("--capture_steps must be non-negative")
    if script_args.max_masks is not None and script_args.max_masks <= 0:
        parser.error("--max_masks must be positive")
    if script_args.dpi <= 0:
        parser.error("--dpi must be positive")
    if script_args.summary_max_steps <= 0:
        parser.error("--summary_max_steps must be positive")
    if not 0.0 <= script_args.overlay_alpha <= 1.0:
        parser.error("--overlay_alpha must be in [0,1]")
    common_args = get_config(remaining)
    return script_args, common_args


def run(script_args, common_args) -> Path | None:
    reference_run = (
        Path(script_args.reference_run).resolve()
        if script_args.reference_run
        else _latest_successful_reference_run()
    )
    samples, reference_metadata = _load_reference_samples(
        reference_run, script_args.capture_steps
    )
    config = OcclusionConfig(
        window_height=int(script_args.window_height),
        window_width=int(script_args.window_width),
        stride=int(script_args.stride),
        temporal_radius=int(script_args.temporal_radius),
        reference_depth=float(script_args.reference_depth),
        batch_size=int(script_args.occlusion_batch_size),
        motion=str(script_args.motion),
    )
    plans = [describe_mask_plan(sample.depth, config) for sample in samples]
    print(f"[AOSA] Reusing successful trajectory: {reference_run}")
    print(
        "[AOSA] Selected steps: "
        + ", ".join(str(sample.step) for sample in samples)
    )
    print("[AOSA] Mask plan: " + json.dumps(plans[0], ensure_ascii=False))
    if script_args.dry_run:
        print("[AOSA] Dry run completed; no model was loaded and no output written.")
        return None

    model_seed = int(script_args.model_seed)
    _configure_reproducibility(model_seed)
    device = _resolve_device(script_args.device)
    if device.type != "cuda":
        print(
            "[AOSA] Warning: VSSM/Mamba kernels may not support CPU execution; "
            "use --device cuda in the AirSim environment."
        )
    vssm_checkpoint = resolve_checkpoint(
        script_args.vssm_checkpoint,
        str(
            REPO_ROOT
            / "models"
            / "CL-VSSM-SAC"
            / f"seed{model_seed}"
            / "test.pth"
        ),
    )
    sac_checkpoint = resolve_checkpoint(
        script_args.sac_checkpoint,
        str(
            REPO_ROOT
            / "models"
            / "CL-SAC"
            / f"seed{model_seed}"
            / "async_final.pth"
        ),
    )
    output_dir = (
        Path(script_args.output_dir).resolve()
        if script_args.output_dir
        else _default_output_dir(model_seed, reference_run)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    vssm, sac = _make_agents(
        common_args,
        samples[0],
        model_seed=model_seed,
        device=device,
        vssm_checkpoint=vssm_checkpoint,
        sac_checkpoint=sac_checkpoint,
    )
    vssm_adapter = ActorMeanAdapter(vssm, kind="vssm")
    sac_adapter = ActorMeanAdapter(sac, kind="sac")
    print(f"[AOSA] Loaded CL-VSSM-SAC Actor: {vssm_checkpoint}")
    print(f"[AOSA] Loaded CL-SAC Actor: {sac_checkpoint}")

    records: list[PairedRecord] = []
    for sample in samples:
        print(f"[AOSA] Explaining paired step {sample.step}")
        vssm_result = SpatiotemporalOcclusionExplainer(
            vssm_adapter, config
        ).explain(
            sample.base_state,
            sample.depth,
            max_masks=script_args.max_masks,
        )
        sac_result = SpatiotemporalOcclusionExplainer(
            sac_adapter, config
        ).explain(
            sample.base_state,
            sample.depth,
            max_masks=script_args.max_masks,
        )
        if not np.array_equal(vssm_result.coverage, sac_result.coverage):
            raise RuntimeError(
                "Paired models did not use identical occlusion coverage"
            )
        record = PairedRecord(
            sample=sample,
            vssm=vssm_result,
            sac=sac_result,
            vssm_physical_action=vssm_adapter.physical_action(
                vssm_result.original_output
            ),
            sac_physical_action=sac_adapter.physical_action(
                sac_result.original_output
            ),
        )
        records.append(record)
        _save_record(record, output_dir)
        if script_args.summary_only:
            continue
        _render_original_frames(
            sample,
            output_dir / f"step_{sample.step:04d}_original_frames.png",
            dpi=int(script_args.dpi),
        )
        for action_index, action_key in enumerate(ACTION_KEYS):
            _render_action_map(
                record,
                action_index,
                output_dir
                / f"step_{sample.step:04d}_{action_key}_absolute.png",
                signed=False,
                overlay_alpha=float(script_args.overlay_alpha),
                dpi=int(script_args.dpi),
            )
            if not script_args.skip_signed:
                _render_action_map(
                    record,
                    action_index,
                    output_dir
                    / f"step_{sample.step:04d}_{action_key}_signed.png",
                    signed=True,
                    overlay_alpha=float(script_args.overlay_alpha),
                    dpi=int(script_args.dpi),
                )
        if not script_args.skip_signed:
            _render_all_actions_signed(
                record,
                output_dir
                / f"step_{sample.step:04d}_all_actions_signed.png",
                overlay_alpha=float(script_args.overlay_alpha),
                dpi=int(script_args.dpi),
            )
        _render_frame_deletion(
            record,
            output_dir / f"step_{sample.step:04d}_frame_deletion.png",
            dpi=int(script_args.dpi),
        )

    summary_records = _select_summary_records(
        records, int(script_args.summary_max_steps)
    )
    if len(records) > 1 or script_args.summary_only:
        action_scales = _trajectory_action_scales(summary_records)
        _render_paired_trajectory_heatmap(
            summary_records,
            output_dir / "trajectory_paired_all_steps_absolute_heatmap.png",
            action_scales=action_scales,
            dpi=int(script_args.dpi),
        )

    metadata = {
        "method": "AOSA-style flow-guided spatiotemporal occlusion",
        "implementation_variant": (
            "motion-tracked rectangular 3-D occlusion tubes; continuous "
            "pre-tanh Actor means replace classification scores"
        ),
        "paired_models": list(MODEL_LABELS),
        "paired_inputs": True,
        "paired_input_fields": ["base_state", "depth"],
        "input_provenance": {
            str(sample.step): {
                "source_npz": str(
                    reference_run
                    / str(
                        reference_metadata.get(
                            "sample_file_pattern",
                            "step_{step:04d}_mambalrp.npz",
                        )
                    ).format(step=sample.step)
                ),
                "depth_sha256": _depth_sha256(sample.depth),
                "depth_shape": list(map(int, sample.depth.shape)),
                "depth_transform_for_display": "none; grayscale [0,255]",
            }
            for sample in samples
        },
        "model_seed": model_seed,
        "device": str(device),
        "reference_run": str(reference_run),
        "reference_source_type": reference_metadata.get(
            "source_type", "legacy_mambalrp_trajectory"
        ),
        "reference_termination": reference_metadata.get("termination"),
        "reference_episode_seed": reference_metadata.get("episode_seed"),
        "reference_environment_seed": reference_metadata.get(
            "environment_seed"
        ),
        "capture_steps_completed": [record.sample.step for record in records],
        "trajectory_summary": {
            "rendered": bool(len(records) > 1 or script_args.summary_only),
            "summary_only": bool(script_args.summary_only),
            "maximum_step_blocks": int(script_args.summary_max_steps),
            "step_blocks_per_row": len(summary_records),
            "step_layout": "all selected steps in one horizontal row",
            "steps_shown": [
                record.sample.step for record in summary_records
            ],
            "frames_shown_per_step": ["t-3", "t-2", "t-1", "t"],
            "row_order": [
                "depth image",
                "CL-VSSM-SAC forward velocity",
                "CL-VSSM-SAC yaw rate",
                "CL-VSSM-SAC vertical velocity",
                "CL-SAC forward velocity",
                "CL-SAC yaw rate",
                "CL-SAC vertical velocity",
            ],
            "map": "absolute influence",
            "colormap": "turbo",
            "figure_title": None,
            "colorbar": False,
            "subplot_spacing": {
                "horizontal": 0.035,
                "vertical": 0.035,
            },
            "normalization": (
                "per action; shared across both models and all shown steps"
            ),
        },
        "vssm_checkpoint": os.path.abspath(vssm_checkpoint),
        "sac_checkpoint": os.path.abspath(sac_checkpoint),
        "policy_targets": [
            f"pre_tanh_mean:{action_key}" for action_key in ACTION_KEYS
        ],
        "action_labels": list(ACTION_LABELS),
        "occlusion": {
            "window": [config.window_height, config.window_width],
            "stride": config.stride,
            "temporal_radius": config.temporal_radius,
            "motion": config.motion,
            "reference_depth": config.reference_depth,
            "reference_semantics": "obstacle-free/far depth",
            "batch_size": config.batch_size,
            "max_masks_debug_cap": script_args.max_masks,
            "candidate_counts": {
                str(record.sample.step): record.vssm.candidate_count
                for record in records
            },
        },
        "display": {
            "absolute_colormap": "inferno",
            "signed_colormap": "coolwarm",
            "zero_influence_transparent": True,
            "scale": (
                "shared 99th absolute percentile across both models and all "
                "four frames, separately for each action and figure"
            ),
            "all_actions_signed_layout": (
                "rows: original, three CL-VSSM-SAC actions, three CL-SAC "
                "actions; each action is normalized separately with a scale "
                "shared by both models and all four frames"
            ),
        },
        "frame_deletion_delta": {
            str(record.sample.step): {
                MODEL_LABELS[0]: record.vssm.frame_deletion_delta.tolist(),
                MODEL_LABELS[1]: record.sac.frame_deletion_delta.tolist(),
            }
            for record in records
        },
        "references": {
            "aosa": (
                "Uchiyama et al., Visually Explaining 3D-CNN Predictions "
                "for Video Classification With an Adaptive Occlusion "
                "Sensitivity Analysis, WACV 2023"
            ),
            "aosa_url": (
                "https://openaccess.thecvf.com/content/WACV2023/papers/"
                "Uchiyama_Visually_Explaining_3D-CNN_Predictions_for_"
                "Video_Classification_With_an_Adaptive_WACV_2023_paper.pdf"
            ),
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(f"[AOSA] Results saved to: {output_dir}")
    return output_dir


def main(argv=None) -> None:
    script_args, common_args = _parse_args(argv)
    run(script_args, common_args)


if __name__ == "__main__":
    main()
