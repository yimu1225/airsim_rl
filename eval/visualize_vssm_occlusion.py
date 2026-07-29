#!/usr/bin/env python3
"""Whole-policy heatmaps for seed-25 VSSM-SAC in the static test scene.

Implemented attribution methods:

* Occlusion Sensitivity (Zeiler & Fergus, ECCV 2014)
* Integrated Gradients (Sundararajan et al., ICML 2017)
* Integrated Gradients with SmoothGrad / NoiseTunnel
* MambaLRP core-rule adaptation (Jafari et al., NeurIPS 2024)

CaMeRL displays one map for a multidimensional policy but does not document its
scalar policy target.  This script makes that choice explicit.  Occlusion uses
the L2 norm of range-normalized action changes.  Gradient methods project the
dimensionless action vector onto the direction of the original deterministic
action, yielding one differentiable whole-policy target.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_config
from eval.eval_common import close_env, resolve_checkpoint, set_agent_eval_mode
from eval.eval_env import SceneEvalAirSimEnv


DEFAULT_MODEL_SEED = 25
DEFAULT_EPISODE_SEED = 25
DEFAULT_NUM_SAMPLES = 6
DEFAULT_MIN_SAMPLE_GAP = 10
ALL_METHODS = ("occlusion", "ig", "ig_smoothgrad", "mambalrp")
METHOD_LABELS = {
    "occlusion": "Occlusion sensitivity",
    "ig": "Integrated Gradients",
    "ig_smoothgrad": "IG + SmoothGrad",
    "mambalrp": "MambaLRP core adaptation",
}
ACTION_LABELS = ("Forward velocity", "Yaw rate", "Vertical velocity")
ACTION_UNITS = ("m/s", "rad/s", "m/s")


@dataclass
class AttributionResult:
    """One method's four-frame heatmaps and reproducibility diagnostics."""

    heatmaps: np.ndarray
    details: dict
    action_heatmaps: np.ndarray | None = None


@dataclass
class TrajectoryStep:
    step: int
    base_state: np.ndarray
    depth: np.ndarray
    action: np.ndarray
    obstacle_proximity: float


@dataclass
class CaptureRecord:
    step: int
    base_state: np.ndarray
    depth: np.ndarray
    action: np.ndarray
    obstacle_proximity: float
    attributions: dict[str, AttributionResult]


def sliding_window_starts(length: int, window: int, stride: int) -> list[int]:
    """Return sliding-window starts, including the trailing boundary."""

    length, window, stride = int(length), int(window), int(stride)
    if length <= 0:
        raise ValueError("length must be positive")
    if window <= 0 or window > length:
        raise ValueError(f"window must be in [1, {length}], got {window}")
    if stride <= 0:
        raise ValueError("stride must be positive")
    last = length - window
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def select_spaced_top_indices(
    scores: Sequence[float],
    *,
    count: int,
    min_gap: int,
) -> list[int]:
    """Select high-scoring observations with deterministic temporal spacing."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    count, min_gap = int(count), int(min_gap)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must contain finite values")
    if count <= 0:
        raise ValueError("count must be positive")
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")

    target_count = min(count, int(values.size))
    ranking = np.argsort(-values, kind="stable").tolist()
    for effective_gap in range(min_gap, -1, -1):
        selected: list[int] = []
        for index in ranking:
            if all(abs(index - previous) >= effective_gap for previous in selected):
                selected.append(index)
                if len(selected) == target_count:
                    return sorted(selected)
    raise AssertionError("gap-zero selection must satisfy the target count")


def _minimum_pair_gap(indices: Sequence[int]) -> int | None:
    ordered = sorted(int(value) for value in indices)
    if len(ordered) < 2:
        return None
    return min(right - left for left, right in zip(ordered, ordered[1:]))


def _prepare_depth(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(
            f"Expected depth (T,H,W) or (T,1,H,W), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("depth contains non-finite values")
    return array


def _obstacle_proximity(depth: np.ndarray) -> float:
    """Robust near-obstacle score; larger means closer visible geometry."""

    latest = _prepare_depth(depth)[-1]
    return float(255.0 - np.percentile(latest, 10.0))


class BatchedDeterministicActor:
    """NumPy batch adapter returning physical deterministic actions."""

    def __init__(self, agent) -> None:
        self.agent = agent

    def __call__(
        self,
        base_batch: np.ndarray,
        depth_batch: np.ndarray,
    ) -> np.ndarray:
        base = torch.as_tensor(
            base_batch, dtype=torch.float32, device=self.agent.device
        )
        depth = torch.as_tensor(
            depth_batch, dtype=torch.float32, device=self.agent.device
        )
        with torch.no_grad():
            state = self.agent._encode_state(
                base, depth, self.agent.actor_encoder
            )
            normalized = self.agent.actor(state, deterministic=True)
            action = self.agent.action_scale * normalized + self.agent.action_bias
        return action.detach().cpu().numpy().astype(np.float32, copy=False)


def _predict_checked(
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    base_batch: np.ndarray,
    depth_batch: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(predictor(base_batch, depth_batch), dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[0] != depth_batch.shape[0]:
        raise ValueError(
            "predictor must return (B,A), "
            f"got {actions.shape} for batch {depth_batch.shape[0]}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("predictor returned non-finite actions")
    return actions


def compute_occlusion_sensitivity(
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
    *,
    action_ranges: np.ndarray,
    window_size: int,
    stride: int,
    baseline_value: float,
    batch_size: int,
) -> AttributionResult:
    """Measure complete normalized action-vector changes under occlusion."""

    depth = _prepare_depth(depth_sequence)
    base = np.asarray(base_state, dtype=np.float32).reshape(-1)
    ranges = np.asarray(action_ranges, dtype=np.float32).reshape(-1)
    original = _predict_checked(predictor, base[None], depth[None])[0]
    if ranges.shape != original.shape or np.any(ranges <= 0):
        raise ValueError("action_ranges must match actions and be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    frames, height, width = depth.shape
    tops = sliding_window_starts(height, window_size, stride)
    lefts = sliding_window_starts(width, window_size, stride)
    windows = [
        (frame, top, left)
        for frame in range(frames)
        for top in tops
        for left in lefts
    ]
    sums = np.zeros(
        (frames, original.size, height, width), dtype=np.float64
    )
    policy_sums = np.zeros((frames, height, width), dtype=np.float64)
    coverage = np.zeros((frames, height, width), dtype=np.float64)

    for offset in range(0, len(windows), int(batch_size)):
        chunk = windows[offset : offset + int(batch_size)]
        perturbed = np.repeat(depth[None], len(chunk), axis=0)
        for index, (frame, top, left) in enumerate(chunk):
            perturbed[
                index,
                frame,
                top : top + window_size,
                left : left + window_size,
            ] = float(baseline_value)
        base_batch = np.repeat(base[None], len(chunk), axis=0)
        actions = _predict_checked(predictor, base_batch, perturbed)
        scores = np.abs(actions - original[None]) / ranges[None]
        for (frame, top, left), score in zip(chunk, scores):
            policy_score = float(np.linalg.norm(score))
            sums[
                frame,
                :,
                top : top + window_size,
                left : left + window_size,
            ] += score[:, None, None]
            policy_sums[
                frame,
                top : top + window_size,
                left : left + window_size,
            ] += policy_score
            coverage[
                frame,
                top : top + window_size,
                left : left + window_size,
            ] += 1.0

    action_maps = np.zeros_like(sums, dtype=np.float32)
    np.divide(
        sums,
        coverage[:, None],
        out=action_maps,
        where=coverage[:, None] > 0,
    )
    policy_maps = np.zeros_like(policy_sums, dtype=np.float32)
    np.divide(
        policy_sums,
        coverage,
        out=policy_maps,
        where=coverage > 0,
    )
    return AttributionResult(
        heatmaps=policy_maps,
        action_heatmaps=action_maps,
        details={
            "definition": (
                "Per-window L2 norm of absolute action-vector changes after "
                "normalization by each physical action range, followed by "
                "overlap averaging"
            ),
            "original_action": original.tolist(),
            "coverage_min": float(coverage.min()),
            "coverage_max": float(coverage.max()),
        },
    )


def _normalized_action(agent, base: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    state = agent._encode_state(base, depth, agent.actor_encoder)
    return agent.actor(state, deterministic=True)


def _policy_direction(agent, base: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        action = _normalized_action(agent, base, depth)
    norm = torch.linalg.vector_norm(action, dim=1, keepdim=True)
    fallback = torch.full_like(action, 1.0 / np.sqrt(action.shape[1]))
    return torch.where(norm > 1e-8, action / norm.clamp_min(1e-8), fallback)


class WholePolicyTarget(nn.Module):
    """Differentiable scalar projection onto the original policy direction."""

    def __init__(self, agent, base_state: torch.Tensor, direction: torch.Tensor):
        super().__init__()
        self.agent = agent
        self.base_state = base_state
        self.direction = direction

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        base = self.base_state.expand(depth.shape[0], -1)
        direction = self.direction.expand(depth.shape[0], -1)
        action = _normalized_action(self.agent, base, depth)
        return (action * direction).sum(dim=1)


def _integrated_gradients_once(
    model: WholePolicyTarget,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    *,
    n_steps: int,
    internal_batch_size: int,
) -> tuple[torch.Tensor, float]:
    """Gauss-Legendre Integrated Gradients without an external dependency."""

    if n_steps <= 0 or internal_batch_size <= 0:
        raise ValueError("IG steps and internal batch size must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(int(n_steps))
    alphas = torch.as_tensor(
        (nodes + 1.0) / 2.0, dtype=inputs.dtype, device=inputs.device
    )
    quadrature = torch.as_tensor(
        weights / 2.0, dtype=inputs.dtype, device=inputs.device
    )
    difference = inputs - baselines
    total_gradient = torch.zeros_like(inputs)

    for offset in range(0, int(n_steps), int(internal_batch_size)):
        alpha = alphas[offset : offset + internal_batch_size]
        weight = quadrature[offset : offset + internal_batch_size]
        scaled = (
            baselines
            + alpha.view(-1, 1, 1, 1) * difference
        ).detach().requires_grad_(True)
        scores = model(scaled)
        gradients = torch.autograd.grad(scores.sum(), scaled)[0]
        total_gradient += (
            gradients * weight.view(-1, 1, 1, 1)
        ).sum(dim=0, keepdim=True)

    attribution = difference * total_gradient
    with torch.no_grad():
        output_delta = model(inputs) - model(baselines)
        convergence_delta = output_delta - attribution.flatten(1).sum(dim=1)
    return attribution, float(convergence_delta.abs().mean().item())


def compute_integrated_gradients(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
    *,
    baseline_value: float,
    n_steps: int,
    internal_batch_size: int,
    smooth_samples: int = 0,
    noise_stdev: float = 5.0,
    noise_seed: int = 25,
) -> AttributionResult:
    """Compute IG, optionally averaged over a SmoothGrad NoiseTunnel."""

    base = torch.as_tensor(
        base_state, dtype=torch.float32, device=agent.device
    ).view(1, -1)
    depth = torch.as_tensor(
        _prepare_depth(depth_sequence),
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)
    baseline = torch.full_like(depth, float(baseline_value))
    direction = _policy_direction(agent, base, depth)
    target = WholePolicyTarget(agent, base, direction)

    sample_count = max(1, int(smooth_samples))
    generator = torch.Generator(device=agent.device)
    generator.manual_seed(int(noise_seed))
    attributions: list[torch.Tensor] = []
    deltas: list[float] = []
    for _ in range(sample_count):
        if smooth_samples > 0:
            noise = torch.randn(
                depth.shape,
                dtype=depth.dtype,
                device=depth.device,
                generator=generator,
            ) * float(noise_stdev)
            sample = (depth + noise).clamp(0.0, 255.0)
        else:
            sample = depth
        attribution, delta = _integrated_gradients_once(
            target,
            sample,
            baseline,
            n_steps=int(n_steps),
            internal_batch_size=int(internal_batch_size),
        )
        attributions.append(attribution.detach())
        deltas.append(delta)

    mean_attribution = torch.stack(attributions).mean(dim=0)
    heatmaps = mean_attribution.abs()[0].cpu().numpy().astype(np.float32)
    return AttributionResult(
        heatmaps=heatmaps,
        details={
            "definition": (
                "absolute Integrated Gradients for normalized action projected "
                "onto the original deterministic action direction"
            ),
            "policy_direction": direction[0].detach().cpu().tolist(),
            "n_steps": int(n_steps),
            "quadrature": "gauss_legendre",
            "noise_tunnel": smooth_samples > 0,
            "noise_tunnel_type": "smoothgrad" if smooth_samples > 0 else None,
            "noise_samples": int(smooth_samples),
            "noise_stdev_depth_units": (
                float(noise_stdev) if smooth_samples > 0 else None
            ),
            "mean_absolute_convergence_delta": float(np.mean(deltas)),
        },
    )


def _stabilize(value: torch.Tensor) -> torch.Tensor:
    return value + ((value == 0).to(value) + value.sign()) * 1e-6


def _mambalrp_silu(value: torch.Tensor) -> torch.Tensor:
    """SiLU forward with the relevance-conserving MambaLRP backward rule."""

    output = F.silu(value)
    return value * (output / _stabilize(value)).detach()


class MambaLRPMixer(nn.Module):
    """Forward-equivalent Mamba-1 mixer with MambaLRP propagation rules."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.source = source

    @staticmethod
    def _scan_branch(
        projected: torch.Tensor,
        *,
        conv1d: nn.Module,
        x_proj: nn.Module,
        dt_proj: nn.Module,
        A_log: torch.Tensor,
        D: torch.Tensor,
        d_state: int,
        d_inner: int,
    ) -> torch.Tensor:
        sequence_length = projected.shape[-1]
        values, gate = projected.chunk(2, dim=1)
        values = _mambalrp_silu(
            conv1d(values)[..., :sequence_length]
        ).transpose(1, 2)
        parameters = x_proj(values)
        dt_rank = int(dt_proj.weight.shape[1])
        delta, B, C = torch.split(
            parameters, [dt_rank, d_state, d_state], dim=-1
        )
        delta = F.softplus(dt_proj(delta))
        A = -torch.exp(A_log.float())

        discrete_A = torch.exp(
            torch.einsum("bld,dn->bldn", delta.float(), A)
        ).detach()
        discrete_B = torch.einsum(
            "bld,bln->bldn", delta.float(), B.float()
        ).detach()
        C = C.detach()
        state = torch.zeros(
            (projected.shape[0], d_inner, d_state),
            dtype=torch.float32,
            device=projected.device,
        )
        outputs: list[torch.Tensor] = []
        for position in range(sequence_length):
            state = (
                discrete_A[:, position] * state
                + discrete_B[:, position] * values[:, position, :, None].float()
            )
            outputs.append(
                torch.einsum(
                    "bdn,bn->bd", state, C[:, position].float()
                )
            )
        scanned = torch.stack(outputs, dim=1)
        scanned = scanned + values.float() * D.float()
        gated = scanned * _mambalrp_silu(gate.transpose(1, 2).float())
        return gated / 2.0 + (gated / 2.0).detach()

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
    ) -> torch.Tensor:
        if inference_params is not None:
            raise ValueError("MambaLRP does not support inference caches")
        source = self.source
        projected = source.in_proj(hidden_states).transpose(1, 2)
        forward = self._scan_branch(
            projected,
            conv1d=source.conv1d,
            x_proj=source.x_proj,
            dt_proj=source.dt_proj,
            A_log=source.A_log,
            D=source.D,
            d_state=int(source.d_state),
            d_inner=int(source.d_inner),
        )
        bimamba_type = str(getattr(source, "bimamba_type", "none")).lower()
        if bimamba_type == "v2":
            required = (
                "conv1d_b",
                "x_proj_b",
                "dt_proj_b",
                "A_b_log",
                "D_b",
            )
            missing = [name for name in required if not hasattr(source, name)]
            if missing:
                raise RuntimeError(
                    "BiMamba-v2 mixer is missing reverse parameters: "
                    + ", ".join(missing)
                )
            backward = self._scan_branch(
                projected.flip(-1),
                conv1d=source.conv1d_b,
                x_proj=source.x_proj_b,
                dt_proj=source.dt_proj_b,
                A_log=source.A_b_log,
                D=source.D_b,
                d_state=int(source.d_state),
                d_inner=int(source.d_inner),
            )
            combined = forward + backward.flip(1)
            if bool(getattr(source, "if_divide_out", False)):
                combined = combined / 2.0
        elif bimamba_type in {"none", "", "false"}:
            combined = forward
        else:
            raise RuntimeError(
                f"Unsupported Mamba variant for MambaLRP: {bimamba_type}"
            )
        output = source.out_proj(combined.to(source.out_proj.weight.dtype))
        if getattr(source, "init_layer_scale", None) is not None:
            output = output * source.gamma
        return output


def _looks_like_mamba(module: nn.Module) -> bool:
    required = (
        "in_proj",
        "conv1d",
        "x_proj",
        "dt_proj",
        "A_log",
        "D",
        "out_proj",
        "d_state",
        "d_inner",
        "dt_rank",
    )
    return module.__class__.__name__ == "Mamba" and all(
        hasattr(module, name) for name in required
    )


def _load_actor_for_evaluation(agent, checkpoint_path: str) -> None:
    """Strictly load the bidirectional actor inference modules."""

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_actor = checkpoint.get("actor_encoder")
    if not isinstance(checkpoint_actor, dict):
        raise ValueError("Checkpoint has no actor_encoder state dictionary")
    agent.actor_encoder.load_state_dict(checkpoint_actor, strict=True)
    actor = checkpoint.get("actor")
    if not isinstance(actor, dict):
        raise ValueError("Checkpoint has no actor state dictionary")
    agent.actor.load_state_dict(actor, strict=True)
    del checkpoint


@contextmanager
def _mambalrp_mixers(root: nn.Module) -> Iterator[int]:
    """Temporarily replace every spatial and temporal Mamba mixer."""

    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if _looks_like_mamba(child):
                replacements.append((parent, name, child))
    if not replacements:
        raise RuntimeError("No compatible Mamba-1 mixers found in actor encoder")
    for parent, name, source in replacements:
        setattr(parent, name, MambaLRPMixer(source))
    try:
        yield len(replacements)
    finally:
        for parent, name, source in replacements:
            setattr(parent, name, source)


def compute_mambalrp(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
) -> AttributionResult:
    """Propagate policy relevance to Vision-Mamba patch embeddings."""

    base = torch.as_tensor(
        base_state, dtype=torch.float32, device=agent.device
    ).view(1, -1)
    depth_np = _prepare_depth(depth_sequence)
    depth = torch.as_tensor(
        depth_np, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    direction = _policy_direction(agent, base, depth)
    with torch.no_grad():
        native_action = _normalized_action(agent, base, depth)

    captured_embeddings: list[torch.Tensor] = []

    def capture_patch_embeddings(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Vision Mamba patch embedding must return a tensor")
        output.retain_grad()
        captured_embeddings.append(output)

    hook = agent.actor_encoder.vim.patch_embed.register_forward_hook(
        capture_patch_embeddings
    )
    agent.actor_encoder.zero_grad(set_to_none=True)
    agent.actor.zero_grad(set_to_none=True)
    try:
        with _mambalrp_mixers(agent.actor_encoder) as mixer_count:
            lrp_action = _normalized_action(agent, base, depth)
            forward_error = float(
                (lrp_action.detach() - native_action).abs().max().item()
            )
            if forward_error > 5e-4:
                raise RuntimeError(
                    "MambaLRP forward-equivalence check failed: "
                    f"max action error={forward_error:.6g}"
                )
            target = (lrp_action * direction).sum()
            target.backward()
    finally:
        hook.remove()

    if len(captured_embeddings) != 1:
        raise RuntimeError(
            "Expected one Vision Mamba patch-embedding call, "
            f"captured {len(captured_embeddings)}"
        )
    embeddings = captured_embeddings[0]
    if embeddings.grad is None:
        raise RuntimeError("MambaLRP did not reach patch embeddings")
    token_relevance = (embeddings * embeddings.grad).sum(dim=-1).abs()
    grid_height, grid_width = agent.actor_encoder.vim.patch_embed.grid_size
    frames = depth_np.shape[0]
    if token_relevance.shape[0] != frames:
        raise RuntimeError(
            f"Expected {frames} frame-token batches, got {token_relevance.shape[0]}"
        )
    patch_maps = token_relevance.view(
        frames, 1, int(grid_height), int(grid_width)
    )
    heatmaps = F.interpolate(
        patch_maps,
        size=depth_np.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return AttributionResult(
        heatmaps=heatmaps.detach().cpu().numpy().astype(np.float32),
        details={
            "definition": (
                "MambaLRP core-rule adaptation: absolute patch-token "
                "relevance using SiLU conservation, selective-SSM A/B/C "
                "detach, and half-gate propagation"
            ),
            "scope_limitation": (
                "Patch-token adaptation of Mamba-specific rules; not an "
                "end-to-end reproduction of every convolution, normalization, "
                "residual, and actor-layer LRP rule, and no claim of global "
                "relevance conservation"
            ),
            "policy_target": (
                "normalized action projected onto original deterministic "
                "action direction"
            ),
            "policy_direction": direction[0].detach().cpu().tolist(),
            "mamba_mixers_replaced": int(mixer_count),
            "forward_equivalence_max_action_error": forward_error,
            "patch_grid": [int(grid_height), int(grid_width)],
        },
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate Occlusion, IG, IG+SmoothGrad, and a MambaLRP core-rule "
            "adaptation for seed-25 CL-VSSM-SAC in the static test scene."
        )
    )
    parser.add_argument("--model_seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--episode_seed", type=int, default=DEFAULT_EPISODE_SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=list(ALL_METHODS),
    )
    parser.add_argument(
        "--capture_steps",
        type=int,
        nargs="+",
        default=None,
        help="Explicit steps; default selects near-obstacle spaced samples.",
    )
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--min_sample_gap", type=int, default=DEFAULT_MIN_SAMPLE_GAP
    )
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--occlusion_batch_size", type=int, default=64)
    parser.add_argument("--baseline_value", type=float, default=255.0)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--ig_internal_batch_size", type=int, default=8)
    parser.add_argument("--noise_samples", type=int, default=8)
    parser.add_argument("--noise_stdev", type=float, default=5.0)
    parser.add_argument("--smooth_sigma", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run dependency-free internal attribution tests and exit.",
    )
    script_args, remaining = parser.parse_known_args(argv)

    positive_names = (
        "num_samples",
        "window_size",
        "stride",
        "occlusion_batch_size",
        "ig_steps",
        "ig_internal_batch_size",
        "noise_samples",
        "dpi",
    )
    for name in positive_names:
        if int(getattr(script_args, name)) <= 0:
            parser.error(f"--{name} must be positive")
    if script_args.min_sample_gap < 0:
        parser.error("--min_sample_gap must be non-negative")
    if script_args.noise_stdev < 0 or script_args.smooth_sigma < 0:
        parser.error("noise and smoothing standard deviations must be non-negative")
    script_args.methods = list(dict.fromkeys(script_args.methods))
    if script_args.capture_steps is not None:
        script_args.capture_steps = sorted(set(script_args.capture_steps))
        if script_args.capture_steps[0] < 0:
            parser.error("--capture_steps must be non-negative")

    args = get_config(remaining)
    args.algorithm_name = "CL-VSSM-SAC"
    args.seed = int(script_args.model_seed)
    params_path = REPO_ROOT / "algorithm" / "SB_PER_VSSM_SAC" / "params.yaml"
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    params = params.get("params", params)
    if not isinstance(params, dict):
        raise ValueError(f"Algorithm params must be a mapping: {params_path}")
    args.algorithm_params = dict(params)
    for key, value in params.items():
        setattr(args, key, value)
    return script_args, args


def _default_checkpoint(model_seed: int) -> str:
    return str(
        REPO_ROOT
        / "models"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / "async_final.pth"
    )


def _default_output_dir(model_seed: int, episode_seed: int) -> Path:
    run_stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    return (
        REPO_ROOT
        / "results"
        / "explainability"
        / "occlusion"
        / "test_scene"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / f"episode{int(episode_seed)}"
        / run_stamp
    )


def _smooth_maps(heatmaps: np.ndarray, sigma: float) -> np.ndarray:
    maps = np.asarray(heatmaps, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"Expected (T,H,W) maps, got {maps.shape}")
    if sigma <= 0:
        return maps.copy()
    return np.stack(
        [
            cv2.GaussianBlur(
                frame, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma)
            )
            for frame in maps
        ]
    )


def _method_limits(
    records: Sequence[CaptureRecord],
    methods: Sequence[str],
    smooth_sigma: float,
) -> dict[str, float]:
    limits: dict[str, float] = {}
    for method in methods:
        values = np.concatenate(
            [
                _smooth_maps(
                    record.attributions[method].heatmaps, smooth_sigma
                ).reshape(-1)
                for record in records
            ]
        )
        finite = values[np.isfinite(values)]
        limit = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        limits[method] = max(limit, 1e-8)
    return limits


def _overlay(ax, depth, heatmap, *, limit: float, alpha: float = 0.58):
    ax.imshow(depth, cmap="gray", vmin=0, vmax=255)
    ax.imshow(
        heatmap,
        cmap="turbo",
        norm=Normalize(vmin=0, vmax=limit),
        alpha=alpha,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _add_method_colorbars(figure, axes, methods, limits):
    for row, method in enumerate(methods, start=1):
        scalar = ScalarMappable(
            norm=Normalize(vmin=0, vmax=limits[method]), cmap="turbo"
        )
        scalar.set_array([])
        figure.colorbar(
            scalar,
            ax=axes[row, :],
            shrink=0.7,
            label=f"{METHOD_LABELS[method]} magnitude",
        )


def _render_record(
    record: CaptureRecord,
    output_path: Path,
    *,
    methods: Sequence[str],
    limits: dict[str, float],
    smooth_sigma: float,
    dpi: int,
) -> None:
    frames = record.depth.shape[0]
    figure, axes = plt.subplots(
        len(methods) + 1,
        frames,
        figsize=(3.0 * frames, 2.65 * (len(methods) + 1)),
        squeeze=False,
        constrained_layout=True,
    )
    for frame in range(frames):
        axes[0, frame].imshow(
            record.depth[frame], cmap="gray", vmin=0, vmax=255
        )
        lag = frames - frame - 1
        axes[0, frame].set_title("t" if lag == 0 else f"t-{lag}")
        axes[0, frame].set_xticks([])
        axes[0, frame].set_yticks([])
    axes[0, 0].set_ylabel("Original")

    for row, method in enumerate(methods, start=1):
        maps = _smooth_maps(
            record.attributions[method].heatmaps, smooth_sigma
        )
        for frame in range(frames):
            _overlay(
                axes[row, frame],
                record.depth[frame],
                maps[frame],
                limit=limits[method],
            )
        axes[row, 0].set_ylabel(METHOD_LABELS[method])
    _add_method_colorbars(figure, axes, methods, limits)
    figure.suptitle(
        f"CL-VSSM-SAC whole-policy explanations — step {record.step}"
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _render_summary(
    records: Sequence[CaptureRecord],
    output_path: Path,
    *,
    methods: Sequence[str],
    limits: dict[str, float],
    smooth_sigma: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(methods) + 1,
        len(records),
        figsize=(3.0 * len(records), 2.65 * (len(methods) + 1)),
        squeeze=False,
        constrained_layout=True,
    )
    for column, record in enumerate(records):
        depth = record.depth[-1]
        axes[0, column].imshow(depth, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(
            f"Step {record.step}\nproximity={record.obstacle_proximity:.1f}"
        )
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        for row, method in enumerate(methods, start=1):
            heatmap = _smooth_maps(
                record.attributions[method].heatmaps, smooth_sigma
            )[-1]
            _overlay(
                axes[row, column],
                depth,
                heatmap,
                limit=limits[method],
            )
    axes[0, 0].set_ylabel("Original")
    for row, method in enumerate(methods, start=1):
        axes[row, 0].set_ylabel(METHOD_LABELS[method])
    _add_method_colorbars(figure, axes, methods, limits)
    figure.suptitle(
        "CL-VSSM-SAC — current-frame whole-policy explanations"
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_record(record: CaptureRecord, output_dir: Path) -> None:
    arrays = {
        "step": np.int32(record.step),
        "base_state": record.base_state.astype(np.float32),
        "depth": record.depth.astype(np.float32),
        "original_action": record.action.astype(np.float32),
        "obstacle_proximity": np.float32(record.obstacle_proximity),
    }
    for method, result in record.attributions.items():
        arrays[f"heatmap_{method}"] = result.heatmaps.astype(np.float32)
        if result.action_heatmaps is not None:
            arrays[f"action_heatmaps_{method}"] = (
                result.action_heatmaps.astype(np.float32)
            )
    np.savez_compressed(
        output_dir / f"step_{record.step:04d}_attributions.npz", **arrays
    )


def _write_metadata(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _compute_attributions(
    sample: TrajectoryStep,
    *,
    methods: Sequence[str],
    predictor: BatchedDeterministicActor,
    agent,
    action_ranges: np.ndarray,
    script_args,
) -> dict[str, AttributionResult]:
    results: dict[str, AttributionResult] = {}
    if "occlusion" in methods:
        results["occlusion"] = compute_occlusion_sensitivity(
            predictor,
            sample.base_state,
            sample.depth,
            action_ranges=action_ranges,
            window_size=int(script_args.window_size),
            stride=int(script_args.stride),
            baseline_value=float(script_args.baseline_value),
            batch_size=int(script_args.occlusion_batch_size),
        )
    if "ig" in methods:
        results["ig"] = compute_integrated_gradients(
            agent,
            sample.base_state,
            sample.depth,
            baseline_value=float(script_args.baseline_value),
            n_steps=int(script_args.ig_steps),
            internal_batch_size=int(script_args.ig_internal_batch_size),
        )
    if "ig_smoothgrad" in methods:
        results["ig_smoothgrad"] = compute_integrated_gradients(
            agent,
            sample.base_state,
            sample.depth,
            baseline_value=float(script_args.baseline_value),
            n_steps=int(script_args.ig_steps),
            internal_batch_size=int(script_args.ig_internal_batch_size),
            smooth_samples=int(script_args.noise_samples),
            noise_stdev=float(script_args.noise_stdev),
            noise_seed=int(script_args.model_seed) * 100000 + int(sample.step),
        )
    if "mambalrp" in methods:
        results["mambalrp"] = compute_mambalrp(
            agent, sample.base_state, sample.depth
        )
    return results


def run_visualization(script_args, args) -> Path:
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError(
            "VSSM-SAC attribution requires CUDA because this repository uses "
            "fused Mamba/Triton kernels."
        )
    from main_async import _configure_reproducibility, get_agent_class

    model_seed = int(script_args.model_seed)
    _configure_reproducibility(model_seed, args)
    output_dir = (
        Path(script_args.output_dir).resolve()
        if script_args.output_dir
        else _default_output_dir(model_seed, int(script_args.episode_seed))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(
        script_args.checkpoint, _default_checkpoint(model_seed)
    )

    env = SceneEvalAirSimEnv(
        takeoff_height=args.takeoff_height,
        config=args,
        stack_frames=int(args.n_frames),
    )
    trajectory: list[TrajectoryStep] = []
    records: list[CaptureRecord] = []
    termination = "max_steps"
    try:
        obs, _ = env.reset(seed=int(script_args.episode_seed))
        depth_shape = tuple(int(value) for value in obs["depth"].shape)
        expected = (int(args.n_frames), 128, 128)
        if depth_shape != expected:
            raise ValueError(
                f"Seed-25 CL-VSSM-SAC expects depth {expected}, got {depth_shape}"
            )
        if script_args.window_size > min(depth_shape[-2:]):
            raise ValueError("--window_size exceeds the depth image")

        agent_class = get_agent_class(args.algorithm_name)
        agent = agent_class(
            obs["base"].shape[0],
            (1, depth_shape[-2], depth_shape[-1]),
            env.action_space,
            args,
            device=torch.device("cuda"),
            seed=model_seed,
        )
        _load_actor_for_evaluation(agent, checkpoint)
        print(f"[Attribution] Loaded actor model: {checkpoint}")
        set_agent_eval_mode(agent)
        predictor = BatchedDeterministicActor(agent)
        action_ranges = np.asarray(
            env.action_space.high - env.action_space.low, dtype=np.float32
        )

        max_steps = int(getattr(args, "episode_length", 300))
        for step in range(max_steps):
            base = np.asarray(obs["base"], dtype=np.float32)
            depth = _prepare_depth(obs["depth"])
            action = predictor(base[None], depth[None])[0]
            trajectory.append(
                TrajectoryStep(
                    step=step,
                    base_state=base.copy(),
                    depth=depth.copy(),
                    action=action.copy(),
                    obstacle_proximity=_obstacle_proximity(depth),
                )
            )
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if bool(info.get("is_success", False)):
                    termination = "success"
                elif bool(info.get("has_collided", False)):
                    termination = "collision"
                elif truncated:
                    termination = "timeout"
                else:
                    termination = "other_failure"
                print(f"[Attribution] Episode ended at step {step}: {termination}")
                break
        if not trajectory:
            raise RuntimeError("The test episode produced no trajectory steps")

        if script_args.capture_steps is None:
            selected_indices = select_spaced_top_indices(
                [item.obstacle_proximity for item in trajectory],
                count=int(script_args.num_samples),
                min_gap=int(script_args.min_sample_gap),
            )
            selection_method = "top_obstacle_proximity_with_temporal_spacing"
            missing_steps: list[int] = []
        else:
            requested = set(script_args.capture_steps)
            selected_indices = [
                index
                for index, item in enumerate(trajectory)
                if item.step in requested
            ]
            missing_steps = sorted(
                requested - {trajectory[index].step for index in selected_indices}
            )
            selection_method = "explicit_capture_steps"
        if not selected_indices:
            raise RuntimeError("No selected observations exist in the trajectory")

        selected_steps = [trajectory[index].step for index in selected_indices]
        print(
            "[Attribution] Selected steps: "
            + ", ".join(str(value) for value in selected_steps)
        )
        for index in selected_indices:
            sample = trajectory[index]
            print(
                f"[Attribution] step={sample.step}, methods="
                + ",".join(script_args.methods)
            )
            attributions = _compute_attributions(
                sample,
                methods=script_args.methods,
                predictor=predictor,
                agent=agent,
                action_ranges=action_ranges,
                script_args=script_args,
            )
            record = CaptureRecord(
                step=sample.step,
                base_state=sample.base_state,
                depth=sample.depth,
                action=sample.action,
                obstacle_proximity=sample.obstacle_proximity,
                attributions=attributions,
            )
            records.append(record)
            _save_record(record, output_dir)

        limits = _method_limits(
            records, script_args.methods, float(script_args.smooth_sigma)
        )
        for record in records:
            _render_record(
                record,
                output_dir / f"step_{record.step:04d}_four_frames.png",
                methods=script_args.methods,
                limits=limits,
                smooth_sigma=float(script_args.smooth_sigma),
                dpi=int(script_args.dpi),
            )
        _render_summary(
            records,
            output_dir / "current_frame_summary.png",
            methods=script_args.methods,
            limits=limits,
            smooth_sigma=float(script_args.smooth_sigma),
            dpi=int(script_args.dpi),
        )

        metadata = {
            "algorithm": "CL-VSSM-SAC",
            "model_seed": model_seed,
            "checkpoint": os.path.abspath(checkpoint),
            "environment": "static_test_environment",
            "episode_seed": int(script_args.episode_seed),
            "termination": termination,
            "trajectory_steps": len(trajectory),
            "sample_selection": selection_method,
            "capture_steps_requested": script_args.capture_steps,
            "capture_steps_completed": selected_steps,
            "capture_steps_missing": missing_steps,
            "preferred_min_sample_gap": int(script_args.min_sample_gap),
            "effective_min_sample_gap": _minimum_pair_gap(selected_steps),
            "obstacle_proximity_scores": {
                str(record.step): record.obstacle_proximity for record in records
            },
            "methods": list(script_args.methods),
            "method_details": {
                str(record.step): {
                    method: result.details
                    for method, result in record.attributions.items()
                }
                for record in records
            },
            "policy_target_note": (
                "CaMeRL does not disclose how its multidimensional policy "
                "output is scalarized. This implementation documents every "
                "method's target in method_details."
            ),
            "window_size": int(script_args.window_size),
            "stride": int(script_args.stride),
            "baseline_value": float(script_args.baseline_value),
            "ig_steps": int(script_args.ig_steps),
            "noise_samples": int(script_args.noise_samples),
            "noise_stdev": float(script_args.noise_stdev),
            "action_labels": list(ACTION_LABELS),
            "action_units": list(ACTION_UNITS),
            "action_ranges": action_ranges.tolist(),
            "shared_color_limits_per_method": limits,
            "smooth_sigma_render_only": float(script_args.smooth_sigma),
            "depth_encoding": "0=near, 255=15m/free-space clipping limit",
            "references": {
                "occlusion": "Zeiler and Fergus, ECCV 2014",
                "integrated_gradients": "Sundararajan et al., ICML 2017",
                "smoothgrad": "Smilkov et al., ICML Workshop 2017",
                "mambalrp": "Jafari et al., NeurIPS 2024",
            },
        }
        _write_metadata(output_dir / "metadata.json", metadata)
        if missing_steps:
            print(
                "[Attribution] Missing explicit steps: "
                + ", ".join(str(value) for value in missing_steps)
            )
        print(f"[Attribution] Results saved to: {output_dir}")
        return output_dir
    finally:
        close_env(env, label="CL-VSSM-SAC attribution visualization")


def run_self_tests() -> None:
    """Small dependency-free checks kept in this single requested script."""

    assert sliding_window_starts(10, 4, 4) == [0, 4, 6]
    assert select_spaced_top_indices(
        [3.0, 2.0, 1.0], count=3, min_gap=10
    ) == [0, 1, 2]

    depth = np.ones((2, 4, 4), dtype=np.float32)

    def predictor(base_batch, depth_batch):
        del base_batch
        overall = depth_batch.mean(axis=(1, 2, 3))
        latest = depth_batch[:, 1].mean(axis=(1, 2))
        return np.stack([overall, latest], axis=1)

    result = compute_occlusion_sensitivity(
        predictor,
        np.array([0.0], dtype=np.float32),
        depth,
        action_ranges=np.ones(2, dtype=np.float32),
        window_size=2,
        stride=2,
        baseline_value=0.0,
        batch_size=3,
    )
    np.testing.assert_allclose(result.action_heatmaps[0, 0], 0.125)
    np.testing.assert_allclose(result.action_heatmaps[1, 1], 0.25)

    class ToyAgent:
        device = torch.device("cpu")

        def __init__(self):
            self.actor_encoder = nn.Identity()
            self.actor = lambda state, deterministic: state[:, :2]

        def _encode_state(self, base, image, encoder):
            del encoder
            return torch.cat([image.flatten(1), base], dim=1)

    toy = ToyAgent()
    toy_depth = np.ones((1, 2, 2), dtype=np.float32)
    ig = compute_integrated_gradients(
        toy,
        np.array([0.0], dtype=np.float32),
        toy_depth,
        baseline_value=0.0,
        n_steps=8,
        internal_batch_size=4,
    )
    assert ig.heatmaps.shape == toy_depth.shape
    assert np.all(np.isfinite(ig.heatmaps))

    class Mamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = 2
            self.d_state = 2
            self.d_conv = 2
            self.expand = 1
            self.d_inner = 2
            self.dt_rank = 1
            self.in_proj = nn.Linear(2, 4, bias=False)
            self.conv1d = nn.Conv1d(
                2, 2, kernel_size=2, padding=1, groups=2
            )
            self.x_proj = nn.Linear(2, 5, bias=False)
            self.dt_proj = nn.Linear(1, 2)
            self.A_log = nn.Parameter(torch.zeros(2, 2))
            self.D = nn.Parameter(torch.ones(2))
            self.out_proj = nn.Linear(2, 2, bias=False)

    source = Mamba()
    wrapper = MambaLRPMixer(source)
    sequence = torch.randn(2, 3, 2)

    def native_branch(
        projected,
        conv1d,
        x_proj,
        dt_proj,
        A_log,
        D,
    ):
        values, gate = projected.chunk(2, dim=1)
        values = F.silu(conv1d(values)[..., :3]).transpose(1, 2)
        parameters = x_proj(values)
        delta, B, C = torch.split(parameters, [1, 2, 2], dim=-1)
        delta = F.softplus(dt_proj(delta))
        A = -torch.exp(A_log.float())
        discrete_A = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        discrete_B = torch.einsum("bld,bln->bldn", delta, B)
        state = torch.zeros(2, 2, 2)
        outputs = []
        for position in range(3):
            state = (
                discrete_A[:, position] * state
                + discrete_B[:, position] * values[:, position, :, None]
            )
            outputs.append(
                torch.einsum("bdn,bn->bd", state, C[:, position])
            )
        scanned = torch.stack(outputs, dim=1) + values * D
        return scanned * F.silu(gate.transpose(1, 2))

    with torch.no_grad():
        projected = source.in_proj(sequence).transpose(1, 2)
        expected = source.out_proj(
            native_branch(
                projected,
                source.conv1d,
                source.x_proj,
                source.dt_proj,
                source.A_log,
                source.D,
            )
        )
        actual = wrapper(sequence)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    bidirectional = Mamba()
    bidirectional.bimamba_type = "v2"
    bidirectional.if_divide_out = True
    bidirectional.conv1d_b = copy.deepcopy(bidirectional.conv1d)
    bidirectional.x_proj_b = copy.deepcopy(bidirectional.x_proj)
    bidirectional.dt_proj_b = copy.deepcopy(bidirectional.dt_proj)
    bidirectional.A_b_log = nn.Parameter(
        bidirectional.A_log.detach().clone()
    )
    bidirectional.D_b = nn.Parameter(bidirectional.D.detach().clone())
    with torch.no_grad():
        projected = bidirectional.in_proj(sequence).transpose(1, 2)
        forward = native_branch(
            projected,
            bidirectional.conv1d,
            bidirectional.x_proj,
            bidirectional.dt_proj,
            bidirectional.A_log,
            bidirectional.D,
        )
        backward = native_branch(
            projected.flip(-1),
            bidirectional.conv1d_b,
            bidirectional.x_proj_b,
            bidirectional.dt_proj_b,
            bidirectional.A_b_log,
            bidirectional.D_b,
        )
        expected = bidirectional.out_proj(
            (forward + backward.flip(1)) / 2.0
        )
        actual = MambaLRPMixer(bidirectional)(sequence)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    print("All internal attribution tests passed.")


def main(argv=None) -> None:
    script_args, args = _parse_args(argv)
    if script_args.self_test:
        run_self_tests()
        return
    run_visualization(script_args, args)


if __name__ == "__main__":
    main()
