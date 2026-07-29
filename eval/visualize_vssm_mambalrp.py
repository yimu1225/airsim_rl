#!/usr/bin/env python3
"""Paper-faithful MambaLRP explanations for CL-VSSM-SAC.

This standalone script adapts Jafari et al. (NeurIPS 2024) to a continuous
VSSM-SAC policy.  It follows the authors' public Vision-Mamba implementation:

* relevance starts at the deterministic policy output;
* ordinary layers use the Gradient x Input / LRP-0 path;
* every Mamba mixer uses the paper's SiLU, selective-SSM and half-gate rules;
* only spatial Vision-Mamba Conv1d layers use LRP-gamma with gamma=0.25;
* signed relevance is read at post-position-embedding patch tokens;
* the middle CLS token is excluded from the spatial map;
* patch relevance is interpolated only for visualization, never presented as
  higher-resolution raw relevance.

The ImageNet paper explains a predicted class logit.  A continuous policy has
no predicted class, so the primary scalar target is the L2 norm of the
normalized deterministic action.  Signed per-action explanations are also
saved so this unavoidable task adaptation remains auditable.
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
from typing import Iterator, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

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
LRP_GAMMA = 0.25
STABILIZER = 1e-6
ACTION_LABELS = ("Forward velocity", "Yaw rate", "Vertical velocity")
ACTION_KEYS = ("forward_velocity", "yaw_rate", "vertical_velocity")


@dataclass
class TrajectoryStep:
    step: int
    base_state: np.ndarray
    depth: np.ndarray
    physical_action: np.ndarray
    obstacle_proximity: float


@dataclass
class MambaLRPResult:
    patch_relevance: np.ndarray
    display_relevance: np.ndarray
    action_patch_relevance: np.ndarray
    action_display_relevance: np.ndarray
    details: dict


@dataclass
class CaptureRecord:
    sample: TrajectoryStep
    result: MambaLRPResult


def _stabilize(value: torch.Tensor) -> torch.Tensor:
    """Paper-compatible denominator stabilizer."""

    return value + ((value == 0).to(value) + value.sign()) * STABILIZER


def _mambalrp_identity_activation(
    value: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Keep an activation's forward value and use the LRP identity backward."""

    return value * (output / _stabilize(value)).detach()


def _mambalrp_silu(value: torch.Tensor) -> torch.Tensor:
    """Algorithm 1: SiLU with a relevance-conserving backward pass."""

    return _mambalrp_identity_activation(value, F.silu(value))


def _conv1d_with_parameters(
    layer: nn.Conv1d,
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.conv1d(
        value,
        weight,
        bias,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )


def _gamma_parameters(
    parameter: torch.Tensor,
    *,
    gamma: float,
    positive: bool,
) -> torch.Tensor:
    selected = parameter.clamp(min=0) if positive else parameter.clamp(max=0)
    return parameter + float(gamma) * selected


def _mambalrp_gamma_conv1d(
    layer: nn.Conv1d,
    value: torch.Tensor,
    *,
    gamma: float = LRP_GAMMA,
) -> torch.Tensor:
    """Generalized LRP-gamma rule used for Vim Conv1d layers in the paper."""

    native = layer(value)
    positive_value = value.clamp(min=0)
    negative_value = value.clamp(max=0)
    weight_positive = _gamma_parameters(
        layer.weight, gamma=gamma, positive=True
    )
    weight_negative = _gamma_parameters(
        layer.weight, gamma=gamma, positive=False
    )
    if layer.bias is None:
        bias_positive = bias_negative = zero_bias = None
    else:
        bias_positive = _gamma_parameters(
            layer.bias, gamma=gamma, positive=True
        )
        bias_negative = _gamma_parameters(
            layer.bias, gamma=gamma, positive=False
        )
        zero_bias = torch.zeros_like(layer.bias)

    positive_output = _conv1d_with_parameters(
        layer, positive_value, weight_positive, bias_positive
    ) + _conv1d_with_parameters(
        layer, negative_value, weight_negative, zero_bias
    )
    negative_output = _conv1d_with_parameters(
        layer, positive_value, weight_negative, bias_negative
    ) + _conv1d_with_parameters(
        layer, negative_value, weight_positive, zero_bias
    )
    redistributed = torch.where(
        native > STABILIZER,
        positive_output,
        torch.where(native < -STABILIZER, negative_output, native),
    )
    return redistributed * (
        native / _stabilize(redistributed)
    ).detach()


class MambaLRPMixer(nn.Module):
    """Forward-equivalent Mamba-1 mixer with the paper's backward rules."""

    def __init__(
        self,
        source: nn.Module,
        *,
        conv1d_gamma: float | None = LRP_GAMMA,
    ):
        super().__init__()
        self.source = source
        self.conv1d_gamma = conv1d_gamma

    @staticmethod
    def _scan_branch(
        projected: torch.Tensor,
        *,
        conv1d: nn.Conv1d,
        x_proj: nn.Module,
        dt_proj: nn.Module,
        A_log: torch.Tensor,
        D: torch.Tensor,
        d_state: int,
        d_inner: int,
        conv1d_gamma: float | None,
    ) -> torch.Tensor:
        sequence_length = projected.shape[-1]
        values, gate = projected.chunk(2, dim=1)
        convolved = (
            conv1d(values)
            if conv1d_gamma is None
            else _mambalrp_gamma_conv1d(
                conv1d, values, gamma=conv1d_gamma
            )
        )[..., :sequence_length]
        values = _mambalrp_silu(convolved).transpose(1, 2)

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
                + discrete_B[:, position]
                * values[:, position, :, None].float()
            )
            outputs.append(
                torch.einsum(
                    "bdn,bn->bd", state, C[:, position].float()
                )
            )
        scanned = torch.stack(outputs, dim=1)
        scanned = scanned + values.float() * D.float()
        gated = scanned * _mambalrp_silu(
            gate.transpose(1, 2).float()
        )
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
            conv1d_gamma=self.conv1d_gamma,
        )

        bimamba_type = str(
            getattr(source, "bimamba_type", "none")
        ).lower()
        if bimamba_type == "v2":
            required = (
                "conv1d_b",
                "x_proj_b",
                "dt_proj_b",
                "A_b_log",
                "D_b",
            )
            missing = [
                name for name in required if not hasattr(source, name)
            ]
            if missing:
                raise RuntimeError(
                    "BiMamba-v2 mixer lacks reverse parameters: "
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
                conv1d_gamma=self.conv1d_gamma,
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

        output = source.out_proj(
            combined.to(source.out_proj.weight.dtype)
        )
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


@contextmanager
def _paper_lrp_modules(agent) -> Iterator[dict[str, int]]:
    """Temporarily install paper rules without modifying learned weights."""

    replacements: list[
        tuple[nn.Module, str, nn.Module, nn.Module]
    ] = []
    spatial_count = 0
    temporal_count = 0
    for root, gamma, category in (
        (agent.actor_encoder.vim, LRP_GAMMA, "spatial"),
        (agent.actor_encoder.temporal_mamba, None, "temporal"),
    ):
        for parent in list(root.modules()):
            for name, child in list(parent.named_children()):
                if _looks_like_mamba(child):
                    replacements.append(
                        (
                            parent,
                            name,
                            child,
                            MambaLRPMixer(
                                child, conv1d_gamma=gamma
                            ),
                        )
                    )
                    if category == "spatial":
                        spatial_count += 1
                    else:
                        temporal_count += 1
    if not any(
        isinstance(replacement, MambaLRPMixer)
        for _, _, _, replacement in replacements
    ):
        raise RuntimeError("No compatible Mamba-1 mixers found")
    for parent, name, _source, replacement in replacements:
        setattr(parent, name, replacement)
    try:
        yield {
            "mamba_mixers": spatial_count + temporal_count,
            "spatial_mamba_mixers": spatial_count,
            "temporal_mamba_mixers": temporal_count,
        }
    finally:
        for parent, name, source, _replacement in replacements:
            setattr(parent, name, source)


def _remove_middle_cls_relevance(
    token_relevance: torch.Tensor,
    *,
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove Vim's middle CLS token and reshape patches to the 2-D grid."""

    grid_height, grid_width = map(int, grid_size)
    patch_count = grid_height * grid_width
    if token_relevance.ndim != 2:
        raise ValueError(
            "token_relevance must be (B, number_of_tokens)"
        )
    if token_relevance.shape[1] != patch_count + 1:
        raise ValueError(
            f"Expected {patch_count + 1} tokens, "
            f"got {token_relevance.shape[1]}"
        )
    cls_position = patch_count // 2
    cls_relevance = token_relevance[:, cls_position]
    patches = torch.cat(
        (
            token_relevance[:, :cls_position],
            token_relevance[:, cls_position + 1 :],
        ),
        dim=1,
    )
    return (
        patches.reshape(
            token_relevance.shape[0], grid_height, grid_width
        ),
        cls_relevance,
    )


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
    latest = _prepare_depth(depth)[-1]
    return float(255.0 - np.percentile(latest, 10.0))


def select_spaced_top_indices(
    scores: Sequence[float],
    *,
    count: int,
    min_gap: int,
) -> list[int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must contain finite values")
    target_count = min(int(count), int(values.size))
    ranking = np.argsort(-values, kind="stable").tolist()
    for effective_gap in range(int(min_gap), -1, -1):
        selected: list[int] = []
        for index in ranking:
            if all(
                abs(index - previous) >= effective_gap
                for previous in selected
            ):
                selected.append(index)
                if len(selected) == target_count:
                    return sorted(selected)
    raise AssertionError("gap-zero selection must satisfy the target count")


def _minimum_pair_gap(indices: Sequence[int]) -> int | None:
    ordered = sorted(int(value) for value in indices)
    if len(ordered) < 2:
        return None
    return min(
        right - left for left, right in zip(ordered, ordered[1:])
    )


def _normalized_action(
    agent,
    base: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    state = agent._encode_state(base, depth, agent.actor_encoder)
    return agent.actor(state, deterministic=True)


def _physical_action(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
) -> np.ndarray:
    base = torch.as_tensor(
        base_state, dtype=torch.float32, device=agent.device
    ).view(1, -1)
    depth = torch.as_tensor(
        _prepare_depth(depth_sequence),
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)
    with torch.no_grad():
        normalized = _normalized_action(agent, base, depth)
        physical = agent.action_scale * normalized + agent.action_bias
    return physical[0].detach().cpu().numpy().astype(np.float32)


def _normalized_policy_score(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
) -> float:
    base = torch.as_tensor(
        base_state, dtype=torch.float32, device=agent.device
    ).view(1, -1)
    depth = torch.as_tensor(
        _prepare_depth(depth_sequence),
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)
    with torch.no_grad():
        action = _normalized_action(agent, base, depth)
        score = torch.linalg.vector_norm(action, ord=2, dim=1)
    return float(score[0].item())


def _policy_scalar(
    normalized_action: torch.Tensor,
    target_index: int | None,
) -> torch.Tensor:
    if target_index is None:
        return torch.linalg.vector_norm(
            normalized_action, ord=2, dim=1
        ).sum()
    return normalized_action[:, int(target_index)].sum()


def _normalize_signed_maps(maps: np.ndarray) -> np.ndarray:
    values = np.asarray(maps, dtype=np.float32)
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum <= 0:
        return np.zeros_like(values)
    return values / maximum


def _mask_ranked_patches(
    depth_sequence: np.ndarray,
    ranked_flat_indices: np.ndarray,
    *,
    masked_count: int,
    grid_size: tuple[int, int],
    mask_value: float,
) -> np.ndarray:
    """Mask complete attribution units; one unit is one frame patch."""

    depth = _prepare_depth(depth_sequence).copy()
    frames, height, width = depth.shape
    grid_height, grid_width = map(int, grid_size)
    if height % grid_height or width % grid_width:
        raise ValueError(
            "Depth resolution must be divisible by the patch relevance grid"
        )
    patch_height = height // grid_height
    patch_width = width // grid_width
    unit_count = frames * grid_height * grid_width
    ranking = np.asarray(ranked_flat_indices, dtype=np.int64).reshape(-1)
    if ranking.size != unit_count:
        raise ValueError(
            f"Expected a ranking of {unit_count} patches, got {ranking.size}"
        )
    for flat_index in ranking[: int(masked_count)]:
        frame, remainder = divmod(
            int(flat_index), grid_height * grid_width
        )
        row, column = divmod(remainder, grid_width)
        top = row * patch_height
        left = column * patch_width
        depth[
            frame,
            top : top + patch_height,
            left : left + patch_width,
        ] = float(mask_value)
    return depth


def evaluate_patch_flipping(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
    patch_relevance: np.ndarray,
    *,
    mask_value: float,
    points: int = 11,
) -> dict:
    """Paper-style MoRF/LeRF flipping adapted to patch input features."""

    relevance = np.asarray(patch_relevance, dtype=np.float32)
    if relevance.ndim != 3:
        raise ValueError("patch_relevance must be (T,Hg,Wg)")
    if int(points) < 2:
        raise ValueError("points must be at least two")
    fractions = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    flat = relevance.reshape(-1)
    morf_order = np.argsort(-flat, kind="stable")
    lerf_order = np.argsort(flat, kind="stable")
    curves: dict[str, list[float]] = {"morf": [], "lerf": []}
    for name, ranking in (("morf", morf_order), ("lerf", lerf_order)):
        for fraction in fractions:
            masked_count = int(round(float(fraction) * flat.size))
            perturbed = _mask_ranked_patches(
                depth_sequence,
                ranking,
                masked_count=masked_count,
                grid_size=tuple(relevance.shape[-2:]),
                mask_value=float(mask_value),
            )
            curves[name].append(
                _normalized_policy_score(agent, base_state, perturbed)
            )
    morf_auc = float(np.trapezoid(curves["morf"], fractions))
    lerf_auc = float(np.trapezoid(curves["lerf"], fractions))
    return {
        "feature_unit": "frame_patch",
        "ranking": "signed_relevance",
        "fractions": fractions.tolist(),
        "morf_scores": curves["morf"],
        "lerf_scores": curves["lerf"],
        "morf_auc": morf_auc,
        "lerf_auc": lerf_auc,
        "delta_a_f_lerf_minus_morf": lerf_auc - morf_auc,
        "mask_value_depth_units": float(mask_value),
        "score": "l2_norm_of_normalized_deterministic_action",
    }


def _interpolate_patch_maps(
    patch_maps: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    tensor = torch.as_tensor(patch_maps, dtype=torch.float32)
    resized = F.interpolate(
        tensor[:, None],
        size=tuple(map(int, output_size)),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return resized.cpu().numpy().astype(np.float32)


def _single_target_relevance(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
    *,
    target_index: int | None,
) -> tuple[np.ndarray, dict]:
    """Run one paper-style backward pass for one policy scalar."""

    base = torch.as_tensor(
        base_state, dtype=torch.float32, device=agent.device
    ).view(1, -1)
    base = base.detach().requires_grad_(True)
    depth_np = _prepare_depth(depth_sequence)
    depth = torch.as_tensor(
        depth_np, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)

    with torch.no_grad():
        native_action = _normalized_action(agent, base.detach(), depth)

    captured_embeddings: list[torch.Tensor] = []

    def detach_post_position_embeddings(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Vim position embedding output must be a tensor")
        detached = output.detach().requires_grad_(True)
        captured_embeddings.append(detached)
        return detached

    vim = agent.actor_encoder.vim
    if not hasattr(vim, "pos_drop"):
        raise RuntimeError(
            "Paper-faithful Vision MambaLRP requires absolute position "
            "embeddings and vim.pos_drop"
        )
    hook = vim.pos_drop.register_forward_hook(
        detach_post_position_embeddings
    )
    agent.actor_encoder.zero_grad(set_to_none=True)
    agent.actor.zero_grad(set_to_none=True)
    try:
        with _paper_lrp_modules(agent) as replacements:
            lrp_action = _normalized_action(agent, base, depth)
            forward_error = float(
                (lrp_action.detach() - native_action).abs().max().item()
            )
            if forward_error > 5e-4:
                raise RuntimeError(
                    "MambaLRP forward-equivalence check failed: "
                    f"max normalized action error={forward_error:.6g}"
                )
            target = _policy_scalar(lrp_action, target_index)
            target.backward()
    finally:
        hook.remove()

    if len(captured_embeddings) != 1:
        raise RuntimeError(
            "Expected one post-position-embedding tensor, "
            f"captured {len(captured_embeddings)}"
        )
    embeddings = captured_embeddings[0]
    if embeddings.grad is None:
        raise RuntimeError("MambaLRP did not reach patch embeddings")

    token_relevance = (embeddings * embeddings.grad).sum(dim=-1)
    patch_relevance, cls_relevance = _remove_middle_cls_relevance(
        token_relevance,
        grid_size=tuple(vim.patch_embed.grid_size),
    )
    frames = depth_np.shape[0]
    if patch_relevance.shape[0] != frames:
        raise RuntimeError(
            f"Expected {frames} frame maps, "
            f"got {patch_relevance.shape[0]}"
        )

    target_value = float(target.detach().item())
    token_sum = float(token_relevance.detach().sum().item())
    patch_sum = float(patch_relevance.detach().sum().item())
    cls_sum = float(cls_relevance.detach().sum().item())
    base_sum = (
        float((base * base.grad).detach().sum().item())
        if base.grad is not None
        else 0.0
    )
    attributed_sum = token_sum + base_sum
    absolute_error = abs(target_value - attributed_sum)
    relative_error = absolute_error / max(
        abs(target_value), STABILIZER
    )

    details = {
        "target": (
            "l2_norm_of_normalized_deterministic_action"
            if target_index is None
            else ACTION_KEYS[int(target_index)]
        ),
        "target_value": target_value,
        "normalized_action": native_action[0].detach().cpu().tolist(),
        "sum_all_token_relevance": token_sum,
        "sum_patch_relevance": patch_sum,
        "sum_cls_relevance": cls_sum,
        "sum_base_state_relevance": base_sum,
        "sum_attributed_roots": attributed_sum,
        "conservation_absolute_error": absolute_error,
        "conservation_relative_error": relative_error,
        "forward_equivalence_max_normalized_action_error": forward_error,
        "mamba_mixers_replaced": replacements["mamba_mixers"],
        "spatial_mamba_mixers_replaced": (
            replacements["spatial_mamba_mixers"]
        ),
        "temporal_mamba_mixers_replaced": (
            replacements["temporal_mamba_mixers"]
        ),
    }
    return (
        patch_relevance.detach().cpu().numpy().astype(np.float32),
        details,
    )


def compute_mambalrp(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
    *,
    evaluate_faithfulness: bool = True,
    mask_value: float = 255.0,
) -> MambaLRPResult:
    """Compute paper-style whole-policy and per-action signed relevance."""

    depth_np = _prepare_depth(depth_sequence)
    patch_relevance, policy_details = _single_target_relevance(
        agent,
        base_state,
        depth_np,
        target_index=None,
    )
    action_maps: list[np.ndarray] = []
    action_details: dict[str, dict] = {}
    action_count = int(agent.actor.action_dim)
    if action_count != len(ACTION_KEYS):
        raise RuntimeError(
            f"Expected {len(ACTION_KEYS)} actions, got {action_count}"
        )
    for index, key in enumerate(ACTION_KEYS):
        relevance, details = _single_target_relevance(
            agent,
            base_state,
            depth_np,
            target_index=index,
        )
        action_maps.append(relevance)
        action_details[key] = details
    action_patch_relevance = np.stack(action_maps, axis=0)

    display_relevance = _interpolate_patch_maps(
        patch_relevance,
        output_size=depth_np.shape[-2:],
    )
    action_display_relevance = np.stack(
        [
            _interpolate_patch_maps(
                action_patch_relevance[index],
                output_size=depth_np.shape[-2:],
            )
            for index in range(action_count)
        ],
        axis=0,
    )

    details = {
        "definition": (
            "Signed post-position-embedding patch relevance using "
            "Gradient x Input / LRP-0, MambaLRP SiLU, selective-SSM "
            "A/B/C detach, half-gate propagation, and generalized "
            "LRP-gamma for spatial Vision-Mamba Conv1d only"
        ),
        "paper_configuration": {
            "gamma": LRP_GAMMA,
            "gamma_layers": [
                "spatial_vim.conv1d",
                "spatial_vim.conv1d_b",
            ],
            "temporal_mamba_conv1d_rule": "LRP-0",
            "lrp_zero_layers": ["in_proj", "out_proj", "x_proj", "dt_proj"],
            "ssm_detached_quantities": ["discrete_A", "discrete_B", "C"],
            "multiplicative_gate_rule": "half_relevance",
            "relevance_root": "post_position_embedding_tokens",
            "signed_relevance": True,
        },
        "continuous_policy_adaptation": {
            "primary_target": (
                "L2 norm of the normalized deterministic action vector"
            ),
            "supplementary_targets": list(ACTION_KEYS),
            "reason": (
                "A continuous SAC policy has no predicted-class logit; "
                "all three action-coordinate explanations are saved."
            ),
        },
        "patch_grid": list(map(int, patch_relevance.shape[-2:])),
        "patch_size": list(
            map(int, agent.actor_encoder.vim.patch_embed.patch_size)
        ),
        "raw_relevance_shape": list(map(int, patch_relevance.shape)),
        "display_interpolation": {
            "method": "bilinear",
            "align_corners": False,
            "output_shape": list(map(int, display_relevance.shape)),
            "note": (
                "Interpolation is visualization-only and does not increase "
                "the raw patch relevance resolution."
            ),
        },
        "policy": policy_details,
        "actions": action_details,
    }
    if evaluate_faithfulness:
        details["patch_flipping_faithfulness"] = evaluate_patch_flipping(
            agent,
            base_state,
            depth_np,
            patch_relevance,
            mask_value=float(mask_value),
        )
    return MambaLRPResult(
        patch_relevance=patch_relevance,
        display_relevance=display_relevance,
        action_patch_relevance=action_patch_relevance,
        action_display_relevance=action_display_relevance,
        details=details,
    )


def _load_actor_for_evaluation(agent, checkpoint_path: str) -> None:
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


def _render_overlay(
    axis,
    depth: np.ndarray,
    relevance: np.ndarray,
    *,
    alpha: float,
) -> None:
    axis.imshow(depth, cmap="gray", vmin=0, vmax=255)
    axis.imshow(
        relevance,
        cmap="jet",
        norm=Normalize(vmin=-1.0, vmax=1.0),
        alpha=float(alpha),
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
    normalized = _normalize_signed_maps(record.result.display_relevance)
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
        axes[0, frame].imshow(depth[frame], cmap="gray", vmin=0, vmax=255)
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
    grid = "x".join(map(str, record.result.patch_relevance.shape[-2:]))
    figure.suptitle(
        "CL-VSSM-SAC paper-style MambaLRP "
        f"— step {record.sample.step} — raw grid {grid}"
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
    action_maps = record.result.action_display_relevance
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
        axes[0, frame].imshow(depth[frame], cmap="gray", vmin=0, vmax=255)
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
            record.result.display_relevance
        )[-1]
        axes[0, column].imshow(depth, cmap="gray", vmin=0, vmax=255)
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
    figure.suptitle("CL-VSSM-SAC paper-style MambaLRP summary")
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
        patch_relevance=result.patch_relevance.astype(np.float32),
        display_relevance=result.display_relevance.astype(np.float32),
        action_patch_relevance=(
            result.action_patch_relevance.astype(np.float32)
        ),
        action_display_relevance=(
            result.action_display_relevance.astype(np.float32)
        ),
    )


def _write_metadata(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            metadata, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")


def _default_checkpoint(model_seed: int) -> str:
    return str(
        REPO_ROOT
        / "models"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / "async_final.pth"
    )


def _default_output_dir(model_seed: int, episode_seed: int) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    return (
        REPO_ROOT
        / "results"
        / "explainability"
        / "mambalrp"
        / "test_scene"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / f"episode{int(episode_seed)}"
        / stamp
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate standalone paper-style MambaLRP explanations for "
            "CL-VSSM-SAC in the static test scene."
        )
    )
    parser.add_argument("--model_seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--episode_seed", type=int, default=DEFAULT_EPISODE_SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--capture_steps", type=int, nargs="+", default=None)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--min_sample_gap", type=int, default=DEFAULT_MIN_SAMPLE_GAP
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.58)
    parser.add_argument(
        "--mask_value",
        type=float,
        default=255.0,
        help=(
            "Depth value used by patch flipping; 255 is free space in this "
            "repository's depth encoding."
        ),
    )
    parser.add_argument(
        "--skip_faithfulness",
        action="store_true",
        help="Skip the paper-style MoRF/LeRF patch-flipping evaluation.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--self_test", action="store_true")
    script_args, remaining = parser.parse_known_args(argv)
    if script_args.num_samples <= 0 or script_args.dpi <= 0:
        parser.error("--num_samples and --dpi must be positive")
    if script_args.min_sample_gap < 0:
        parser.error("--min_sample_gap must be non-negative")
    if not 0.0 <= script_args.overlay_alpha <= 1.0:
        parser.error("--overlay_alpha must be in [0, 1]")
    if script_args.capture_steps is not None:
        script_args.capture_steps = sorted(set(script_args.capture_steps))
        if script_args.capture_steps[0] < 0:
            parser.error("--capture_steps must be non-negative")

    args = get_config(remaining)
    args.algorithm_name = "CL-VSSM-SAC"
    args.seed = int(script_args.model_seed)
    params_path = (
        REPO_ROOT / "algorithm" / "SB_PER_VSSM_SAC" / "params.yaml"
    )
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    params = params.get("params", params)
    if not isinstance(params, dict):
        raise ValueError(f"Algorithm params must be a mapping: {params_path}")
    args.algorithm_params = dict(params)
    for key, value in params.items():
        setattr(args, key, value)
    return script_args, args


def run_visualization(script_args, args) -> Path:
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError(
            "VSSM-SAC MambaLRP requires CUDA because the encoder uses "
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
                f"Expected depth {expected}, got {depth_shape}"
            )

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
        set_agent_eval_mode(agent)
        print(f"[MambaLRP] Loaded actor model: {checkpoint}")

        max_steps = int(getattr(args, "episode_length", 300))
        for step in range(max_steps):
            base = np.asarray(obs["base"], dtype=np.float32)
            depth = _prepare_depth(obs["depth"])
            action = _physical_action(agent, base, depth)
            trajectory.append(
                TrajectoryStep(
                    step=step,
                    base_state=base.copy(),
                    depth=depth.copy(),
                    physical_action=action.copy(),
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
                print(
                    f"[MambaLRP] Episode ended at step {step}: "
                    f"{termination}"
                )
                break
        if not trajectory:
            raise RuntimeError("The test episode produced no trajectory")

        if script_args.capture_steps is None:
            selected_indices = select_spaced_top_indices(
                [item.obstacle_proximity for item in trajectory],
                count=int(script_args.num_samples),
                min_gap=int(script_args.min_sample_gap),
            )
            selection_method = (
                "top_obstacle_proximity_with_temporal_spacing"
            )
            missing_steps: list[int] = []
        else:
            requested = set(script_args.capture_steps)
            selected_indices = [
                index
                for index, item in enumerate(trajectory)
                if item.step in requested
            ]
            completed = {
                trajectory[index].step for index in selected_indices
            }
            missing_steps = sorted(requested - completed)
            selection_method = "explicit_capture_steps"
        if not selected_indices:
            raise RuntimeError("No selected samples exist in the trajectory")

        selected_steps = [
            trajectory[index].step for index in selected_indices
        ]
        print(
            "[MambaLRP] Selected steps: "
            + ", ".join(map(str, selected_steps))
        )
        for index in selected_indices:
            sample = trajectory[index]
            print(f"[MambaLRP] Explaining step {sample.step}")
            result = compute_mambalrp(
                agent,
                sample.base_state,
                sample.depth,
                evaluate_faithfulness=not script_args.skip_faithfulness,
                mask_value=float(script_args.mask_value),
            )
            record = CaptureRecord(sample=sample, result=result)
            records.append(record)
            _save_record(record, output_dir)
            _render_four_frames(
                record,
                output_dir / f"step_{sample.step:04d}_four_frames.png",
                alpha=float(script_args.overlay_alpha),
                dpi=int(script_args.dpi),
            )
            _render_action_frames(
                record,
                output_dir / f"step_{sample.step:04d}_actions.png",
                alpha=float(script_args.overlay_alpha),
                dpi=int(script_args.dpi),
            )
        _render_summary(
            records,
            output_dir / "current_frame_summary.png",
            alpha=float(script_args.overlay_alpha),
            dpi=int(script_args.dpi),
        )

        metadata = {
            "algorithm": "CL-VSSM-SAC",
            "method": "MambaLRP",
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
            "gamma": LRP_GAMMA,
            "signed_relevance": True,
            "raw_relevance_unit": "post_position_embedding_patch_token",
            "display_colormap": "jet",
            "display_range": [-1.0, 1.0],
            "display_interpolation": "bilinear_visualization_only",
            "faithfulness_evaluation": (
                "disabled"
                if script_args.skip_faithfulness
                else "MoRF_and_LeRF_patch_flipping"
            ),
            "faithfulness_mask_value_depth_units": float(
                script_args.mask_value
            ),
            "action_labels": list(ACTION_LABELS),
            "method_details": {
                str(record.sample.step): record.result.details
                for record in records
            },
            "reference": {
                "paper": (
                    "Jafari et al., MambaLRP: Explaining Selective State "
                    "Space Sequence Models, NeurIPS 2024"
                ),
                "official_repository": (
                    "https://github.com/FarnoushRJ/MambaLRP"
                ),
                "vision_configuration": (
                    "Appendix C.2-C.2.1: generalized LRP-gamma with "
                    "gamma=0.25 on Vision-Mamba convolution layers only"
                ),
            },
        }
        _write_metadata(output_dir / "metadata.json", metadata)
        if missing_steps:
            print(
                "[MambaLRP] Missing explicit steps: "
                + ", ".join(map(str, missing_steps))
            )
        print(f"[MambaLRP] Results saved to: {output_dir}")
        return output_dir
    finally:
        close_env(env, label="CL-VSSM-SAC MambaLRP visualization")


def run_self_tests() -> None:
    """Paper-rule tests kept in this standalone script."""

    assert select_spaced_top_indices(
        [3.0, 2.0, 1.0], count=3, min_gap=10
    ) == [0, 1, 2]

    conv = nn.Conv1d(
        2, 2, kernel_size=2, padding=1, groups=2, bias=True
    )
    value = torch.randn(2, 2, 3, requires_grad=True)
    gamma_output = _mambalrp_gamma_conv1d(
        conv, value, gamma=LRP_GAMMA
    )
    native_output = conv(value)
    torch.testing.assert_close(
        gamma_output, native_output, rtol=1e-5, atol=1e-6
    )
    gamma_output.sum().backward()
    assert value.grad is not None
    assert torch.all(torch.isfinite(value.grad))

    token_relevance = torch.arange(
        17, dtype=torch.float32
    ).view(1, 17)
    patch_relevance, cls_relevance = _remove_middle_cls_relevance(
        token_relevance, grid_size=(4, 4)
    )
    assert patch_relevance.shape == (1, 4, 4)
    torch.testing.assert_close(cls_relevance, torch.tensor([8.0]))
    torch.testing.assert_close(
        patch_relevance.flatten(),
        torch.cat(
            [token_relevance[0, :8], token_relevance[0, 9:]]
        ),
    )

    signed = np.array([[[-2.0, 1.0], [0.0, 0.5]]], dtype=np.float32)
    normalized = _normalize_signed_maps(signed)
    np.testing.assert_allclose(normalized.min(), -1.0)
    np.testing.assert_allclose(normalized.max(), 0.5)

    toy_depth = np.zeros((1, 4, 4), dtype=np.float32)
    masked = _mask_ranked_patches(
        toy_depth,
        np.array([3, 0, 1, 2]),
        masked_count=1,
        grid_size=(2, 2),
        mask_value=7.0,
    )
    np.testing.assert_allclose(masked[0, 2:, 2:], 7.0)
    np.testing.assert_allclose(masked[0, :2, :2], 0.0)

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

    def native_branch(
        source,
        projected,
        *,
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
        discrete_A = torch.exp(
            torch.einsum("bld,dn->bldn", delta, A)
        )
        discrete_B = torch.einsum(
            "bld,bln->bldn", delta, B
        )
        state = torch.zeros(2, 2, 2)
        outputs = []
        for position in range(3):
            state = (
                discrete_A[:, position] * state
                + discrete_B[:, position]
                * values[:, position, :, None]
            )
            outputs.append(
                torch.einsum(
                    "bdn,bn->bd", state, C[:, position]
                )
            )
        scanned = torch.stack(outputs, dim=1) + values * D
        return scanned * F.silu(gate.transpose(1, 2))

    source = Mamba()
    sequence = torch.randn(2, 3, 2)
    with torch.no_grad():
        projected = source.in_proj(sequence).transpose(1, 2)
        expected = source.out_proj(
            native_branch(
                source,
                projected,
                conv1d=source.conv1d,
                x_proj=source.x_proj,
                dt_proj=source.dt_proj,
                A_log=source.A_log,
                D=source.D,
            )
        )
        actual = MambaLRPMixer(source)(sequence)
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
    bidirectional.D_b = nn.Parameter(
        bidirectional.D.detach().clone()
    )
    with torch.no_grad():
        projected = bidirectional.in_proj(sequence).transpose(1, 2)
        forward = native_branch(
            bidirectional,
            projected,
            conv1d=bidirectional.conv1d,
            x_proj=bidirectional.x_proj,
            dt_proj=bidirectional.dt_proj,
            A_log=bidirectional.A_log,
            D=bidirectional.D,
        )
        backward = native_branch(
            bidirectional,
            projected.flip(-1),
            conv1d=bidirectional.conv1d_b,
            x_proj=bidirectional.x_proj_b,
            dt_proj=bidirectional.dt_proj_b,
            A_log=bidirectional.A_b_log,
            D=bidirectional.D_b,
        )
        expected = bidirectional.out_proj(
            (forward + backward.flip(1)) / 2.0
        )
        actual = MambaLRPMixer(bidirectional)(sequence)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    print("All standalone MambaLRP tests passed.")


def main(argv=None) -> None:
    script_args, args = _parse_args(argv)
    if script_args.self_test:
        run_self_tests()
        return
    run_visualization(script_args, args)


if __name__ == "__main__":
    main()
