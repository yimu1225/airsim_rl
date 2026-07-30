"""Policy targets, input-pixel relevance, and faithfulness diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .rules import LRP_GAMMA, STABILIZER, _paper_lrp_modules


CONSERVATION_DIAGNOSTIC_RTOL = 1e-3
CONSERVATION_DIAGNOSTIC_ATOL = 1e-6
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
    pixel_relevance: np.ndarray
    patch_relevance: np.ndarray
    action_pixel_relevance: np.ndarray
    action_patch_relevance: np.ndarray
    details: dict

    @property
    def display_relevance(self) -> np.ndarray:
        """Backward-compatible alias; values are raw pixels, not interpolated."""

        return self.pixel_relevance

    @property
    def action_display_relevance(self) -> np.ndarray:
        """Backward-compatible alias for raw per-action pixel relevance."""

        return self.action_pixel_relevance


@dataclass
class CaptureRecord:
    sample: TrajectoryStep
    result: MambaLRPResult


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


def _sum_pixel_relevance_by_patch(
    pixel_relevance: np.ndarray,
    *,
    patch_size: tuple[int, int],
) -> np.ndarray:
    """Aggregate raw input-pixel relevance into non-overlapping patches."""

    relevance = np.asarray(pixel_relevance, dtype=np.float32)
    if relevance.ndim != 3:
        raise ValueError("pixel_relevance must be (T,H,W)")
    patch_height, patch_width = map(int, patch_size)
    frames, height, width = relevance.shape
    if height % patch_height or width % patch_width:
        raise ValueError(
            "Input resolution must be divisible by the patch size"
        )
    return relevance.reshape(
        frames,
        height // patch_height,
        patch_height,
        width // patch_width,
        patch_width,
    ).sum(axis=(2, 4))


def _single_target_relevance(
    agent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
    *,
    target_index: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run one paper-style backward pass for one policy scalar."""

    base = torch.as_tensor(
        base_state, dtype=torch.float32, device=agent.device
    ).view(1, -1)
    base = base.detach().requires_grad_(True)
    depth_np = _prepare_depth(depth_sequence)
    depth = torch.as_tensor(
        depth_np, dtype=torch.float32, device=agent.device
    ).unsqueeze(0).detach().requires_grad_(True)

    with torch.no_grad():
        native_action = _normalized_action(
            agent, base.detach(), depth.detach()
        )

    captured_embeddings: list[torch.Tensor] = []

    def retain_post_position_embeddings(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Vim position embedding output must be a tensor")
        output.retain_grad()
        captured_embeddings.append(output)
        return output

    vim = agent.actor_encoder.vim
    if not hasattr(vim, "pos_drop"):
        raise RuntimeError(
            "Input-level Vision MambaLRP requires absolute position "
            "embeddings and vim.pos_drop"
        )
    hook = vim.pos_drop.register_forward_hook(
        retain_post_position_embeddings
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
    if depth.grad is None:
        raise RuntimeError("MambaLRP did not reach input depth pixels")

    token_relevance = (embeddings * embeddings.grad).sum(dim=-1)
    token_patch_relevance, cls_relevance = _remove_middle_cls_relevance(
        token_relevance,
        grid_size=tuple(vim.patch_embed.grid_size),
    )
    pixel_relevance = (
        depth * depth.grad
    )[0].detach().cpu().numpy().astype(np.float32)
    patch_size = tuple(map(int, vim.patch_embed.patch_size))
    patch_relevance = _sum_pixel_relevance_by_patch(
        pixel_relevance,
        patch_size=patch_size,
    )
    frames = depth_np.shape[0]
    if patch_relevance.shape[0] != frames:
        raise RuntimeError(
            f"Expected {frames} frame maps, "
            f"got {patch_relevance.shape[0]}"
        )
    if not np.all(np.isfinite(pixel_relevance)):
        raise RuntimeError("Input-pixel relevance contains non-finite values")

    target_value = float(target.detach().item())
    token_sum = float(token_relevance.detach().sum().item())
    token_patch_sum = float(
        token_patch_relevance.detach().sum().item()
    )
    patch_sum = float(patch_relevance.sum(dtype=np.float64))
    cls_sum = float(cls_relevance.detach().sum().item())
    pixel_sum = float(pixel_relevance.sum(dtype=np.float64))
    base_sum = (
        float((base * base.grad).detach().sum().item())
        if base.grad is not None
        else 0.0
    )
    position_sum = (
        float(
            (vim.pos_embed * vim.pos_embed.grad)
            .detach()
            .sum()
            .item()
        )
        if getattr(vim, "pos_embed", None) is not None
        and vim.pos_embed.grad is not None
        else 0.0
    )
    cls_parameter = getattr(vim, "cls_token", None)
    learned_cls_sum = (
        float(
            (cls_parameter * cls_parameter.grad)
            .detach()
            .sum()
            .item()
        )
        if cls_parameter is not None and cls_parameter.grad is not None
        else 0.0
    )
    variable_input_sum = pixel_sum + base_sum
    all_root_sum = (
        variable_input_sum + position_sum + learned_cls_sum
    )
    absolute_error = abs(target_value - all_root_sum)
    relative_error = absolute_error / max(
        abs(target_value), STABILIZER
    )
    numerical_tolerance = (
        CONSERVATION_DIAGNOSTIC_ATOL
        + CONSERVATION_DIAGNOSTIC_RTOL * abs(target_value)
    )

    details = {
        "target": (
            "l2_norm_of_normalized_deterministic_action"
            if target_index is None
            else ACTION_KEYS[int(target_index)]
        ),
        "target_value": target_value,
        "normalized_action": native_action[0].detach().cpu().tolist(),
        "sum_post_position_token_relevance": token_sum,
        "sum_post_position_patch_relevance": token_patch_sum,
        "sum_patch_relevance": patch_sum,
        "sum_input_depth_relevance": pixel_sum,
        "sum_cls_relevance": cls_sum,
        "sum_position_embedding_relevance": position_sum,
        "sum_learned_cls_parameter_relevance": learned_cls_sum,
        "sum_base_state_relevance": base_sum,
        "sum_variable_input_relevance": variable_input_sum,
        "sum_all_root_relevance": all_root_sum,
        "conservation_absolute_error": absolute_error,
        "conservation_relative_error": relative_error,
        "conservation_diagnostic_rtol": CONSERVATION_DIAGNOSTIC_RTOL,
        "conservation_diagnostic_atol": CONSERVATION_DIAGNOSTIC_ATOL,
        "conservation_numerically_close": (
            absolute_error <= numerical_tolerance
        ),
        "conservation_note": (
            "This is a numerical diagnostic, not a paper-defined "
            "acceptance threshold. Nonzero learned biases can retain "
            "relevance not assigned to observation inputs."
        ),
        "forward_equivalence_max_normalized_action_error": forward_error,
        "mamba_mixers_replaced": replacements["mamba_mixers"],
        "spatial_mamba_mixers_replaced": (
            replacements["spatial_mamba_mixers"]
        ),
        "temporal_mamba_mixers_replaced": (
            replacements["temporal_mamba_mixers"]
        ),
        "normalization_layers_replaced": (
            replacements["normalization_layers"]
        ),
        "patch_embedding_gamma_layers_replaced": (
            replacements["patch_embedding_gamma_layers"]
        ),
        "actor_wrappers_replaced": replacements["actor_wrappers"],
    }
    if not details["conservation_numerically_close"]:
        warnings.warn(
            "MambaLRP conservation diagnostic is not numerically close for "
            f"{details['target']}: target={target_value:.6g}, "
            f"all_roots={all_root_sum:.6g}, "
            f"relative_error={relative_error:.6g}. The relevance map is "
            "saved with these diagnostics and must not be presented as "
            "strictly conservative.",
            RuntimeWarning,
            stacklevel=2,
        )
    return (
        pixel_relevance,
        patch_relevance,
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
    pixel_relevance, patch_relevance, policy_details = (
        _single_target_relevance(
            agent,
            base_state,
            depth_np,
            target_index=None,
        )
    )
    action_pixel_maps: list[np.ndarray] = []
    action_patch_maps: list[np.ndarray] = []
    action_details: dict[str, dict] = {}
    action_count = int(agent.actor.action_dim)
    if action_count != len(ACTION_KEYS):
        raise RuntimeError(
            f"Expected {len(ACTION_KEYS)} actions, got {action_count}"
        )
    for index, key in enumerate(ACTION_KEYS):
        pixels, patches, details = _single_target_relevance(
            agent,
            base_state,
            depth_np,
            target_index=index,
        )
        action_pixel_maps.append(pixels)
        action_patch_maps.append(patches)
        action_details[key] = details
    action_pixel_relevance = np.stack(action_pixel_maps, axis=0)
    action_patch_relevance = np.stack(action_patch_maps, axis=0)

    details = {
        "definition": (
            "Signed input-depth-pixel relevance using Gradient x Input / "
            "LRP-0, detached-denominator LayerNorm/RMSNorm, explicit "
            "residual addition, MambaLRP SiLU/selective-SSM/half-gate "
            "rules, Actor identity-backward SiLU/tanh, and generalized "
            "LRP-gamma for Vision-Mamba Conv1d and patch Conv2d"
        ),
        "paper_configuration": {
            "gamma": LRP_GAMMA,
            "gamma_layers": [
                "spatial_vim.patch_embed.proj",
                "spatial_vim.conv1d",
                "spatial_vim.conv1d_b",
            ],
            "temporal_mamba_conv1d_rule": "LRP-0",
            "lrp_zero_layers": ["in_proj", "out_proj", "x_proj", "dt_proj"],
            "ssm_detached_quantities": ["discrete_A", "discrete_B", "C"],
            "multiplicative_gate_rule": "half_relevance",
            "normalization_rule": "denominator_detach",
            "residual_rule": "explicit_addition_LRP-0",
            "actor_activation_rule": "identity_backward_SiLU_ReLU_tanh",
            "relevance_root": "input_depth_pixels",
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
        "raw_pixel_relevance_shape": list(
            map(int, pixel_relevance.shape)
        ),
        "patch_relevance_shape": list(map(int, patch_relevance.shape)),
        "display_interpolation": {
            "method": "none",
            "output_shape": list(map(int, pixel_relevance.shape)),
            "note": (
                "The displayed map is raw input-pixel relevance at the "
                "model's actual depth resolution."
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
        pixel_relevance=pixel_relevance,
        patch_relevance=patch_relevance,
        action_pixel_relevance=action_pixel_relevance,
        action_patch_relevance=action_patch_relevance,
        details=details,
    )
