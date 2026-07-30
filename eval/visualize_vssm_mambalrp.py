#!/usr/bin/env python3
"""Input-level MambaLRP adaptation for CL-VSSM-SAC.

This standalone script adapts Jafari et al. (NeurIPS 2024) to a continuous
VSSM-SAC policy.  It follows the authors' public Vision-Mamba implementation:

* relevance starts at the deterministic policy output;
* ordinary layers use the Gradient x Input / LRP-0 path;
* every Mamba mixer uses the paper's SiLU, selective-SSM and half-gate rules;
* spatial Mamba Conv1d and patch-embedding Conv2d use LRP-gamma (γ=0.25);
* normalization denominators are detached and residual additions are explicit;
* patch embedding uses the paper's generalized LRP-gamma convolution rule;
* relevance continues through patch embedding to the raw depth input;
* displayed maps retain the input depth resolution without interpolation.

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
import warnings
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
CONSERVATION_DIAGNOSTIC_RTOL = 1e-3
CONSERVATION_DIAGNOSTIC_ATOL = 1e-6
OFFICIAL_MAMBALRP_COMMIT = "b4462a5f6d55ec38a1251683f7ca0f4d2a576e98"
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


def _stabilize(value: torch.Tensor) -> torch.Tensor:
    """Paper-compatible denominator stabilizer."""

    return value + ((value == 0).to(value) + value.sign()) * STABILIZER


def _mambalrp_identity_activation(
    value: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Keep an activation's forward value and use the LRP identity backward."""

    surrogate = value * (output / _stabilize(value)).detach()
    return surrogate + (output - surrogate).detach()


def _mambalrp_silu(value: torch.Tensor) -> torch.Tensor:
    """Algorithm 1: SiLU with a relevance-conserving backward pass."""

    return _mambalrp_identity_activation(value, F.silu(value))


def _forward_value_with_surrogate(
    native: torch.Tensor,
    surrogate: torch.Tensor,
) -> torch.Tensor:
    """Use ``native`` in the forward pass and ``surrogate`` for propagation."""

    return surrogate + (native - surrogate).detach()


def _mambalrp_layer_norm(
    layer: nn.LayerNorm,
    value: torch.Tensor,
) -> torch.Tensor:
    """LayerNorm rule: keep centering linear and detach only the scale."""

    native = layer(value)
    centered = value - value.mean(
        dim=tuple(range(value.ndim - len(layer.normalized_shape), value.ndim)),
        keepdim=True,
    )
    variance = centered.square().mean(
        dim=tuple(range(value.ndim - len(layer.normalized_shape), value.ndim)),
        keepdim=True,
    )
    normalized = centered * torch.rsqrt(variance + layer.eps).detach()
    surrogate = normalized
    if layer.elementwise_affine:
        surrogate = surrogate * layer.weight
        if layer.bias is not None:
            surrogate = surrogate + layer.bias
    return _forward_value_with_surrogate(native, surrogate)


def _mambalrp_rms_norm(
    layer: nn.Module,
    value: torch.Tensor,
) -> torch.Tensor:
    """Official MambaLRP RMSNorm rule with a detached denominator."""

    native = layer(value)
    normalized = value * torch.rsqrt(
        value.square().mean(dim=-1, keepdim=True) + float(layer.eps)
    ).detach()
    surrogate = normalized * layer.weight
    bias = getattr(layer, "bias", None)
    if bias is not None:
        surrogate = surrogate + bias
    return _forward_value_with_surrogate(native, surrogate)


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
    surrogate = redistributed * (
        native / _stabilize(redistributed)
    ).detach()
    return _forward_value_with_surrogate(native, surrogate)


def _conv2d_with_parameters(
    layer: nn.Conv2d,
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.conv2d(
        value,
        weight,
        bias,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )


def _mambalrp_gamma_conv2d(
    layer: nn.Conv2d,
    value: torch.Tensor,
    *,
    gamma: float = LRP_GAMMA,
) -> torch.Tensor:
    """Official generalized LRP-gamma rule for Vim patch embedding."""

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
    positive_output = _conv2d_with_parameters(
        layer, positive_value, weight_positive, bias_positive
    ) + _conv2d_with_parameters(
        layer, negative_value, weight_negative, zero_bias
    )
    negative_output = _conv2d_with_parameters(
        layer, positive_value, weight_negative, bias_negative
    ) + _conv2d_with_parameters(
        layer, negative_value, weight_positive, zero_bias
    )
    redistributed = torch.where(
        native > STABILIZER,
        positive_output,
        torch.where(native < -STABILIZER, negative_output, native),
    )
    surrogate = redistributed * (
        native / _stabilize(redistributed)
    ).detach()
    return _forward_value_with_surrogate(native, surrogate)


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


class MambaLRPActor(nn.Module):
    """Forward-equivalent deterministic SAC actor with LRP propagation."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.source = source
        self.action_dim = int(source.action_dim)

    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if not deterministic:
            raise ValueError(
                "MambaLRPActor only supports deterministic policy outputs"
            )
        latent = _mambalrp_layer_norm(
            self.source.input_norm, observation
        )
        for layer in self.source.trunk:
            if isinstance(layer, (nn.SiLU, nn.ReLU)):
                latent = _mambalrp_identity_activation(
                    latent, layer(latent)
                )
            else:
                latent = layer(latent)
        mean = self.source.mean_linear(latent)
        return _mambalrp_identity_activation(mean, torch.tanh(mean))


class MambaLRPNormalization(nn.Module):
    """Forward-equivalent LayerNorm/RMSNorm propagation wrapper."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.source = source

    @property
    def weight(self):
        return self.source.weight

    @property
    def bias(self):
        return getattr(self.source, "bias", None)

    @property
    def eps(self):
        return self.source.eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if isinstance(self.source, nn.LayerNorm):
            return _mambalrp_layer_norm(self.source, value)
        return _mambalrp_rms_norm(self.source, value)


class MambaLRPGammaConv2d(nn.Module):
    """Forward-equivalent wrapper for the official patch-embedding rule."""

    def __init__(self, source: nn.Conv2d):
        super().__init__()
        self.source = source

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _mambalrp_gamma_conv2d(
            self.source, value, gamma=LRP_GAMMA
        )


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
        tuple[object, str, object, object]
    ] = []
    attributes: list[tuple[object, str, object]] = []
    spatial_count = 0
    temporal_count = 0
    normalization_count = 0
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

    vim = agent.actor_encoder.vim
    patch_projection = vim.patch_embed.proj
    if not isinstance(patch_projection, nn.Conv2d):
        raise TypeError("Vim patch embedding must use nn.Conv2d")
    replacements.append(
        (
            vim.patch_embed,
            "proj",
            patch_projection,
            MambaLRPGammaConv2d(patch_projection),
        )
    )

    for parent, name in [(vim, "norm_f")] + [
        (block, "norm") for block in vim.layers
    ]:
        source = getattr(parent, name)
        replacements.append(
            (
                parent,
                name,
                source,
                MambaLRPNormalization(source),
            )
        )
        normalization_count += 1
    for block in agent.actor_encoder.temporal_mamba.mamba_layers:
        source = block.norm
        replacements.append(
            (
                block,
                "norm",
                source,
                MambaLRPNormalization(source),
            )
        )
        normalization_count += 1

    source_actor = agent.actor
    replacements.append(
        (agent, "actor", source_actor, MambaLRPActor(source_actor))
    )
    attributes.append((vim, "fused_add_norm", vim.fused_add_norm))
    for block in vim.layers:
        attributes.append(
            (block, "fused_add_norm", block.fused_add_norm)
        )

    for parent, name, _source, replacement in replacements:
        setattr(parent, name, replacement)
    for module, name, _source in attributes:
        setattr(module, name, False)
    try:
        yield {
            "mamba_mixers": spatial_count + temporal_count,
            "spatial_mamba_mixers": spatial_count,
            "temporal_mamba_mixers": temporal_count,
            "normalization_layers": normalization_count,
            "patch_embedding_gamma_layers": 1,
            "actor_wrappers": 1,
        }
    finally:
        for module, name, source in reversed(attributes):
            setattr(module, name, source)
        for parent, name, source, _replacement in reversed(replacements):
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
            "raw_relevance_unit": "input_depth_pixel",
            "display_colormap": "jet",
            "display_range": [-1.0, 1.0],
            "display_interpolation": "none",
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
                "official_repository_commit": OFFICIAL_MAMBALRP_COMMIT,
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

    torch.manual_seed(20260730)

    assert select_spaced_top_indices(
        [3.0, 2.0, 1.0], count=3, min_gap=10
    ) == [0, 1, 2]

    norm_input = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0]], requires_grad=True
    )
    norm = nn.LayerNorm(4, elementwise_affine=True, bias=False)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.7, 1.1, -0.4, 0.9]))
    norm_output = _mambalrp_layer_norm(norm, norm_input)
    torch.testing.assert_close(norm_output, norm(norm_input))
    norm_target = (
        norm_output * torch.tensor([[0.2, -0.3, 0.8, 0.5]])
    ).sum()
    norm_target.backward()
    norm_input_relevance = (norm_input * norm_input.grad).sum()
    torch.testing.assert_close(
        norm_input_relevance, norm_target.detach(), rtol=1e-5, atol=1e-6
    )

    class ToyRMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.tensor([0.8, -0.5, 1.2, 0.4])
            )
            self.bias = None
            self.eps = 1e-5

        def forward(self, value):
            scale = torch.rsqrt(
                value.square().mean(dim=-1, keepdim=True) + self.eps
            )
            return value * scale * self.weight

    rms_input = torch.tensor(
        [[0.5, -1.5, 2.0, 0.25]], requires_grad=True
    )
    rms_norm = ToyRMSNorm()
    rms_output = _mambalrp_rms_norm(rms_norm, rms_input)
    torch.testing.assert_close(rms_output, rms_norm(rms_input))
    rms_target = (
        rms_output * torch.tensor([[0.4, 0.1, -0.6, 0.7]])
    ).sum()
    rms_target.backward()
    rms_input_relevance = (rms_input * rms_input.grad).sum()
    torch.testing.assert_close(
        rms_input_relevance, rms_target.detach(), rtol=1e-5, atol=1e-6
    )

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

    conv2d = nn.Conv2d(1, 3, kernel_size=2, stride=2, bias=True)
    image = torch.randn(2, 1, 4, 4, requires_grad=True)
    gamma_image_output = _mambalrp_gamma_conv2d(
        conv2d, image, gamma=LRP_GAMMA
    )
    torch.testing.assert_close(
        gamma_image_output, conv2d(image), rtol=1e-5, atol=1e-6
    )
    gamma_image_output.sum().backward()
    assert image.grad is not None
    assert torch.all(torch.isfinite(image.grad))

    pixels = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    pooled = _sum_pixel_relevance_by_patch(
        pixels, patch_size=(2, 2)
    )
    expected_pooled = np.array(
        [
            [[10.0, 18.0], [42.0, 50.0]],
            [[74.0, 82.0], [106.0, 114.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pooled, expected_pooled)
    np.testing.assert_allclose(pooled.sum(), pixels.sum())

    class ToyActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(
                4, elementwise_affine=True, bias=False
            )
            self.trunk = nn.Sequential(
                nn.Linear(4, 3, bias=False),
                nn.SiLU(),
                nn.Linear(3, 3, bias=False),
                nn.SiLU(),
            )
            self.mean_linear = nn.Linear(3, 2, bias=False)
            self.log_std_linear = nn.Linear(3, 2, bias=False)
            self.action_dim = 2

        def forward(self, observation, deterministic=False):
            latent = self.trunk(self.input_norm(observation))
            mean = self.mean_linear(latent)
            return torch.tanh(mean)

    toy_actor = ToyActor()
    actor_input = torch.tensor(
        [[0.2, -0.7, 1.4, 0.9]], requires_grad=True
    )
    explainable_actor = MambaLRPActor(toy_actor)
    actor_output = explainable_actor(actor_input, deterministic=True)
    torch.testing.assert_close(
        actor_output, toy_actor(actor_input.detach(), deterministic=True)
    )
    actor_target = actor_output[:, 1].sum()
    actor_target.backward()
    actor_input_relevance = (actor_input * actor_input.grad).sum()
    torch.testing.assert_close(
        actor_input_relevance, actor_target.detach(), rtol=2e-4, atol=1e-6
    )

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
            self.bimamba_type = "none"
            self.if_divide_out = False
            self.init_layer_scale = None

        def forward(self, hidden_states, inference_params=None):
            if inference_params is not None:
                raise ValueError("Toy Mamba does not use inference caches")
            return native_mamba_output(self, hidden_states)

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
        sequence_length = projected.shape[-1]
        values, gate = projected.chunk(2, dim=1)
        values = F.silu(
            conv1d(values)[..., :sequence_length]
        ).transpose(1, 2)
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
        state = torch.zeros(
            projected.shape[0],
            source.d_inner,
            source.d_state,
            dtype=values.dtype,
            device=values.device,
        )
        outputs = []
        for position in range(sequence_length):
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

    def native_mamba_output(source, hidden_states):
        projected = source.in_proj(hidden_states).transpose(1, 2)
        forward = native_branch(
            source,
            projected,
            conv1d=source.conv1d,
            x_proj=source.x_proj,
            dt_proj=source.dt_proj,
            A_log=source.A_log,
            D=source.D,
        )
        if source.bimamba_type == "v2":
            backward = native_branch(
                source,
                projected.flip(-1),
                conv1d=source.conv1d_b,
                x_proj=source.x_proj_b,
                dt_proj=source.dt_proj_b,
                A_log=source.A_b_log,
                D=source.D_b,
            )
            combined = forward + backward.flip(1)
            if source.if_divide_out:
                combined = combined / 2.0
        else:
            combined = forward
        return source.out_proj(combined)

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

    class TinyRMSNorm(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(size))
            self.bias = None
            self.eps = 1e-5

        def forward(self, value):
            return (
                value
                * torch.rsqrt(
                    value.square().mean(dim=-1, keepdim=True) + self.eps
                )
                * self.weight
            )

    def make_bidirectional_mamba():
        mixer = Mamba()
        mixer.bimamba_type = "v2"
        mixer.if_divide_out = True
        mixer.conv1d_b = copy.deepcopy(mixer.conv1d)
        mixer.x_proj_b = copy.deepcopy(mixer.x_proj)
        mixer.dt_proj_b = copy.deepcopy(mixer.dt_proj)
        mixer.A_b_log = nn.Parameter(mixer.A_log.detach().clone())
        mixer.D_b = nn.Parameter(mixer.D.detach().clone())
        return mixer

    class TinyPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_size = (4, 4)
            self.patch_size = (2, 2)
            self.grid_size = (2, 2)
            self.proj = nn.Conv2d(
                1, 2, kernel_size=2, stride=2, bias=False
            )

        def forward(self, image):
            return self.proj(image).flatten(2).transpose(1, 2)

    class TinyVimBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_in_fp32 = False
            self.fused_add_norm = True
            self.mixer = make_bidirectional_mamba()
            self.norm = TinyRMSNorm(2)
            self.drop_path = nn.Identity()

        def forward(
            self, hidden_states, residual=None, inference_params=None
        ):
            residual = (
                hidden_states
                if residual is None
                else residual + self.drop_path(hidden_states)
            )
            hidden_states = self.norm(residual)
            hidden_states = self.mixer(
                hidden_states, inference_params=inference_params
            )
            return hidden_states, residual

    class TinyVim(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = TinyPatchEmbed()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 2))
            self.pos_embed = nn.Parameter(torch.zeros(1, 5, 2))
            self.pos_drop = nn.Identity()
            self.layers = nn.ModuleList([TinyVimBlock()])
            self.norm_f = TinyRMSNorm(2)
            self.drop_path = nn.Identity()
            self.fused_add_norm = True

        def forward(self, image, return_features=False):
            tokens = self.patch_embed(image)
            middle = tokens.shape[1] // 2
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat(
                (tokens[:, :middle], cls, tokens[:, middle:]), dim=1
            )
            hidden = self.pos_drop(tokens + self.pos_embed)
            residual = None
            for layer in self.layers:
                hidden, residual = layer(hidden, residual)
            residual = hidden if residual is None else residual + hidden
            hidden = self.norm_f(residual)
            return hidden[:, middle]

    class TinyTemporalLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(2, bias=False)
            self.mamba = Mamba()

        def forward(self, sequence):
            return self.mamba(self.norm(sequence))

    class TinyTemporalStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba_layers = nn.ModuleList([TinyTemporalLayer()])

        def forward(self, sequence):
            for layer in self.mamba_layers:
                sequence = layer(sequence)
            return sequence

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.vim = TinyVim()
            self.temporal_mamba = TinyTemporalStack()

        def forward(self, depth_sequence):
            if depth_sequence.ndim == 4:
                depth_sequence = depth_sequence.unsqueeze(2)
            batch, frames, channels, height, width = depth_sequence.shape
            frame_features = self.vim(
                depth_sequence.reshape(
                    batch * frames, channels, height, width
                ),
                return_features=True,
            ).reshape(batch, frames, 2)
            return self.temporal_mamba(frame_features).reshape(batch, -1)

    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.LayerNorm(5, bias=False)
            self.trunk = nn.Sequential(
                nn.Linear(5, 4, bias=False),
                nn.SiLU(),
                nn.Linear(4, 4, bias=False),
                nn.SiLU(),
            )
            self.mean_linear = nn.Linear(4, 3, bias=False)
            self.log_std_linear = nn.Linear(4, 3, bias=False)
            self.action_dim = 3

        def forward(self, observation, deterministic=False):
            latent = self.trunk(self.input_norm(observation))
            return torch.tanh(self.mean_linear(latent))

    class TinyAgent:
        def __init__(self):
            self.device = torch.device("cpu")
            self.actor_encoder = TinyEncoder()
            self.actor = TinyPolicy()

        def _encode_state(self, base, depth, encoder):
            return torch.cat((base, encoder(depth)), dim=1)

    tiny_agent = TinyAgent()
    original_actor = tiny_agent.actor
    original_patch_projection = tiny_agent.actor_encoder.vim.patch_embed.proj
    original_spatial_mixers = tuple(
        block.mixer for block in tiny_agent.actor_encoder.vim.layers
    )
    original_spatial_norms = tuple(
        block.norm for block in tiny_agent.actor_encoder.vim.layers
    )
    original_final_norm = tiny_agent.actor_encoder.vim.norm_f
    original_temporal_mixers = tuple(
        block.mamba
        for block in tiny_agent.actor_encoder.temporal_mamba.mamba_layers
    )
    original_temporal_norms = tuple(
        block.norm
        for block in tiny_agent.actor_encoder.temporal_mamba.mamba_layers
    )
    original_fused_flags = (
        tiny_agent.actor_encoder.vim.fused_add_norm,
        tuple(
            block.fused_add_norm
            for block in tiny_agent.actor_encoder.vim.layers
        ),
    )

    def assert_tiny_context_restored():
        assert tiny_agent.actor is original_actor
        assert (
            tiny_agent.actor_encoder.vim.patch_embed.proj
            is original_patch_projection
        )
        assert tuple(
            block.mixer for block in tiny_agent.actor_encoder.vim.layers
        ) == original_spatial_mixers
        assert tuple(
            block.norm for block in tiny_agent.actor_encoder.vim.layers
        ) == original_spatial_norms
        assert tiny_agent.actor_encoder.vim.norm_f is original_final_norm
        assert tuple(
            block.mamba
            for block in tiny_agent.actor_encoder.temporal_mamba.mamba_layers
        ) == original_temporal_mixers
        assert tuple(
            block.norm
            for block in tiny_agent.actor_encoder.temporal_mamba.mamba_layers
        ) == original_temporal_norms
        assert (
            tiny_agent.actor_encoder.vim.fused_add_norm,
            tuple(
                block.fused_add_norm
                for block in tiny_agent.actor_encoder.vim.layers
            ),
        ) == original_fused_flags

    tiny_depth = np.linspace(
        0.2, 1.8, num=2 * 4 * 4, dtype=np.float32
    ).reshape(2, 4, 4)
    tiny_result = compute_mambalrp(
        tiny_agent,
        np.array([0.3], dtype=np.float32),
        tiny_depth,
        evaluate_faithfulness=False,
    )
    assert tiny_result.pixel_relevance.shape == (2, 4, 4)
    assert tiny_result.patch_relevance.shape == (2, 2, 2)
    assert tiny_result.action_pixel_relevance.shape == (3, 2, 4, 4)
    assert tiny_result.details["display_interpolation"]["method"] == "none"
    assert tiny_result.details["policy"]["conservation_numerically_close"], (
        tiny_result.details["policy"]
    )
    np.testing.assert_allclose(
        tiny_result.pixel_relevance.sum(),
        tiny_result.patch_relevance.sum(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert_tiny_context_restored()

    class ExpectedContextError(Exception):
        pass

    try:
        with _paper_lrp_modules(tiny_agent):
            assert isinstance(tiny_agent.actor, MambaLRPActor)
            assert not tiny_agent.actor_encoder.vim.fused_add_norm
            raise ExpectedContextError
    except ExpectedContextError:
        pass
    assert_tiny_context_restored()

    print("All standalone MambaLRP tests passed.")


def main(argv=None) -> None:
    script_args, args = _parse_args(argv)
    if script_args.self_test:
        run_self_tests()
        return
    run_visualization(script_args, args)


if __name__ == "__main__":
    main()
