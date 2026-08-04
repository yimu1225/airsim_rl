#!/usr/bin/env python3
"""Generate paper-style LRP explanations for the convolutional SAC actor.

The script explains the deterministic actor mean, not a sampled SAC action and
not a critic Q-value.  Conventional LRP rules propagate signed relevance from
the policy output through the Actor and shared NatureCNN to all input pixels.

For a controlled comparison with VSSM-SAC MambaLRP, this script never flies a
second trajectory.  It reads the exact states and depth sequences selected from
one successful MambaLRP reference run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.SAC.agent import SACAgent
from algorithm.config_loader import apply_algorithm_params
from config import get_config
from eval.eval_common import resolve_checkpoint, set_agent_eval_mode


matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 13,
        "font.weight": "bold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

DEFAULT_MODEL_SEED = 25
DEFAULT_DPI = 600
LRP_GAMMA = 0.25
LRP_EPSILON = 1e-6
STABILIZER = 1e-6
DISPLAY_ABS_PERCENTILE = 99.0
ALGORITHM_NAME = "CL-SAC"
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
class LRPResult:
    policy_relevance: np.ndarray
    action_relevance: np.ndarray
    normalized_action: np.ndarray
    physical_action: np.ndarray
    details: dict


@dataclass
class CaptureRecord:
    sample: TrajectoryStep
    result: LRPResult


def _stabilize(value: torch.Tensor) -> torch.Tensor:
    return value + ((value == 0).to(value) + value.sign()) * STABILIZER


def _identity_activation(
    value: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Keep the native forward value and use the LRP identity backward."""

    surrogate = value * (output / _stabilize(value)).detach()
    return surrogate + (output - surrogate).detach()


def _forward_value_with_surrogate(
    native: torch.Tensor, surrogate: torch.Tensor
) -> torch.Tensor:
    return surrogate + (native - surrogate).detach()


def _lrp_layer_norm(
    layer: nn.LayerNorm, value: torch.Tensor
) -> torch.Tensor:
    """Detach the normalization scale while retaining centering."""

    native = layer(value)
    axes = tuple(
        range(value.ndim - len(layer.normalized_shape), value.ndim)
    )
    centered = value - value.mean(dim=axes, keepdim=True)
    variance = centered.square().mean(dim=axes, keepdim=True)
    surrogate = centered * torch.rsqrt(
        variance + layer.eps
    ).detach()
    if layer.elementwise_affine:
        surrogate = surrogate * layer.weight
        if layer.bias is not None:
            surrogate = surrogate + layer.bias
    return _forward_value_with_surrogate(native, surrogate)


def _gamma_parameters(
    parameter: torch.Tensor,
    *,
    gamma: float,
    positive: bool,
) -> torch.Tensor:
    selected = (
        parameter.clamp(min=0)
        if positive
        else parameter.clamp(max=0)
    )
    return parameter + float(gamma) * selected


def _conv2d(
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


def _lrp_gamma_conv2d(
    layer: nn.Conv2d,
    value: torch.Tensor,
    *,
    gamma: float = LRP_GAMMA,
) -> torch.Tensor:
    """Generalized LRP-gamma redistribution for signed activations."""

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
    positive_output = _conv2d(
        layer, positive_value, weight_positive, bias_positive
    ) + _conv2d(
        layer, negative_value, weight_negative, zero_bias
    )
    negative_output = _conv2d(
        layer, positive_value, weight_negative, bias_negative
    ) + _conv2d(
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


def _linear(
    layer: nn.Linear,
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.linear(value, weight, bias)


def _lrp_gamma_linear(
    layer: nn.Linear,
    value: torch.Tensor,
    *,
    gamma: float = LRP_GAMMA,
) -> torch.Tensor:
    """Generalized LRP-gamma redistribution for a linear layer."""

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
    positive_output = _linear(
        layer, positive_value, weight_positive, bias_positive
    ) + _linear(
        layer, negative_value, weight_negative, zero_bias
    )
    negative_output = _linear(
        layer, positive_value, weight_negative, bias_negative
    ) + _linear(
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


def _lrp_epsilon_linear(
    layer: nn.Linear,
    value: torch.Tensor,
    *,
    epsilon: float = LRP_EPSILON,
) -> torch.Tensor:
    """LRP-epsilon redistribution for a linear layer."""

    native = layer(value)
    denominator = native + (
        (native == 0).to(native) + native.sign()
    ) * float(epsilon)
    surrogate = native * (native / denominator).detach()
    return _forward_value_with_surrogate(native, surrogate)


class SACLRPEncoder(nn.Module):
    """Forward-equivalent NatureCNN with bounded-input LRP propagation."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.source = source
        self.repr_dim = int(source.repr_dim)
        self.input_channels = int(source.input_channels)
        self.frame_wise = bool(source.frame_wise)
        self.flatten_all_tokens = bool(source.flatten_all_tokens)
        self.sequence_length = int(source.sequence_length)
        self.single_frame_dim = int(source.single_frame_dim)
        self._zbox_inputs: list[torch.Tensor] = []
        self._zbox_roots: list[torch.Tensor] = []

    @property
    def zbox_roots(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._zbox_roots)

    @staticmethod
    def _conv_transpose_output_padding(
        layer: nn.Conv2d,
        output: torch.Tensor,
        input_value: torch.Tensor,
    ) -> tuple[int, int]:
        padding: list[int] = []
        for axis in range(2):
            base_size = (
                (int(output.shape[-2 + axis]) - 1)
                * int(layer.stride[axis])
                - 2 * int(layer.padding[axis])
                + int(layer.dilation[axis])
                * (int(layer.kernel_size[axis]) - 1)
                + 1
            )
            required = int(input_value.shape[-2 + axis]) - base_size
            if not 0 <= required < int(layer.stride[axis]):
                raise RuntimeError(
                    "Invalid zBox transposed-convolution output padding: "
                    f"axis={axis}, required={required}"
                )
            padding.append(required)
        return int(padding[0]), int(padding[1])

    def zbox_input_relevance(
        self,
        root_gradients: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, float]:
        """Propagate first-convolution relevance to bounded depth pixels."""

        if len(root_gradients) != len(self._zbox_roots):
            raise ValueError(
                "Expected one gradient per zBox root, got "
                f"{len(root_gradients)} for {len(self._zbox_roots)} roots"
            )
        layer = self.source.net.conv1
        weight_positive = layer.weight.clamp(min=0)
        weight_negative = layer.weight.clamp(max=0)
        input_relevances: list[torch.Tensor] = []
        bias_relevance = 0.0
        for input_value, root, root_gradient in zip(
            self._zbox_inputs, self._zbox_roots, root_gradients
        ):
            lower = torch.zeros_like(input_value)
            upper = torch.ones_like(input_value)
            relevance_out = root * root_gradient
            denominator = (
                root
                - _conv2d(layer, lower, weight_positive, None)
                - _conv2d(layer, upper, weight_negative, None)
            )
            redistribution = relevance_out / _stabilize(denominator)
            output_padding = self._conv_transpose_output_padding(
                layer, redistribution, input_value
            )

            def transpose(weight: torch.Tensor) -> torch.Tensor:
                return F.conv_transpose2d(
                    redistribution,
                    weight,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    output_padding=output_padding,
                    groups=layer.groups,
                    dilation=layer.dilation,
                )

            input_relevance = (
                input_value * transpose(layer.weight)
                - lower * transpose(weight_positive)
                - upper * transpose(weight_negative)
            )
            input_relevances.append(input_relevance)
            if layer.bias is not None:
                bias_relevance += float(
                    (
                        layer.bias.view(1, -1, 1, 1)
                        * redistribution
                    )
                    .detach()
                    .sum()
                    .item()
                )
        return torch.cat(input_relevances, dim=0), bias_relevance

    def _forward_nature_cnn(self, value: torch.Tensor) -> torch.Tensor:
        net = self.source.net
        native_first = net.conv1(value)
        self._zbox_inputs.append(value.detach())
        value = native_first.detach().requires_grad_(True)
        self._zbox_roots.append(value)
        value = _identity_activation(value, net.relu1(value))
        value = _lrp_gamma_conv2d(net.conv2, value)
        value = _identity_activation(value, net.relu2(value))
        value = _lrp_gamma_conv2d(net.conv3, value)
        value = _identity_activation(value, net.relu3(value))
        value = net.flatten(value)
        linear = net.linear[0]
        activation = net.linear[1]
        value = _lrp_epsilon_linear(linear, value)
        return _identity_activation(value, activation(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        if (
            value.numel() > 0
            and float(value.detach().max().item()) > 1.5
        ):
            value = value / 255.0
        value = value.clamp(0.0, 1.0)
        if self.frame_wise:
            sequence = self.source._prepare_frame_sequence(value)
            batch, frames, height, width = sequence.shape
            images = sequence.reshape(batch * frames, 1, height, width)
            features = self._forward_nature_cnn(images).view(
                batch, frames, self.single_frame_dim
            )
            if self.flatten_all_tokens:
                return features.reshape(batch, -1)
            return features[:, -1]
        if value.dim() == 2:
            value = value.unsqueeze(0).unsqueeze(0)
        elif value.dim() == 3:
            if (
                value.size(0) == self.input_channels
                and self.input_channels != 1
            ):
                value = value.unsqueeze(0)
            else:
                value = value.unsqueeze(1)
        return self._forward_nature_cnn(value)


class SACLRPActor(nn.Module):
    """Forward-equivalent deterministic SAC Actor with LRP propagation."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.source = source
        self.action_dim = int(source.action_dim)

    def forward(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        if not deterministic:
            raise ValueError("SAC-LRP only explains deterministic actions")
        latent = _lrp_layer_norm(self.source.input_norm, observation)
        for layer in self.source.trunk:
            if isinstance(layer, nn.Linear):
                latent = _lrp_epsilon_linear(layer, latent)
            elif isinstance(layer, nn.ReLU):
                before = (
                    latent.clone()
                    if bool(getattr(layer, "inplace", False))
                    else latent
                )
                latent = _identity_activation(before, layer(latent))
            else:
                latent = layer(latent)
        mean = _lrp_epsilon_linear(self.source.mean_linear, latent)
        return _identity_activation(mean, torch.tanh(mean))


@contextmanager
def _lrp_modules(agent: SACAgent) -> Iterator[SACLRPEncoder]:
    source_encoder = agent.actor_encoder
    source_actor = agent.actor
    lrp_encoder = SACLRPEncoder(source_encoder).to(agent.device)
    agent.actor_encoder = lrp_encoder
    agent.actor = SACLRPActor(source_actor).to(agent.device)
    try:
        yield lrp_encoder
    finally:
        agent.actor = source_actor
        agent.actor_encoder = source_encoder


def _configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _prepare_depth(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(
            f"Expected depth (T,H,W) or (T,1,H,W), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Depth contains non-finite values")
    return array


def _deterministic_normalized_action(
    agent: SACAgent,
    base: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    state = agent._concat_state(base, depth, agent.actor_encoder)
    if isinstance(agent.actor, SACLRPActor):
        return agent.actor(state, deterministic=True)
    mean, _ = agent.actor(state, compute_pi=False)
    return torch.tanh(mean)


def _policy_scalar(
    normalized_action: torch.Tensor,
    target_index: int | None,
) -> torch.Tensor:
    if target_index is None:
        return torch.linalg.vector_norm(
            normalized_action, ord=2, dim=1
        ).sum()
    return normalized_action[:, int(target_index)].sum()


def _unique_affine_bias_parameters(
    *roots: nn.Module,
) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for root in roots:
        for module in root.modules():
            bias = getattr(module, "bias", None)
            if (
                isinstance(bias, nn.Parameter)
                and bias.requires_grad
                and id(bias) not in seen
            ):
                parameters.append(bias)
                seen.add(id(bias))
    return parameters


def _single_target_lrp(
    agent: SACAgent,
    base_state: np.ndarray,
    depth_sequence: np.ndarray,
    *,
    target_index: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    base = torch.as_tensor(
        base_state, dtype=torch.float32, device=agent.device
    ).view(1, -1).detach().requires_grad_(True)
    depth_np = _prepare_depth(depth_sequence)
    depth = torch.as_tensor(
        depth_np, dtype=torch.float32, device=agent.device
    ).detach().requires_grad_(True)
    with torch.no_grad():
        native_action = _deterministic_normalized_action(
            agent, base.detach(), depth.detach()
        )

    with _lrp_modules(agent) as lrp_encoder:
        actor_input = agent._concat_state(
            base, depth, agent.actor_encoder
        )
        lrp_action = agent.actor(actor_input, deterministic=True)
        forward_error = float(
            (lrp_action.detach() - native_action).abs().max().item()
        )
        if forward_error > 1e-5:
            raise RuntimeError(
                "SAC-LRP forward-equivalence check failed: "
                f"max normalized action error={forward_error:.6g}"
            )
        target = _policy_scalar(lrp_action, target_index)
        bias_parameters = _unique_affine_bias_parameters(
            agent.actor_encoder, agent.actor
        )
        requested: list[torch.Tensor] = [
            base,
            actor_input,
            *lrp_encoder.zbox_roots,
            *bias_parameters,
        ]
        gradients = torch.autograd.grad(
            target, requested, allow_unused=True
        )
    gradient_by_id = {
        id(value): gradient
        for value, gradient in zip(requested, gradients)
    }
    base_gradient = gradient_by_id.get(id(base))
    actor_input_gradient = gradient_by_id.get(id(actor_input))
    if base_gradient is None or actor_input_gradient is None:
        raise RuntimeError("SAC-LRP did not reach all Actor inputs")
    root_gradients = tuple(
        gradient_by_id.get(id(root)) for root in lrp_encoder.zbox_roots
    )
    if not root_gradients or any(
        gradient is None for gradient in root_gradients
    ):
        raise RuntimeError(
            "SAC-LRP did not reach every first-convolution zBox root"
        )
    pixel_relevance, first_conv_bias_sum = (
        lrp_encoder.zbox_input_relevance(root_gradients)
    )

    if pixel_relevance.numel() != int(depth_np.size):
        raise ValueError(
            "Expected zBox input relevance with "
            f"{depth_np.size} values, got {pixel_relevance.numel()}"
        )
    relevance = (
        pixel_relevance.detach()
        .reshape(depth_np.shape)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if not np.all(np.isfinite(relevance)):
        raise RuntimeError("SAC-LRP produced non-finite relevance")

    target_value = float(target.detach().item())
    pixel_sum = float(relevance.sum(dtype=np.float64))
    base_sum = float((base * base_gradient).detach().sum().item())
    actor_input_sum = float(
        (actor_input * actor_input_gradient).detach().sum().item()
    )
    downstream_bias_sum = float(
        sum(
            (parameter * gradient_by_id[id(parameter)])
            .detach()
            .sum()
            .item()
            for parameter in bias_parameters
            if gradient_by_id.get(id(parameter)) is not None
        )
    )
    bias_sum = downstream_bias_sum + first_conv_bias_sum
    accounted_sum = pixel_sum + base_sum + bias_sum
    return (
        relevance,
        native_action[0].detach().cpu().numpy().astype(np.float32),
        {
            "target": (
                "l2_norm_of_normalized_deterministic_action"
                if target_index is None
                else ACTION_KEYS[int(target_index)]
            ),
            "target_value": target_value,
            "normalized_action": (
                native_action[0].detach().cpu().tolist()
            ),
            "sum_input_depth_relevance": pixel_sum,
            "sum_base_state_relevance": base_sum,
            "sum_actor_input_relevance": actor_input_sum,
            "sum_non_attributable_affine_bias_relevance": bias_sum,
            "sum_all_accounted_relevance": accounted_sum,
            "accounting_absolute_error": abs(
                target_value - accounted_sum
            ),
            "forward_equivalence_max_normalized_action_error": (
                forward_error
            ),
            "first_conv_rule": "zBox",
            "first_conv_input_bounds": [0.0, 1.0],
            "intermediate_conv_rule": "generalized_LRP_gamma",
            "linear_rule": "LRP_epsilon",
            "gamma": LRP_GAMMA,
            "epsilon": LRP_EPSILON,
            "activation_rule": "identity",
            "layer_norm_rule": "detached_scale",
            "affine_bias_handling": (
                "non_attributable_accounting_only"
            ),
        },
    )


def compute_lrp(
    agent: SACAgent,
    base_state: np.ndarray,
    depth: np.ndarray,
) -> LRPResult:
    policy_relevance, normalized_action, policy_details = (
        _single_target_lrp(
            agent, base_state, depth, target_index=None
        )
    )
    action_maps: list[np.ndarray] = []
    action_details: dict[str, dict] = {}
    for action_index, action_key in enumerate(ACTION_KEYS):
        relevance, repeated_action, details = _single_target_lrp(
            agent,
            base_state,
            depth,
            target_index=action_index,
        )
        if not np.allclose(
            repeated_action,
            normalized_action,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Deterministic Actor output changed between LRP targets"
            )
        action_maps.append(relevance)
        action_details[action_key] = details
    return LRPResult(
        policy_relevance=policy_relevance,
        action_relevance=np.stack(action_maps, axis=0).astype(
            np.float32
        ),
        normalized_action=normalized_action,
        physical_action=(
            agent.action_scale
            * torch.as_tensor(
                normalized_action,
                dtype=torch.float32,
                device=agent.device,
            )
            + agent.action_bias
        ).detach().cpu().numpy().astype(np.float32),
        details={
            "policy": policy_details,
            "actions": action_details,
        },
    )


def _normalize_signed_maps(maps: np.ndarray) -> np.ndarray:
    array = np.asarray(maps, dtype=np.float32)
    scale = float(
        np.percentile(np.abs(array), DISPLAY_ABS_PERCENTILE)
    )
    if scale <= np.finfo(np.float32).eps:
        return np.zeros_like(array)
    return np.clip(array / scale, -1.0, 1.0)


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
        interpolation="bicubic",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _add_horizontal_colorbar(figure, axes) -> None:
    scalar = ScalarMappable(
        norm=Normalize(vmin=-1.0, vmax=1.0), cmap="jet"
    )
    scalar.set_array([])
    colorbar = figure.colorbar(
        scalar,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        location="bottom",
        shrink=0.55,
        pad=0.03,
        aspect=45,
    )
    for tick_label in colorbar.ax.get_xticklabels():
        tick_label.set_fontfamily("Times New Roman")
        tick_label.set_fontsize(11)
        tick_label.set_fontweight("bold")


def _render_four_frames(
    record: CaptureRecord,
    output_path: Path,
    *,
    alpha: float,
    dpi: int,
) -> None:
    depth = record.sample.depth
    relevance = _normalize_signed_maps(
        record.result.policy_relevance
    )
    frames = depth.shape[0]
    figure, axes = plt.subplots(
        2,
        frames,
        figsize=(2.60 * frames, 5.7),
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
            relevance[frame],
            alpha=alpha,
        )
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("LRP")
    _add_horizontal_colorbar(figure, axes)
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
    frames = depth.shape[0]
    figure, axes = plt.subplots(
        len(ACTION_LABELS) + 1,
        frames,
        figsize=(2.60 * frames, 2.65 * (len(ACTION_LABELS) + 1)),
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
        relevance = _normalize_signed_maps(
            record.result.action_relevance[action_index]
        )
        row = action_index + 1
        for frame in range(frames):
            _render_overlay(
                axes[row, frame],
                depth[frame],
                relevance[frame],
                alpha=alpha,
            )
        axes[row, 0].set_ylabel(label)
    _add_horizontal_colorbar(figure, axes)
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
        figsize=(2.60 * len(records), 5.7),
        squeeze=False,
        constrained_layout=True,
    )
    for column, record in enumerate(records):
        depth = record.sample.depth[-1]
        relevance = _normalize_signed_maps(
            record.result.policy_relevance
        )[-1]
        axes[0, column].imshow(
            depth,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"Step {record.sample.step}")
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        _render_overlay(
            axes[1, column], depth, relevance, alpha=alpha
        )
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("LRP")
    _add_horizontal_colorbar(figure, axes)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _save_record(record: CaptureRecord, output_dir: Path) -> None:
    sample = record.sample
    result = record.result
    np.savez_compressed(
        output_dir / f"step_{sample.step:04d}_lrp.npz",
        step=np.int32(sample.step),
        base_state=sample.base_state.astype(np.float32),
        depth=sample.depth.astype(np.float32),
        reference_physical_action=sample.physical_action.astype(np.float32),
        sac_physical_action=result.physical_action.astype(np.float32),
        normalized_action=result.normalized_action.astype(np.float32),
        obstacle_proximity=np.float32(sample.obstacle_proximity),
        policy_relevance=result.policy_relevance.astype(np.float32),
        action_relevance=result.action_relevance.astype(np.float32),
    )


def _write_metadata(path: Path, metadata: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            metadata, handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")


def _load_actor_for_evaluation(
    agent: SACAgent, checkpoint_path: str
) -> None:
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


def _default_checkpoint(model_seed: int) -> str:
    return str(
        REPO_ROOT
        / "models"
        / ALGORITHM_NAME
        / f"seed{int(model_seed)}"
        / "async_final.pth"
    )


def _default_output_dir(
    model_seed: int, reference_run: Path
) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    return (
        REPO_ROOT
        / "results"
        / "explainability"
        / "lrp"
        / "paired_reference"
        / ALGORITHM_NAME
        / f"seed{int(model_seed)}"
        / reference_run.name
        / stamp
    )


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
            "No successful CL-VSSM-SAC MambaLRP reference run was found; "
            "pass --reference_run explicitly."
        )
    return sorted(candidates)[-1]


def _load_reference_samples(
    reference_run: Path,
    requested_steps: Sequence[int] | None,
) -> tuple[list[TrajectoryStep], dict]:
    metadata_path = reference_run / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Reference metadata does not exist: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("termination") != "success":
        raise ValueError(
            "Paired explanations require a successful reference trajectory; "
            f"got termination={metadata.get('termination')!r}"
        )
    if metadata.get("algorithm") != "CL-VSSM-SAC":
        raise ValueError(
            "Reference run must come from CL-VSSM-SAC MambaLRP, got "
            f"{metadata.get('algorithm')!r}"
        )
    completed = [
        int(value)
        for value in metadata.get("capture_steps_completed", [])
    ]
    steps = (
        sorted(set(map(int, requested_steps)))
        if requested_steps is not None
        else completed
    )
    if not steps:
        raise ValueError("Reference run contains no selected steps")
    missing = sorted(set(steps) - set(completed))
    if missing:
        raise ValueError(
            "Requested steps are not present in the reference run: "
            + ", ".join(map(str, missing))
        )

    samples: list[TrajectoryStep] = []
    for step in steps:
        path = reference_run / f"step_{step:04d}_mambalrp.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference sample does not exist: {path}"
            )
        with np.load(path, allow_pickle=False) as data:
            samples.append(
                TrajectoryStep(
                    step=int(data["step"]),
                    base_state=np.asarray(
                        data["base_state"], dtype=np.float32
                    ),
                    depth=_prepare_depth(data["depth"]).copy(),
                    physical_action=np.asarray(
                        data["original_physical_action"],
                        dtype=np.float32,
                    ),
                    obstacle_proximity=float(data["obstacle_proximity"]),
                )
            )
    return samples, metadata


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate signed LRP explanations for CL-SAC from the exact "
            "samples of one successful CL-VSSM-SAC MambaLRP trajectory."
        )
    )
    parser.add_argument("--model_seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--reference_run",
        type=str,
        default=None,
        help=(
            "Successful CL-VSSM-SAC MambaLRP run directory. If omitted, "
            "the latest successful run is used."
        ),
    )
    parser.add_argument("--capture_steps", type=int, nargs="+", default=None)
    parser.add_argument("--overlay_alpha", type=float, default=0.58)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--output_dir", type=str, default=None)
    script_args, remaining = parser.parse_known_args(argv)
    if script_args.dpi <= 0:
        parser.error("--dpi must be positive")
    if not 0.0 <= script_args.overlay_alpha <= 1.0:
        parser.error("--overlay_alpha must be in [0, 1]")
    if script_args.capture_steps is not None:
        script_args.capture_steps = sorted(set(script_args.capture_steps))
        if script_args.capture_steps[0] < 0:
            parser.error("--capture_steps must be non-negative")

    args = get_config(remaining)
    args.algorithm_name = ALGORITHM_NAME
    args.seed = int(script_args.model_seed)
    apply_algorithm_params(args, ALGORITHM_NAME)
    return script_args, args


def run_visualization(script_args, args) -> Path:
    model_seed = int(script_args.model_seed)
    _configure_reproducibility(
        model_seed, bool(getattr(args, "cuda_deterministic", True))
    )
    checkpoint = resolve_checkpoint(
        script_args.checkpoint, _default_checkpoint(model_seed)
    )
    reference_run = (
        Path(script_args.reference_run).resolve()
        if script_args.reference_run
        else _latest_successful_reference_run()
    )
    samples, reference_metadata = _load_reference_samples(
        reference_run, script_args.capture_steps
    )
    output_dir = (
        Path(script_args.output_dir).resolve()
        if script_args.output_dir
        else _default_output_dir(model_seed, reference_run)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    first = samples[0]
    depth_shape = tuple(map(int, first.depth.shape))
    if len(depth_shape) != 3:
        raise ValueError(
            f"Expected reference depth (T,H,W), got {depth_shape}"
        )
    action_space = spaces.Box(
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
    device = torch.device(
        "cuda"
        if bool(args.cuda) and torch.cuda.is_available()
        else "cpu"
    )
    agent = SACAgent(
        first.base_state.shape[0],
        depth_shape,
        action_space,
        args,
        device=device,
        seed=model_seed,
    )
    _load_actor_for_evaluation(agent, checkpoint)
    set_agent_eval_mode(agent)
    print(f"[SAC-LRP] Loaded Actor model: {checkpoint}")
    print(f"[SAC-LRP] Fixed successful reference: {reference_run}")

    records: list[CaptureRecord] = []
    for sample in samples:
        print(f"[SAC-LRP] Explaining reference step {sample.step}")
        result = compute_lrp(
            agent, sample.base_state, sample.depth
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

    selected_steps = [sample.step for sample in samples]
    metadata = {
        "algorithm": ALGORITHM_NAME,
        "method": "LRP",
        "model_seed": model_seed,
        "checkpoint": os.path.abspath(checkpoint),
        "reference_run": str(reference_run.resolve()),
        "reference_algorithm": reference_metadata.get("algorithm"),
        "reference_method": reference_metadata.get("method"),
        "reference_termination": reference_metadata.get("termination"),
        "reference_model_seed": reference_metadata.get("model_seed"),
        "reference_environment_seed": reference_metadata.get(
            "environment_seed"
        ),
        "reference_episode_seed": reference_metadata.get("episode_seed"),
        "capture_steps_completed": selected_steps,
        "paired_inputs": True,
        "paired_input_fields": ["base_state", "depth"],
        "policy_target": (
            "l2_norm_of_normalized_deterministic_action"
        ),
        "action_labels": list(ACTION_LABELS),
        "gamma": LRP_GAMMA,
        "epsilon": LRP_EPSILON,
        "first_conv_rule": "zBox",
        "first_conv_input_bounds": [0.0, 1.0],
        "intermediate_conv_rule": "generalized_LRP_gamma",
        "linear_rule": "LRP_epsilon",
        "signed_relevance": True,
        "raw_relevance_unit": "input_depth_pixel",
        "display_colormap": "jet",
        "display_range": [-1.0, 1.0],
        "display_absolute_percentile_clip": DISPLAY_ABS_PERCENTILE,
        "display_normalization": (
            "whole_policy_across_four_frames; "
            "each_action_independently_across_four_frames"
        ),
        "method_details": {
            str(record.sample.step): record.result.details
            for record in records
        },
        "references": {
            "original_lrp": (
                "Bach et al., On Pixel-Wise Explanations for Non-Linear "
                "Classifier Decisions by Layer-Wise Relevance Propagation, "
                "PLOS ONE, 2015"
            ),
            "lrp_overview": (
                "Montavon et al., Layer-Wise Relevance Propagation: "
                "An Overview, Explainable AI, 2019"
            ),
        },
    }
    _write_metadata(output_dir / "metadata.json", metadata)
    print(f"[SAC-LRP] Results saved to: {output_dir}")
    return output_dir


def main(argv=None) -> None:
    script_args, args = _parse_args(argv)
    run_visualization(script_args, args)


if __name__ == "__main__":
    main()
