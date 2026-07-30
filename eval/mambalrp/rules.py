"""MambaLRP propagation rules adapted to the repository's Mamba-1 modules."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


LRP_GAMMA = 0.25
STABILIZER = 1e-6


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
