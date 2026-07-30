"""Standalone CPU tests for the MambaLRP package."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attribution import (
    MambaLRPResult,
    TrajectoryStep,
    _mask_ranked_patches,
    _normalize_signed_maps,
    _remove_middle_cls_relevance,
    _sum_pixel_relevance_by_patch,
    compute_mambalrp,
    select_spaced_top_indices,
)
from .rules import (
    LRP_GAMMA,
    MambaLRPActor,
    MambaLRPMixer,
    _mambalrp_gamma_conv1d,
    _mambalrp_gamma_conv2d,
    _mambalrp_layer_norm,
    _mambalrp_rms_norm,
    _paper_lrp_modules,
)


def run_self_tests() -> None:
    """Paper-rule tests kept in this standalone script."""

    torch.manual_seed(20260730)

    sample_depth = np.zeros((1, 2, 2), dtype=np.float32)
    sample = TrajectoryStep(
        step=3,
        base_state=np.array([0.1], dtype=np.float32),
        depth=sample_depth,
        physical_action=np.zeros(3, dtype=np.float32),
        obstacle_proximity=4.0,
    )
    sample_result = MambaLRPResult(
        pixel_relevance=sample_depth,
        patch_relevance=sample_depth,
        action_pixel_relevance=np.zeros((3, 1, 2, 2), dtype=np.float32),
        action_patch_relevance=np.zeros((3, 1, 2, 2), dtype=np.float32),
        details={},
    )
    assert sample.step == 3
    assert sample_result.display_relevance is sample_depth

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



if __name__ == "__main__":
    run_self_tests()
