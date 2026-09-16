"""All-Mamba perception, causal memory, and patch decoder modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch import nn


MambaFactory = Callable[..., nn.Module]


def _mamba_factory(**kwargs: Any) -> nn.Module:
    # Lazy import keeps dataset/replay/CLI help usable on machines without a
    # working CUDA/Triton runtime.
    from mamba_ssm import Mamba

    return Mamba(**kwargs)


class MambaState(NamedTuple):
    """Internal recurrent state, carried to the next frame, not decoder input."""
    conv: torch.Tensor
    ssm: torch.Tensor


class _MambaResidual(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        mamba_factory: MambaFactory,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = mamba_factory(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.mixer(self.norm(values))

    def allocate_cache(self, batch_size: int, dtype: torch.dtype) -> MambaState:
        conv, ssm = self.mixer.allocate_inference_cache(
            batch_size=batch_size, max_seqlen=1, dtype=dtype
        )
        return MambaState(conv, ssm)

    def step(
        self, values: torch.Tensor, state: MambaState
    ) -> tuple[torch.Tensor, MambaState]:
        mixed, conv, ssm = self.mixer.step(
            self.norm(values), state.conv, state.ssm
        )
        return values + mixed, MambaState(conv, ssm)


class TemporalMambaMemory(nn.Module):
    """Causal Mamba stack with equivalent sequence and one-step interfaces."""

    def __init__(
        self,
        latent_dim: int,
        memory_dim: int,
        *,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_factory: MambaFactory = _mamba_factory,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.RMSNorm(memory_dim),
        )
        self.layers = nn.ModuleList(
            _MambaResidual(
                memory_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mamba_factory=mamba_factory,
            )
            for _ in range(depth)
        )
        self.output_norm = nn.RMSNorm(memory_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return y_t for every causal prefix, not the internal recurrent states.

        Each layer scans the sequence from zero state. This is the differentiable
        training path; do not replace it with the inference cache update loop.
        """
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, time, latent_dim]")
        values = self.input_projection(latents)
        for layer in self.layers:
            values = layer(values)
        return self.output_norm(values)

    def forward_sequence(
        self,
        latents: torch.Tensor,
        episode_start_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal batched forward, resetting at any marked episode boundary."""
        if episode_start_mask is None:
            return self(latents)
        if episode_start_mask.shape != latents.shape[:2]:
            raise ValueError("episode_start_mask must have shape [batch, time]")
        if not episode_start_mask[:, 1:].any():
            return self(latents)
        sequences = []
        for batch_index in range(latents.shape[0]):
            starts = torch.nonzero(
                episode_start_mask[batch_index], as_tuple=False
            ).flatten().tolist()
            starts = sorted(set([0, *starts, latents.shape[1]]))
            segments = [
                self(latents[batch_index : batch_index + 1, begin:end])
                for begin, end in zip(starts[:-1], starts[1:])
                if begin < end
            ]
            sequences.append(torch.cat(segments, dim=1))
        return torch.cat(sequences, dim=0)

    def reset(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[MambaState]:
        parameter = next(self.parameters())
        requested_device = torch.device(device) if device is not None else parameter.device
        if parameter.device != requested_device:
            raise ValueError(
                f"memory is on {parameter.device}, but cache was requested on {requested_device}"
            )
        cache_dtype = dtype or parameter.dtype
        return [layer.allocate_cache(batch_size, cache_dtype) for layer in self.layers]

    def step(
        self, latent: torch.Tensor, cache: list[MambaState]
    ) -> tuple[torch.Tensor, list[MambaState]]:
        """Consume (x_t, s_previous) and return (y_t, s_updated).

        Use y_t for reconstruction/policy features. Carry s_updated to the next
        frame and reset it at episode boundaries. Mixer caches may update in place.
        """
        if latent.ndim != 2:
            raise ValueError("latent must have shape [batch, latent_dim]")
        if len(cache) != len(self.layers):
            raise ValueError("cache does not match the temporal Mamba depth")
        values = self.input_projection(latent).unsqueeze(1)
        next_cache: list[MambaState] = []
        for layer, state in zip(self.layers, cache):
            values, state = layer.step(values, state)
            next_cache.append(state)
        return self.output_norm(values).squeeze(1), next_cache


class VisionMambaEncoder(nn.Module):
    """Single-frame Vision Mamba encoder returning one latent vector."""

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        channels: int = 1,
        patch_size: int = 8,
        latent_dim: int = 128,
        depth: int = 4,
        d_state: int = 16,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        vision_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if vision_model is None:
            from Vim.vim.models_mamba import VisionMamba

            vision_model = VisionMamba(
                img_size=image_size,
                patch_size=patch_size,
                stride=patch_size,
                depth=depth,
                embed_dim=latent_dim,
                d_state=d_state,
                channels=channels,
                num_classes=0,
                if_bidirectional=False,
                if_abs_pos_embed=True,
                if_rope=False,
                if_rope_residual=False,
                fused_add_norm=True,
                residual_in_fp32=True,
                if_cls_token=True,
                use_middle_cls_token=True,
                final_pool_type="none",
                if_bimamba=True,
                bimamba_type="v2",
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
        self.vim = vision_model

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError("frames must have shape [batch, channels, height, width]")
        features = self.vim(frames, return_features=True)
        if isinstance(features, tuple):
            features = features[0]
        if features.ndim == 3:
            features = features.mean(dim=1)
        return features


class _BidirectionalMambaResidual(nn.Module):
    """Spatial token mixer; both scan directions see the complete patch grid."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        mamba_factory: MambaFactory,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        kwargs = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.forward_mixer = mamba_factory(**kwargs)
        self.backward_mixer = mamba_factory(**kwargs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values)
        forward = self.forward_mixer(normalized)
        backward = self.backward_mixer(normalized.flip(1)).flip(1)
        return values + 0.5 * (forward + backward)


class _LinearPatchExpand(nn.Module):
    """Produce four spatially arranged child tokens from each parent token."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, 4 * dim)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, dim = tokens.shape
        children = self.projection(tokens).reshape(batch, height, width, 2, 2, dim)
        return children.permute(0, 1, 3, 2, 4, 5).reshape(batch, 4 * height * width, dim)


class VisionMambaDecoder(nn.Module):
    """Decode a global latent through a coarse-to-fine spatial Mamba pyramid.

    The only image projection is a linear token-to-patch layer followed by a
    reshape (unpatchify); no convolution or transposed convolution is used.
    """

    def __init__(
        self,
        input_dim: int,
        image_size: tuple[int, int],
        *,
        patch_size: int = 8,
        channels: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_factory: MambaFactory = _mamba_factory,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.grid_size = (height // patch_size, width // patch_size)
        # Halve exactly down to a 4×4 coarse grid; unusual/non-power-of-two
        # grids retain their dimensions.
        grids = [self.grid_size]
        while min(grids[0]) > 4 and all(size % 2 == 0 for size in grids[0]):
            grids.insert(0, tuple(size // 2 for size in grids[0]))
        self.stage_grids = tuple(grids)
        self.coarse_grid = grids[0]
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(input_dim, grids[0][0] * grids[0][1] * embed_dim)
        self.position_embeddings = nn.ParameterList([
            nn.Parameter(torch.empty(1, h * w, embed_dim)) for h, w in grids
        ])
        # depth is the number of Mamba blocks at each spatial resolution.
        self.stages = nn.ModuleList([
            nn.Sequential(*[_BidirectionalMambaResidual(
                embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mamba_factory=mamba_factory,
            ) for _ in range(depth)]) for _ in grids
        ])
        self.upsamplers = nn.ModuleList([
            _LinearPatchExpand(embed_dim) for _ in grids[1:]
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_projection = nn.Linear(
            embed_dim, channels * patch_size * patch_size
        )
        for position in self.position_embeddings:
            nn.init.trunc_normal_(position, std=0.02)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        sequence_input = latent.ndim == 3
        if latent.ndim not in (2, 3):
            raise ValueError("latent must have shape [B,D] or [B,T,D]")
        prefix = latent.shape[:-1]
        flat = latent.reshape(-1, latent.shape[-1])
        tokens = self.input_projection(flat).reshape(flat.shape[0], -1, self.embed_dim)
        for index, stage in enumerate(self.stages):
            if index:
                tokens = self.upsamplers[index - 1](tokens, *self.stage_grids[index - 1])
            tokens = stage(tokens + self.position_embeddings[index])
        patches = self.patch_projection(self.norm(tokens))
        images = torch.sigmoid(self.unpatchify(patches))
        if sequence_input:
            return images.reshape(*prefix, *images.shape[1:])
        return images

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        grid_h, grid_w = self.grid_size
        patch = self.patch_size
        values = patches.reshape(batch, grid_h, grid_w, self.channels, patch, patch)
        return values.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.channels, grid_h * patch, grid_w * patch
        )


class MultiFrameMambaReconstructor(nn.Module):
    """One latent code per target offset, followed by one shared decoder."""

    def __init__(
        self,
        memory_dim: int,
        reconstruction_latent_dim: int,
        offsets: tuple[int, ...],
        decoder: VisionMambaDecoder,
    ) -> None:
        super().__init__()
        if not offsets or len(set(offsets)) != len(offsets):
            raise ValueError("reconstruction offsets must be non-empty and unique")
        self.offsets = offsets
        self.reconstruction_latent_dim = reconstruction_latent_dim
        self.projection = nn.Linear(
            memory_dim, len(offsets) * reconstruction_latent_dim
        )
        self.decoder = decoder

    def forward(self, memory: torch.Tensor) -> dict[int, torch.Tensor]:
        if memory.ndim not in (2, 3):
            raise ValueError("memory must have shape [B,D] or [B,T,D]")
        prefix = memory.shape[:-1]
        codes = self.project_codes(memory)
        # Fold the target branch into the batch so all branches pass through
        # exactly the same decoder parameters in one efficient call.
        flat_codes = codes.reshape(-1, self.reconstruction_latent_dim)
        images = self.decoder(flat_codes).reshape(
            *prefix,
            len(self.offsets),
            self.decoder.channels,
            *self.decoder.image_size,
        )
        return {
            offset: images[..., index, :, :, :]
            for index, offset in enumerate(self.offsets)
        }

    def project_codes(self, memory: torch.Tensor) -> torch.Tensor:
        """Return per-offset latent codes before the shared image decoder."""
        if memory.ndim not in (2, 3):
            raise ValueError("memory must have shape [B,D] or [B,T,D]")
        prefix = memory.shape[:-1]
        return self.projection(memory).reshape(
            *prefix, len(self.offsets), self.reconstruction_latent_dim
        )


class MambaPerceptionMemory(nn.Module):
    """Shared frozen representation used by both bootstrap and final SAC."""

    def __init__(self, encoder: VisionMambaEncoder, memory: TemporalMambaMemory) -> None:
        super().__init__()
        self.encoder = encoder
        self.memory = memory
        self._cache: list[MambaState] | None = None

    def reset(self, batch_size: int = 1) -> None:
        parameter = next(self.parameters())
        self._cache = self.memory.reset(batch_size, parameter.device, parameter.dtype)

    reset_memory = reset

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
        latent = self.encoder(frame)
        if self._cache is None or self._cache[0].conv.shape[0] != frame.shape[0]:
            self.reset(frame.shape[0])
        output_feature, self._cache = self.memory.step(latent, self._cache)
        return output_feature

    def encode_sequence(
        self,
        frames: torch.Tensor,
        episode_start_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,C,H,W]")
        batch, time, channels, height, width = frames.shape
        latent = self.encoder(frames.reshape(batch * time, channels, height, width))
        return self.memory.forward_sequence(
            latent.reshape(batch, time, -1), episode_start_mask
        )

    def freeze(self) -> "MambaPerceptionMemory":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
