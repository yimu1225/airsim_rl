"""Stage-aware checkpoint helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import MAVMConfig
from .networks import MambaPerceptionMemory, TemporalMambaMemory, VisionMambaEncoder


MODEL_VERSION = 3


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "stage" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"{path} is not a MAVM stage checkpoint")
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"{path} uses MAVM model version {checkpoint.get('model_version')!r}; "
            f"this implementation requires version {MODEL_VERSION}"
        )
    return checkpoint


def build_perception(config: MAVMConfig) -> MambaPerceptionMemory:
    encoder = VisionMambaEncoder(
        config.image_size,
        channels=config.channels,
        patch_size=config.patch_size,
        latent_dim=config.latent_dim,
        depth=config.encoder_depth,
        d_state=config.d_state,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
    )
    memory = TemporalMambaMemory(
        config.latent_dim,
        config.memory_dim,
        depth=config.memory_depth,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
    )
    return MambaPerceptionMemory(encoder, memory)


def perception_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device | str, *, freeze: bool = True
) -> tuple[MAVMConfig, MambaPerceptionMemory]:
    config = MAVMConfig.from_mapping(checkpoint["config"])
    perception = build_perception(config).to(device)
    if "perception" in checkpoint:
        perception.load_state_dict(checkpoint["perception"])
    else:
        if "encoder" not in checkpoint or "memory" not in checkpoint:
            raise ValueError("checkpoint does not contain a complete encoder and memory")
        perception.encoder.load_state_dict(checkpoint["encoder"])
        perception.memory.load_state_dict(checkpoint["memory"])
    if freeze:
        perception.freeze()
    return config, perception
