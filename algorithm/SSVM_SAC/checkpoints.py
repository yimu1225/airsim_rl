"""Stage-aware checkpoint helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import SSVMConfig
from .networks import MambaPerceptionMemory, TemporalMambaMemory, VisionMambaEncoder


MODEL_VERSION = 6


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
        raise ValueError(f"{path} is not a SSVM stage checkpoint")
    reusable_vision = checkpoint.get("stage") == "vision" and checkpoint.get("model_version") in {3, 4, 5}
    if checkpoint.get("model_version") != MODEL_VERSION and not reusable_vision:
        raise ValueError(
            f"{path} uses SSVM model version {checkpoint.get('model_version')!r}; "
            f"this implementation requires version {MODEL_VERSION}. "
            "Retrain Memory from a vision-stage checkpoint (version 3 Vision remains supported)."
        )
    return checkpoint


def build_perception(config: SSVMConfig) -> MambaPerceptionMemory:
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
        d_state=config.memory_d_state,
        d_conv=config.d_conv,
        expand=config.expand,
    )
    return MambaPerceptionMemory(encoder, memory)


def perception_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device | str, *, freeze: bool = True
) -> tuple[SSVMConfig, MambaPerceptionMemory]:
    config = SSVMConfig.from_mapping(checkpoint["config"])
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
