"""Serializable model and optimization configuration for MAVM-SAC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class MAVMConfig:
    image_height: int = 128
    image_width: int = 128
    channels: int = 1
    patch_size: int = 16
    latent_dim: int = 64
    encoder_depth: int = 2
    memory_dim: int = 128
    memory_depth: int = 2
    reconstruction_latent_dim: int = 64
    decoder_embed_dim: int = 64
    decoder_depth: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    reconstruction_offsets: tuple[int, ...] = (-5, -10, 5)
    reconstruction_loss: str = "mse"
    charbonnier_epsilon: float = 1e-3
    vision_lr: float = 3e-4
    memory_lr: float = 3e-4
    memory_lrf: float = 0.01
    offline_lr_decay: float = 0.99
    vision_batch_size: int = 256
    memory_batch_size: int = 4
    actor_lr: float = 4e-4
    critic_lr: float = 4e-4
    alpha_lr: float = 4e-4
    hidden_dim: int = 256
    gamma: float = 0.95
    tau: float = 0.003
    initial_alpha: float = 0.2
    auto_entropy: bool = True
    batch_size: int = 8
    replay_capacity: int = 100_000
    gradient_clip: float = 10.0
    sb_per_alpha: float = 0.6
    sb_per_beta0: float = 0.4
    sb_per_beta1: float = 1.0
    sb_per_eps: float = 1e-6
    sb_per_success_capacity_ratio: float = 0.3
    sb_per_success_sample_ratio: float = 0.3
    sb_per_mu_low: float = 0.3
    sb_per_mu_mid: float = 0.4
    sb_per_mu_high: float = 0.45
    sb_per_mu_step1: float = 0.25
    sb_per_mu_step2: float = 0.7

    def __post_init__(self) -> None:
        if not 0.0 <= self.memory_lrf <= 1.0:
            raise ValueError("memory_lrf must be in [0, 1]")
        if not 0.0 < self.offline_lr_decay <= 1.0:
            raise ValueError("offline_lr_decay must be in (0, 1]")
        if self.patch_size <= 0 or self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions and patch_size must be positive")
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        positive = {
            "channels": self.channels,
            "latent_dim": self.latent_dim,
            "memory_dim": self.memory_dim,
            "reconstruction_latent_dim": self.reconstruction_latent_dim,
            "decoder_embed_dim": self.decoder_embed_dim,
            "vision_batch_size": self.vision_batch_size,
            "memory_batch_size": self.memory_batch_size,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"MAVM dimensions and capacities must be positive: {positive}")
        if 0 in self.reconstruction_offsets:
            raise ValueError("memory reconstruction offsets must not include the current frame")
        if self.charbonnier_epsilon <= 0.0:
            raise ValueError("charbonnier_epsilon must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma and tau must be in (0, 1]")
        if self.sb_per_alpha < 0.0 or self.sb_per_eps <= 0.0:
            raise ValueError("SB-PER alpha must be non-negative and eps must be positive")
        unit_interval = {
            "sb_per_beta0": self.sb_per_beta0,
            "sb_per_beta1": self.sb_per_beta1,
            "sb_per_success_capacity_ratio": self.sb_per_success_capacity_ratio,
            "sb_per_success_sample_ratio": self.sb_per_success_sample_ratio,
            "sb_per_mu_low": self.sb_per_mu_low,
            "sb_per_mu_mid": self.sb_per_mu_mid,
            "sb_per_mu_high": self.sb_per_mu_high,
            "sb_per_mu_step1": self.sb_per_mu_step1,
            "sb_per_mu_step2": self.sb_per_mu_step2,
        }
        if any(not 0.0 <= value <= 1.0 for value in unit_interval.values()):
            raise ValueError(f"SB-PER ratios must be in [0, 1]: {unit_interval}")
        if self.sb_per_beta1 < self.sb_per_beta0:
            raise ValueError("sb_per_beta1 must be greater than or equal to sb_per_beta0")
        if self.sb_per_mu_step2 < self.sb_per_mu_step1:
            raise ValueError("sb_per_mu_step2 must be greater than or equal to sb_per_mu_step1")

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def reconstruction_projection_dim(self) -> int:
        return len(self.reconstruction_offsets) * self.reconstruction_latent_dim

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MAVMConfig":
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown MAVM configuration keys: {sorted(unknown)}")
        data = dict(values)
        if "reconstruction_offsets" in data:
            data["reconstruction_offsets"] = tuple(data["reconstruction_offsets"])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path | None) -> "MAVMConfig":
        if path is None:
            path = Path(__file__).with_name("params.yaml")
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reconstruction_offsets"] = list(self.reconstruction_offsets)
        return result
