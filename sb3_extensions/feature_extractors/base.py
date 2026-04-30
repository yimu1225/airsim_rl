"""Shared dict-observation feature extractor scaffolding for AirSim environments."""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def _prod(values: tuple[int, ...]) -> int:
    return int(reduce(mul, values, 1))


class AirSimBaseExtractor(BaseFeaturesExtractor):
    """
    Base extractor for AirSim dict observations.

    Expected observation format:
      {"depth": (..., H, W), "base": (base_dim,), ...}

    Leading depth dimensions are flattened into CNN channels, so both
    (T, H, W) and (T, C, H, W) can be consumed by channel-first CNN backbones.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        base_feature_dim: int = 32,
        vision_output_dim: int | None = None,
        normalize_depth: bool = True,
        depth_scale: float | None = None,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                f"{self.__class__.__name__} expects a gymnasium.spaces.Dict observation space, "
                f"got {type(observation_space)!r}."
            )
        if "depth" not in observation_space.spaces or "base" not in observation_space.spaces:
            raise KeyError("AirSim observation space must contain 'depth' and 'base' keys.")
        if features_dim <= 0:
            raise ValueError("features_dim must be positive.")
        if base_feature_dim <= 0:
            raise ValueError("base_feature_dim must be positive.")

        if vision_output_dim is None:
            vision_output_dim = features_dim - base_feature_dim
        if vision_output_dim <= 0:
            raise ValueError("features_dim must be greater than base_feature_dim.")
        if vision_output_dim + base_feature_dim != features_dim:
            raise ValueError("vision_output_dim + base_feature_dim must equal features_dim.")

        super().__init__(observation_space, features_dim=features_dim)

        self.depth_space = observation_space.spaces["depth"]
        self.base_space = observation_space.spaces["base"]
        if not isinstance(self.depth_space, spaces.Box):
            raise TypeError("'depth' observation must be a Box space.")
        if not isinstance(self.base_space, spaces.Box):
            raise TypeError("'base' observation must be a Box space.")
        if self.depth_space.shape is None or len(self.depth_space.shape) < 3:
            raise ValueError("'depth' observation must have at least 3 dimensions: (..., H, W).")
        if self.base_space.shape is None or len(self.base_space.shape) != 1:
            raise ValueError("'base' observation must be a 1D vector Box space.")

        self.depth_shape = tuple(int(dim) for dim in self.depth_space.shape)
        self.base_dim = int(self.base_space.shape[0])
        self.normalize_depth = bool(normalize_depth)
        self.depth_scale = float(depth_scale) if depth_scale is not None else self._infer_depth_scale(self.depth_space)
        self.vision_input_shape = self._make_vision_input_shape(self.depth_shape)

        vision_space = spaces.Box(
            low=0.0,
            high=1.0 if self.normalize_depth else self.depth_scale,
            shape=self.vision_input_shape,
            dtype=np.float32,
        )
        self.vision_net = self._build_vision_backbone(vision_space, vision_output_dim)
        self.base_net = nn.Sequential(
            nn.Linear(self.base_dim, base_feature_dim),
            nn.ReLU(),
        )

    def _build_vision_backbone(self, depth_space: spaces.Box, output_dim: int) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    def _make_vision_input_shape(depth_shape: tuple[int, ...]) -> tuple[int, int, int]:
        channels = _prod(depth_shape[:-2])
        height, width = depth_shape[-2:]
        return channels, height, width

    @staticmethod
    def _infer_depth_scale(depth_space: spaces.Box) -> float:
        high = np.asarray(depth_space.high, dtype=np.float32)
        finite_high = high[np.isfinite(high)]
        if finite_high.size == 0:
            return 255.0
        max_high = float(np.max(finite_high))
        return max_high if max_high > 1.0 else 1.0

    def _prepare_depth(self, depth: th.Tensor) -> th.Tensor:
        depth = depth.float()
        if depth.dim() == len(self.depth_shape):
            depth = depth.unsqueeze(0)

        expected_dims = len(self.depth_shape) + 1
        if depth.dim() != expected_dims:
            raise ValueError(
                f"Expected depth tensor with {expected_dims} dims including batch for shape "
                f"{self.depth_shape}, got {tuple(depth.shape)}."
            )

        batch_size = depth.shape[0]
        height, width = depth.shape[-2:]
        depth = depth.reshape(batch_size, self.vision_input_shape[0], height, width)

        if self.normalize_depth and self.depth_scale > 1.0:
            depth = depth / self.depth_scale
        return depth

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        depth = self._prepare_depth(observations["depth"])
        base = observations["base"].float()
        if base.dim() == 1:
            base = base.unsqueeze(0)

        vision_features = self.vision_net(depth)
        base_features = self.base_net(base)
        return th.cat((vision_features, base_features), dim=1)
