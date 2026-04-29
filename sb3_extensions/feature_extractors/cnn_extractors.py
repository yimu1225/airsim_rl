"""CNN feature extractors for SB3 AirSim baselines."""

from __future__ import annotations

from gymnasium import spaces
from stable_baselines3.common.torch_layers import NatureCNN
from torch import nn

from sb3_extensions.feature_extractors.base import AirSimBaseExtractor


class AirSimCNNExtractor(AirSimBaseExtractor):
    """
    AirSim baseline extractor: depth CNN features plus base-state MLP features.

    All CNN-based SB3 agents use Stable-Baselines3's default NatureCNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        base_feature_dim: int = 32,
        vision_output_dim: int | None = None,
        cnn_type: str = "nature",
        normalize_depth: bool = True,
        depth_scale: float | None = None,
        algorithm_params: dict | None = None,
    ) -> None:
        del algorithm_params
        self.cnn_type = str(cnn_type).lower()
        if self.cnn_type != "nature":
            raise ValueError("AirSimCNNExtractor only supports SB3 NatureCNN (`cnn_type='nature'`).")
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
            base_feature_dim=base_feature_dim,
            vision_output_dim=vision_output_dim,
            normalize_depth=normalize_depth,
            depth_scale=depth_scale,
        )

    def _build_vision_backbone(self, depth_space: spaces.Box, output_dim: int) -> nn.Module:
        self.vision_backbone_name = "NatureCNN"
        return NatureCNN(depth_space, features_dim=output_dim, normalized_image=True)


class NatureCNNExtractor(AirSimCNNExtractor):
    """Explicit NatureCNN variant for readability in configs."""

    def __init__(self, observation_space: spaces.Dict, **kwargs) -> None:
        kwargs.setdefault("cnn_type", "nature")
        super().__init__(observation_space, **kwargs)
