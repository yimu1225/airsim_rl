"""CNN feature extractors for SB3 AirSim baselines."""

from __future__ import annotations

from gymnasium import spaces
from stable_baselines3.common.torch_layers import NatureCNN
from torch import nn

from sb3_extensions.feature_extractors.base import AirSimBaseExtractor


class AirSimCNNExtractor(AirSimBaseExtractor):
    """
    AirSim baseline extractor: stacked depth CNN features plus base-state MLP features.

    Stacked depth frames are treated as CNN input channels. For the default
    4-frame input, NatureCNN receives a (4, H, W) tensor and outputs a fixed
    256-dimensional visual feature, which is concatenated with the 32-dimensional
    base-state feature.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        vision_output_dim: int | None = None,
        cnn_feature_dim: int = 256,
        cnn_type: str = "nature",
        normalize_depth: bool = True,
        depth_scale: float | None = None,
        algorithm_params: dict | None = None,
    ) -> None:
        del features_dim
        algorithm_params = dict(algorithm_params or {})
        self.cnn_type = str(cnn_type).lower()
        if self.cnn_type != "nature":
            raise ValueError("AirSimCNNExtractor only supports SB3 NatureCNN (`cnn_type='nature'`).")

        if vision_output_dim is None:
            vision_output_dim = int(algorithm_params.get("cnn_feature_dim", cnn_feature_dim))
        base_dim = int(observation_space.spaces["base"].shape[0])
        total_features_dim = int(vision_output_dim) + base_dim

        super().__init__(
            observation_space=observation_space,
            features_dim=total_features_dim,
            vision_output_dim=vision_output_dim,
            normalize_depth=normalize_depth,
            depth_scale=depth_scale,
        )

    def _build_vision_backbone(self, depth_space: spaces.Box, output_dim: int) -> nn.Module:
        self.vision_backbone_name = "StackedNatureCNN"
        return NatureCNN(depth_space, features_dim=output_dim, normalized_image=True)


class NatureCNNExtractor(AirSimCNNExtractor):
    """Explicit NatureCNN variant for readability in configs."""

    def __init__(self, observation_space: spaces.Dict, **kwargs) -> None:
        kwargs.setdefault("cnn_type", "nature")
        super().__init__(observation_space, **kwargs)
