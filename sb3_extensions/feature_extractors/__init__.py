"""Feature extractors used by SB3-based AirSim agents."""

from sb3_extensions.feature_extractors.base import AirSimBaseExtractor
from sb3_extensions.feature_extractors.cnn_extractors import (
    AirSimCNNExtractor,
    NatureCNNExtractor,
)
from sb3_extensions.feature_extractors.lstm_extractors import LSTMExtractor
from sb3_extensions.feature_extractors.mamba_extractors import MambaExtractor
from sb3_extensions.feature_extractors.vim_extractors import (
    DualVimFeatureExtractor,
    STSeqVimFeatureExtractor,
    STSVimFeatureExtractor,
    STVSeqVimFeatureExtractor,
    STVimFeatureExtractor,
    VimFeatureExtractor,
    VimPatchFeatureExtractor,
)

__all__ = [
    "AirSimBaseExtractor",
    "AirSimCNNExtractor",
    "DualVimFeatureExtractor",
    "LSTMExtractor",
    "MambaExtractor",
    "NatureCNNExtractor",
    "STSeqVimFeatureExtractor",
    "STSVimFeatureExtractor",
    "STVSeqVimFeatureExtractor",
    "STVimFeatureExtractor",
    "VimFeatureExtractor",
    "VimPatchFeatureExtractor",
]
