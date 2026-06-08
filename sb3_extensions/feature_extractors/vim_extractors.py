"""Vim-family feature extractors for the SB3 migration layer."""

from __future__ import annotations

from sb3_extensions.feature_extractors.sequence_extractors import (
    AirSimSequenceExtractor,
    DualBranchVideoMambaEncoder,
    STSeqVimEncoder,
    STVSeqVimEncoder,
    STVimEncoder,
    STVimVideoPatchEncoder,
    VimFrameEncoder,
)


class VimFeatureExtractor(AirSimSequenceExtractor):
    """Per-frame VisionMamba, no temporal Mamba (`Vim_TD3`)."""

    encoder_cls = VimFrameEncoder


class STVimFeatureExtractor(AirSimSequenceExtractor):
    """Frame-wise VisionMamba followed by temporal Mamba (`VM*`)."""

    encoder_cls = STVimEncoder


class STSeqVimFeatureExtractor(AirSimSequenceExtractor):
    """ST-Vim with repeated/current base-state sequence fusion (`ST_Seq_Vim_TD3`)."""

    encoder_cls = STSeqVimEncoder
    uses_base_sequence = True


class STVSeqVimFeatureExtractor(AirSimSequenceExtractor):
    """Vim-only visual tokens fused with base-state sequence (`STV_Seq_Vim_TD3`)."""

    encoder_cls = STVSeqVimEncoder
    uses_base_sequence = True


class VimPatchFeatureExtractor(AirSimSequenceExtractor):
    """Sequence-level VideoPatchEmbed variant (`STV_Patch_TD3`)."""

    encoder_cls = STVimVideoPatchEncoder


class STSVimFeatureExtractor(AirSimSequenceExtractor):
    """Spatial Vim safety-layer encoder variant (`SVMTD3`)."""

    encoder_cls = STVimEncoder

    def __init__(self, observation_space, **kwargs):
        kwargs["pre_norm"] = False
        super().__init__(observation_space, **kwargs)


class DualVimFeatureExtractor(AirSimSequenceExtractor):
    """Hierarchical dual-branch VideoMamba (`ST_DualVim_TD3`)."""

    encoder_cls = DualBranchVideoMambaEncoder
