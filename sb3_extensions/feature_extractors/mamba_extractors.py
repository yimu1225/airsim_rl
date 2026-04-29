"""Mamba-only feature extractor migrated from `Mamba_TD3`."""

from __future__ import annotations

from sb3_extensions.feature_extractors.sequence_extractors import AirSimSequenceExtractor, MambaOnlyEncoder


class MambaExtractor(AirSimSequenceExtractor):
    """SB3 NatureCNN per frame followed by temporal Mamba."""

    encoder_cls = MambaOnlyEncoder
