"""LSTM-SAC paper feature extractor migrated to SB3."""

from __future__ import annotations

from sb3_extensions.feature_extractors.sequence_extractors import AirSimSequenceExtractor, LSTMPaperEncoder


class LSTMExtractor(AirSimSequenceExtractor):
    """Self-supervised attention depth features followed by FC/LSTM/FC."""

    encoder_cls = LSTMPaperEncoder
