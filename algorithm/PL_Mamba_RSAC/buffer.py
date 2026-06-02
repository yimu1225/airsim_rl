from algorithm.Mamba_RSAC.buffer import EpisodeReplayBuffer


class ReplayBuffer(EpisodeReplayBuffer):
    """PL Mamba-RSAC replay buffer with clean-depth critic observations."""

    def __init__(self, max_size: int, sequence_length: int, seed=None):
        super().__init__(
            max_size=max_size,
            sequence_length=sequence_length,
            seed=seed,
            store_privileged=True,
        )
