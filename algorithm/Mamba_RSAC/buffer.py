import numpy as np


class EpisodeReplayBuffer:
    """Episode replay buffer that samples contiguous padded sequences."""

    def __init__(self, max_size: int, sequence_length: int, seed=None, store_privileged: bool = False):
        self.max_size = int(max_size)
        self.sequence_length = int(sequence_length)
        self.store_privileged = bool(store_privileged)
        self.rng = np.random.default_rng(seed)
        self.episodes = []
        self.current_episode = []
        self.total_size = 0

    def add(
        self,
        base_state,
        depth,
        prev_action,
        action,
        reward,
        next_base_state,
        next_depth,
        done,
        critic_priv=None,
        next_critic_priv=None,
    ):
        transition = {
            "base": np.asarray(base_state, dtype=np.float32).copy(),
            "depth": np.asarray(depth, dtype=np.float16).copy(),
            "prev_action": np.asarray(prev_action, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "reward": np.asarray([reward], dtype=np.float32),
            "next_base": np.asarray(next_base_state, dtype=np.float32).copy(),
            "next_depth": np.asarray(next_depth, dtype=np.float16).copy(),
            "done": np.asarray([done], dtype=np.float32),
        }
        if self.store_privileged:
            transition["critic_priv"] = np.asarray(critic_priv, dtype=np.float16).copy()
            transition["next_critic_priv"] = np.asarray(next_critic_priv, dtype=np.float16).copy()

        self.current_episode.append(transition)
        self.total_size += 1

        if bool(done):
            self._finish_current_episode()
        self._trim_to_capacity()

    def _finish_current_episode(self):
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def _all_episodes(self):
        if self.current_episode:
            return self.episodes + [self.current_episode]
        return self.episodes

    def _trim_to_capacity(self):
        while self.total_size > self.max_size and self.episodes:
            removed = self.episodes.pop(0)
            self.total_size -= len(removed)

        if self.total_size > self.max_size and self.current_episode:
            overflow = self.total_size - self.max_size
            if overflow > 0:
                del self.current_episode[:overflow]
                self.total_size -= overflow

    @staticmethod
    def _stack(values, dtype=np.float32):
        return np.stack(values, axis=0).astype(dtype, copy=False)

    def _sequence_from_episode(self, episode, end_idx: int):
        start_idx = end_idx - self.sequence_length + 1
        first = episode[0]
        fields = [
            "base",
            "depth",
            "prev_action",
            "action",
            "reward",
            "next_base",
            "next_depth",
            "done",
        ]
        if self.store_privileged:
            fields.extend(["critic_priv", "next_critic_priv"])

        seq = {field: [] for field in fields}
        mask = []
        for idx in range(start_idx, end_idx + 1):
            transition = first if idx < 0 else episode[idx]
            valid = idx >= 0
            for field in fields:
                seq[field].append(transition[field])
            mask.append([1.0 if valid else 0.0])

        batch = {field: self._stack(values) for field, values in seq.items()}
        batch["mask"] = self._stack(mask)
        return batch

    def sample(self, batch_size: int):
        episodes = [episode for episode in self._all_episodes() if len(episode) > 0]
        if not episodes:
            return None

        lengths = np.asarray([len(episode) for episode in episodes], dtype=np.float64)
        probs = lengths / max(float(lengths.sum()), 1.0)
        samples = []
        for _ in range(int(batch_size)):
            episode_idx = int(self.rng.choice(len(episodes), p=probs))
            episode = episodes[episode_idx]
            end_idx = int(self.rng.integers(0, len(episode)))
            samples.append(self._sequence_from_episode(episode, end_idx))

        fields = list(samples[0].keys())
        return tuple(np.stack([sample[field] for sample in samples], axis=0) for field in fields)

    def size(self) -> int:
        return int(self.total_size)

    def end_episode(self):
        self._finish_current_episode()
