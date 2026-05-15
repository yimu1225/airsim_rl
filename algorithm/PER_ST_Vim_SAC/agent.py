import numpy as np

from ..config_loader import get_algo_param
from ..ST_Vim_SAC.agent import STVimSACAgent
from .buffer import PrioritizedReplayBuffer


class PERSTVimSACAgent(STVimSACAgent):
    """ST-Vim-SAC with single-pool prioritized replay."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)
        self.replay_buffer = PrioritizedReplayBuffer(
            args.buffer_size,
            alpha=get_algo_param(args, "per_alpha", 0.6),
            eps=get_algo_param(args, "per_eps", 1e-6),
            seed=seed,
        )

    def _per_beta(self, progress_ratio=0.0) -> float:
        beta0 = float(get_algo_param(self.args, "per_beta0", 0.4))
        beta1 = float(get_algo_param(self.args, "per_beta1", 1.0))
        progress = float(np.clip(progress_ratio, 0.0, 1.0))
        return beta0 * (1.0 - progress) + beta1 * progress

    def _sample_replay(self, progress_ratio=0.0):
        per_beta = self._per_beta(progress_ratio)
        out = self.replay_buffer.sample(self.batch_size, beta=per_beta)
        if out is None:
            return None, None, None, {}
        samples, indices, weights = out
        return samples, indices, weights, {
            "per_beta": per_beta,
            "replay/size": float(self.replay_buffer.size()),
        }

    def _update_replay_priorities(self, refs, td_errors):
        self.replay_buffer.update_priorities(refs, np.asarray(td_errors, dtype=np.float32))


SACAgent = PERSTVimSACAgent
