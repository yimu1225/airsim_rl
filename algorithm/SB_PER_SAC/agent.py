import numpy as np

from ..config_loader import get_algo_param
from ..SAC.agent import SACAgent
from ..SB_PER_VSSM_SAC.buffer import DualPrioritizedReplayBuffer


class SB_PERSACAgent(SACAgent):
    """SAC with Success-Buffer Prioritized Experience Replay and no VSSM encoder."""

    def _make_replay_buffer(self, args, seed=None):
        return DualPrioritizedReplayBuffer(
            args.buffer_size,
            success_capacity_ratio=get_algo_param(args, "sb_per_success_capacity_ratio", 0.3),
            success_sample_ratio=get_algo_param(args, "sb_per_success_sample_ratio", 0.30),
            alpha=get_algo_param(args, "sb_per_alpha", 0.6),
            eps=get_algo_param(args, "sb_per_eps", 1e-6),
            seed=seed,
        )

    def train(self, progress_ratio=0.0):
        self._current_progress_ratio = float(np.clip(progress_ratio, 0.0, 1.0))
        return super().train(progress_ratio=progress_ratio)

    def _sb_per_beta(self, progress_ratio=0.0) -> float:
        beta0 = float(get_algo_param(self.args, "sb_per_beta0", 0.4))
        beta1 = float(get_algo_param(self.args, "sb_per_beta1", 1.0))
        progress = float(np.clip(progress_ratio, 0.0, 1.0))
        return beta0 * (1.0 - progress) + beta1 * progress

    def _get_current_success_sample_ratio(self, progress_ratio: float) -> float:
        mu_low = float(get_algo_param(self.args, "sb_per_mu_low", 0.30))
        mu_mid = float(get_algo_param(self.args, "sb_per_mu_mid", 0.40))
        mu_high = float(get_algo_param(self.args, "sb_per_mu_high", 0.45))
        mu_step1 = float(get_algo_param(self.args, "sb_per_mu_step1", 0.25))
        mu_step2 = float(get_algo_param(self.args, "sb_per_mu_step2", 0.70))

        p = float(np.clip(progress_ratio, 0.0, 1.0))
        if p < mu_step1:
            mu = mu_low
        elif p < mu_step2:
            mu = mu_mid
        else:
            mu = mu_high
        return float(np.clip(mu, 0.0, 0.8))

    def _sample_replay(self):
        progress_ratio = float(getattr(self, "_current_progress_ratio", 0.0))
        sb_per_beta = self._sb_per_beta(progress_ratio)
        current_mu = self._get_current_success_sample_ratio(progress_ratio)
        self.replay_buffer.success_sample_ratio = current_mu

        out = self.replay_buffer.sample(self.batch_size, beta=sb_per_beta)
        if out is None:
            return None, None, None, {}

        samples, refs, weights, mix_info = out
        if isinstance(samples, tuple):
            stacked = samples
        else:
            stacked = tuple(np.stack(items, axis=0) for items in zip(*samples))

        replay_info = {
            "sb_per_beta": sb_per_beta,
            "replay/success_sample_ratio_target": current_mu,
            "replay/success_batch_fraction": mix_info["batch_success_fraction"],
            "replay/success_size": mix_info["success_size"],
            "replay/regular_size": mix_info["regular_size"],
        }
        return stacked, refs, weights, replay_info

    def _update_replay_priorities(self, refs, td_errors):
        self.replay_buffer.update_priorities(refs, np.asarray(td_errors, dtype=np.float32))


SACAgent = SB_PERSACAgent
