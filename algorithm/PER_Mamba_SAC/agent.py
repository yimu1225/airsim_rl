import numpy as np
import torch
import torch.nn.functional as F

from ..config_loader import get_algo_param
from ..Mamba_SAC.agent import MambaSACAgent
from .buffer import PrioritizedReplayBuffer


class PERMambaSACAgent(MambaSACAgent):
    """Mamba-SAC with single-pool prioritized replay."""

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

    def _to_float_tensor(self, data):
        tensor = torch.as_tensor(data, device=self.device)
        return tensor if tensor.dtype == torch.float32 else tensor.float()

    def train(self, progress_ratio=0.0):
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return {}

        sample, replay_refs, replay_weights, replay_info = self._sample_replay(progress_ratio)
        if sample is None:
            return {}
        base_states, depths, actions, rewards, next_base_states, next_depths, dones = sample

        base_states = self._to_float_tensor(base_states)
        depths = self._to_float_tensor(depths)
        real_actions = self._to_float_tensor(actions)
        actions = (real_actions - self.action_bias) / self.action_scale

        rewards = self._to_float_tensor(rewards).view(-1, 1)
        next_base_states = self._to_float_tensor(next_base_states)
        next_depths = self._to_float_tensor(next_depths)
        dones = self._to_float_tensor(dones).view(-1, 1)
        weights = None
        if replay_weights is not None:
            weights = self._to_float_tensor(replay_weights).view(-1, 1)

        # ============ Critic Update ============
        with torch.no_grad():
            next_actor_states = self._concat_state(
                next_base_states, next_depths, self.actor_encoder
            )
            next_target_states = self._concat_state(
                next_base_states, next_depths, self.critic_encoder_target
            )

            next_actions, next_log_probs, _, _ = self.actor(next_actor_states, with_log_prob=True)

            target_q1, target_q2 = self.critic_target(next_target_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            if self.auto_entropy_tuning:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = torch.as_tensor(self.alpha, dtype=torch.float32, device=self.device)
            target_q = target_q - alpha * next_log_probs

            target_q = rewards + (1 - dones) * self.gamma * target_q

        encoded_depths = self.critic_encoder(depths)
        states = torch.cat([base_states, encoded_depths], dim=1)

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss_elements = 0.5 * (
            F.mse_loss(current_q1, target_q, reduction="none")
            + F.mse_loss(current_q2, target_q, reduction="none")
        )
        if weights is not None:
            critic_loss = (critic_loss_elements * weights).mean()
        else:
            critic_loss = critic_loss_elements.mean()
        td_errors = 0.5 * ((current_q1 - target_q).abs() + (current_q2 - target_q).abs())
        target_q_mean_value = float(target_q.mean().detach().item())
        current_q_mean_value = float(torch.min(current_q1, current_q2).mean().detach().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()
        if replay_refs is not None:
            self._update_replay_priorities(replay_refs, td_errors.detach().cpu().numpy().reshape(-1))

        # ============ Actor and Alpha Update ============
        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None

        if self.total_it % self.policy_freq == 0:
            encoded_depths_actor = self.actor_encoder(depths)
            states_actor = torch.cat([base_states, encoded_depths_actor], dim=1)

            sampled_actions, log_probs, _, _ = self.actor(states_actor, with_log_prob=True)

            with torch.no_grad():
                critic_states_for_pi = self._concat_state(
                    base_states, depths, self.critic_encoder
                )

            q1_new, q2_new = self.critic(critic_states_for_pi, sampled_actions)
            q_new = torch.min(q1_new, q2_new)

            if self.auto_entropy_tuning:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = torch.as_tensor(self.alpha, dtype=torch.float32, device=self.device)

            actor_loss = (alpha * log_probs - q_new).mean()
            mean_log_prob_value = float(log_probs.mean().detach().item())
            q_pi_mean_value = float(q_new.mean().detach().item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=self.grad_clip)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().detach().item())
                alpha_loss_value = float(alpha_loss.item())

        # ============ Soft Update Target Networks ============
        if self.total_it % self.target_update_interval == 0:
            self._soft_update()

        result = {
            "critic_loss": float(critic_loss.item()),
            "alpha": self.alpha if isinstance(self.alpha, float) else self.alpha.item(),
            "target_q_mean": target_q_mean_value,
            "current_q_mean": current_q_mean_value,
        }
        if replay_info:
            result.update(replay_info)
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        if mean_log_prob_value is not None:
            result["mean_log_prob"] = mean_log_prob_value
        if q_pi_mean_value is not None:
            result["q_pi_mean"] = q_pi_mean_value
        if alpha_loss_value is not None:
            result["alpha_loss"] = alpha_loss_value
        return result
