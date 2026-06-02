import torch
import torch.nn.functional as F
from torch import nn

from algorithm.Mamba_RSAC.agent import MambaRSACAgent
from .buffer import ReplayBuffer


class PLMambaRSACAgent(MambaRSACAgent):
    """Mamba-RSAC with noisy actor observations and clean-depth critic observations."""

    def _make_replay_buffer(self, seed=None):
        return ReplayBuffer(
            self.args.buffer_size,
            self.sequence_length,
            seed=seed,
        )

    def train(self, progress_ratio=0.0):
        del progress_ratio
        self.total_it += 1
        if self.replay_buffer.size() < self.batch_size:
            return {}

        sample = self._sample_replay()
        if sample is None:
            return {}
        (
            base,
            depth,
            prev_action,
            action,
            reward,
            next_base,
            next_depth,
            done,
            critic_priv,
            next_critic_priv,
            replay_mask,
        ) = sample

        base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        critic_priv = torch.as_tensor(critic_priv, dtype=torch.float32, device=self.device)
        real_prev_action = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
        prev_action_norm = self._normalize_action_tensor(real_prev_action)
        real_action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action_norm = self._normalize_action_tensor(real_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_base = torch.as_tensor(next_base, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)
        next_critic_priv = torch.as_tensor(next_critic_priv, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        replay_mask = torch.as_tensor(replay_mask, dtype=torch.float32, device=self.device)
        learn_mask = self._learning_mask(replay_mask)

        with torch.no_grad():
            next_actor_seq = self._encode_state_sequence(
                next_base,
                next_depth,
                action_norm,
                self.actor_encoder,
                self.actor_base_adapter,
                self.actor_history,
            )
            next_action_pi, next_log_prob = self.actor.action_log_prob(next_actor_seq.reshape(-1, next_actor_seq.size(-1)))
            next_action_pi_seq = next_action_pi.view(base.size(0), base.size(1), self.action_dim)
            next_log_prob_seq = next_log_prob.view(base.size(0), base.size(1), 1)
            next_target_seq = self._encode_state_sequence(
                next_base,
                next_critic_priv,
                action_norm,
                self.critic_encoder_target,
                self.critic_base_adapter_target,
                self.critic_history_target,
            )
            target_q1, target_q2 = self.critic_target(
                next_target_seq.reshape(-1, next_target_seq.size(-1)),
                next_action_pi_seq.reshape(-1, self.action_dim),
            )
            target_q = torch.min(target_q1, target_q2).view(base.size(0), base.size(1), 1)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            target_q = reward + (1.0 - done) * self.gamma * (target_q - alpha * next_log_prob_seq)
            target_q = self._select_learning(target_q, learn_mask)

        critic_seq = self._encode_state_sequence(
            base,
            critic_priv,
            prev_action_norm,
            self.critic_encoder,
            self.critic_base_adapter,
            self.critic_history,
        )
        current_q1, current_q2 = self.critic(
            self._select_learning(critic_seq, learn_mask),
            self._select_learning(action_norm, learn_mask),
        )
        critic_loss = 0.5 * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))
        target_q_mean_value = float(target_q.mean().detach().item())
        current_q_mean_value = float(torch.min(current_q1, current_q2).mean().detach().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip)
        self.critic_optimizer.step()

        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None
        if self.total_it % self.policy_freq == 0:
            actor_seq = self._encode_state_sequence(
                base,
                depth,
                prev_action_norm,
                self.actor_encoder,
                self.actor_base_adapter,
                self.actor_history,
            )
            actor_flat = self._select_learning(actor_seq, learn_mask)
            sampled_actions, log_prob = self.actor.action_log_prob(actor_flat)

            with torch.no_grad():
                critic_state_for_pi = self._select_learning(critic_seq.detach(), learn_mask)
            q1_pi, q2_pi = self.critic(critic_state_for_pi, sampled_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            actor_loss = (alpha * log_prob - min_q_pi).mean()
            mean_log_prob_value = float(log_prob.mean().detach().item())
            q_pi_mean_value = float(min_q_pi.mean().detach().item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().detach().item())
                alpha_loss_value = float(alpha_loss.item())

        if self.total_it % self.target_update_interval == 0:
            self._soft_update()

        result = {
            "critic_loss": float(critic_loss.item()),
            "alpha": float(self.alpha),
            "target_q_mean": target_q_mean_value,
            "current_q_mean": current_q_mean_value,
        }
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        if mean_log_prob_value is not None:
            result["mean_log_prob"] = mean_log_prob_value
        if q_pi_mean_value is not None:
            result["q_pi_mean"] = q_pi_mean_value
        if alpha_loss_value is not None:
            result["alpha_loss"] = alpha_loss_value
        return result
