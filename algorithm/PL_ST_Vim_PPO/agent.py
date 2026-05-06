import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from algorithm.ST_Vim_PPO.agent import STVimPPOAgent
from algorithm.ST_Vim_PPO.networks import Critic

from ..config_loader import get_algo_param
from .buffer import PLRolloutBuffer


class PLSTVimPPOAgent(STVimPPOAgent):
    """ST-Vim PPO with privileged information appended only to the value branch."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)

        self.actor_state_dim = self.state_dim
        self.critic_priv_dim = int(
            get_algo_param(args, "critic_priv_dim", getattr(args, "distance_sensor_count", 0))
        )
        self.critic_state_dim = self.actor_state_dim + self.critic_priv_dim

        hidden_dim = int(getattr(args, "hidden_dim", 256))
        lr = get_algo_param(args, "lr", getattr(args, "actor_lr", 3e-4))
        self.critic = Critic(self.critic_state_dim, hidden_dim).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        buffer_size = get_algo_param(args, "rollout_buffer_size", 2048)
        buffer_depth_shape = (args.n_frames, depth_shape[-2], depth_shape[-1])
        self.rollout_buffer = PLRolloutBuffer(
            buffer_size=buffer_size,
            base_dim=base_dim,
            depth_shape=buffer_depth_shape,
            action_dim=self.action_dim,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def _prepare_priv(self, critic_priv, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.critic_priv_dim <= 0:
            return torch.empty((batch_size, 0), dtype=dtype, device=device)
        if critic_priv is None:
            return torch.zeros((batch_size, self.critic_priv_dim), dtype=dtype, device=device)

        if torch.is_tensor(critic_priv):
            priv = critic_priv.to(device=device, dtype=dtype)
        else:
            priv = torch.as_tensor(critic_priv, dtype=dtype, device=device)

        if priv.dim() == 0:
            priv = priv.view(1, 1)
        elif priv.dim() == 1:
            priv = priv.view(1, -1)
        elif priv.dim() > 2:
            priv = priv.view(priv.shape[0], -1)

        if priv.shape[0] == 1 and batch_size > 1:
            priv = priv.expand(batch_size, -1)
        if priv.shape[0] != batch_size:
            raise ValueError(f"critic_priv batch mismatch: expected {batch_size}, got {priv.shape[0]}")

        if priv.shape[1] > self.critic_priv_dim:
            priv = priv[:, : self.critic_priv_dim]
        elif priv.shape[1] < self.critic_priv_dim:
            pad = torch.zeros(
                (batch_size, self.critic_priv_dim - priv.shape[1]),
                dtype=dtype,
                device=device,
            )
            priv = torch.cat([priv, pad], dim=1)
        return priv

    def _concat_critic_state(self, actor_state: torch.Tensor, critic_priv=None) -> torch.Tensor:
        priv = self._prepare_priv(critic_priv, actor_state.shape[0], actor_state.dtype, actor_state.device)
        return torch.cat([actor_state, priv], dim=1)

    def select_action(self, base_state, depth, critic_priv=None, deterministic=False, progress_ratio=0.0):
        base = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actor_state = self._concat_state(base, depth_tensor)
            critic_state = self._concat_critic_state(actor_state, critic_priv)
            action, log_prob = self.actor(actor_state, deterministic=deterministic)
            value = self.critic(critic_state)

        raw_action = action.cpu().numpy().flatten().astype(np.float32)
        self._last_policy_action = raw_action.copy()
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        real_action = self.action_scale.cpu().numpy() * clipped_action + self.action_bias.cpu().numpy()
        return real_action, float(value.cpu().numpy().flatten()[0]), float(log_prob.cpu().numpy().flatten()[0])

    def store_transition(self, base_state, depth, action, reward, value, log_prob, done, critic_priv=None):
        cached_action = self._last_policy_action
        if cached_action is not None and np.asarray(cached_action).shape == np.asarray(action).shape:
            buffer_action = cached_action.astype(np.float32)
            self._last_policy_action = None
        else:
            buffer_action = (np.asarray(action, dtype=np.float32) - self.action_bias.cpu().numpy()) / self.action_scale.cpu().numpy()
            buffer_action = np.clip(buffer_action, -1.0, 1.0)
        self.rollout_buffer.add(base_state, depth, buffer_action, reward, value, log_prob, done, critic_priv=critic_priv)

    def finish_trajectory(self, last_base_state, last_depth, last_done, critic_priv=None):
        with torch.no_grad():
            base = torch.as_tensor(last_base_state, dtype=torch.float32, device=self.device).view(1, -1)
            depth = torch.as_tensor(last_depth, dtype=torch.float32, device=self.device)
            actor_state = self._concat_state(base, depth)
            critic_state = self._concat_critic_state(actor_state, critic_priv)
            last_value = float(self.critic(critic_state).cpu().numpy().flatten()[0])
        return self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    def _update_policy(self, data, epoch_pbar=None):
        base_states = data["base_states"]
        depth_states = data["depth_states"]
        critic_privs = data["critic_privs"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        returns = data["returns"]
        advantages = data["advantages"]
        old_values = data["values"]

        n_samples = base_states.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0
        approx_kl_divs = []

        continue_training = True
        for epoch in range(self.ppo_epochs):
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb_indices = indices[start:end]

                actor_states = self._concat_state(base_states[mb_indices], depth_states[mb_indices])
                critic_states = self._concat_critic_state(actor_states, critic_privs[mb_indices])
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]

                new_log_probs, entropy = self.actor.get_log_prob(actor_states, mb_actions)
                values = self.critic(critic_states).squeeze(-1)
                log_ratio = new_log_probs.squeeze(-1) - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                if self.normalize_advantage and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = mb_old_values + (values - mb_old_values).clamp(
                        -float(self.clip_range_vf),
                        float(self.clip_range_vf),
                    )
                value_loss = F.mse_loss(values_pred, mb_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    approx_kl_divs.append(float(approx_kl.item()))

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    if epoch_pbar is not None:
                        epoch_pbar.set_postfix_str(f"Early stop (KL: {approx_kl:.4f})")
                    break

                self.encoder_optimizer.zero_grad()
                self.base_encoder_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.base_encoder.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.encoder_optimizer.step()
                self.base_encoder_optimizer.step()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.mean().item())
                total_loss += float(loss.item())
                n_updates += 1

            if epoch_pbar is not None:
                epoch_pbar.update(1)
            if not continue_training:
                break

        self.rollout_buffer.after_update()
        self.num_updates += 1
        return {
            "policy_loss": total_policy_loss / n_updates if n_updates else 0.0,
            "value_loss": total_value_loss / n_updates if n_updates else 0.0,
            "total_loss": total_loss / n_updates if n_updates else 0.0,
            "entropy": total_entropy / n_updates if n_updates else 0.0,
            "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0,
        }


def make_agent(env, initial_obs, args, device=None) -> PLSTVimPPOAgent:
    base_state = initial_obs["base"]
    depth = initial_obs["depth"]
    args.depth_shape = (1, depth.shape[-2], depth.shape[-1])
    return PLSTVimPPOAgent(
        base_dim=base_state.shape[0],
        depth_shape=args.depth_shape,
        action_space=env.action_space,
        args=args,
        device=device,
    )


PPOAgent = PLSTVimPPOAgent
