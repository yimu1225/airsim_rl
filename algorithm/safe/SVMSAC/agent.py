import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from algorithm.VMSAC.agent import STVimSACAgent
from ...config_loader import get_algo_param
from .buffer import ReplayBuffer
from .networks import SafetyConstraintHead, safety_project_actions


class STSVimSACAgent(STVimSACAgent):
    """ST-Vim SAC with a learned safety projection layer."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        super().__init__(base_dim, depth_shape, action_space, args, device=device, seed=seed)
        print(f"ST-Mamba-VimTokens-Safety-SAC Agent using device: {self.device}")

        self.replay_buffer = ReplayBuffer(args.buffer_size, seed=seed)
        self.use_safety_layer = get_algo_param(args, "use_vim_safety_layer", True)
        self.safety_model = SafetyConstraintHead(
            latent_dim=self.state_dim,
            action_dim=self.action_dim,
        ).to(self.device)
        self.safety_model_target = copy.deepcopy(self.safety_model)

        self.safety_end_to_end = get_algo_param(args, "safety_end_to_end", False)
        self.safety_params = list(self.safety_model.parameters())
        if self.safety_end_to_end:
            self.safety_params += list(self.actor_encoder.parameters())
        self.safety_optimizer = Adam(
            self.safety_params,
            lr=get_algo_param(args, "safety_lr", args.actor_lr),
        )

        self.safety_loss_coef = get_algo_param(args, "safety_loss_coef", 1.0)
        self.safety_actor_penalty_coef = get_algo_param(args, "safety_actor_penalty_coef", 0.05)
        self.safety_warmup_steps = get_algo_param(args, "safety_warmup_steps", 0)
        self.safety_label_mode = get_algo_param(args, "safety_label_mode", "collision")

    def _apply_safety_projection(self, action, state_features, safety_model=None):
        if not self.use_safety_layer:
            return action, None
        model = self.safety_model if safety_model is None else safety_model
        g, h = model(state_features)
        safe_action, violation = safety_project_actions(action, g, h)
        return safe_action.clamp(-1.0, 1.0), violation

    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False, progress_ratio=0.0):
        base = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            state = self._encode_state(base, depth_tensor, self.actor_encoder)
            log_prob = None
            if with_log_prob and not deterministic:
                action, log_prob = self.actor.action_log_prob(state)
            else:
                action = self.actor(state, deterministic=deterministic)
            action, _ = self._apply_safety_projection(action, state)
            real_action = self.action_scale * action + self.action_bias
            if log_prob is not None:
                return real_action.cpu().numpy().flatten(), log_prob.cpu().numpy()
            return real_action.cpu().numpy().flatten()

    def train(self, progress_ratio=0.0):
        self.total_it += 1
        if self.replay_buffer.size() < self.batch_size:
            return {}

        sample, replay_refs, replay_weights, replay_info = self._sample_replay()
        if sample is None:
            return {}
        (
            base_states,
            depths,
            actions,
            rewards,
            next_base_states,
            next_depths,
            dones,
            collision_flags,
        ) = sample

        base_states = torch.as_tensor(base_states, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = ((real_actions - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_base_states, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depths, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        collision_flags = torch.as_tensor(collision_flags, dtype=torch.float32, device=self.device).view(-1, 1)
        weights = None
        if replay_weights is not None:
            weights = torch.as_tensor(replay_weights, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_actor_state = self._encode_state(
                next_base_states, next_depths, self.actor_encoder
            )
            next_actions, next_log_prob = self.actor.action_log_prob(next_actor_state)
            next_actions, _ = self._apply_safety_projection(
                next_actions,
                next_actor_state,
                safety_model=self.safety_model_target,
            )
            next_target_state = self._encode_state(
                next_base_states, next_depths, self.critic_encoder_target
            )
            next_q1, next_q2 = self.critic_target(next_target_state, next_actions)
            next_q = torch.min(next_q1, next_q2)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            target_q = rewards + (1.0 - dones) * self.gamma * (next_q - alpha * next_log_prob)

        critic_state = self._encode_state(base_states, depths, self.critic_encoder)
        current_q1, current_q2 = self.critic(critic_state, actions)
        critic_loss_elements = 0.5 * (
            F.mse_loss(current_q1, target_q, reduction="none")
            + F.mse_loss(current_q2, target_q, reduction="none")
        )
        critic_loss = (critic_loss_elements * weights).mean() if weights is not None else critic_loss_elements.mean()
        td_errors = 0.5 * ((current_q1 - target_q).abs() + (current_q2 - target_q).abs())
        target_q_mean_value = float(target_q.mean().detach().item())
        current_q_mean_value = float(torch.min(current_q1, current_q2).mean().detach().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip)
        self.critic_optimizer.step()
        if replay_refs is not None:
            self._update_replay_priorities(replay_refs, td_errors.detach().cpu().numpy().reshape(-1))

        safety_loss_value = 0.0
        safety_violation_rate = 0.0
        if self.use_safety_layer and self.total_it >= self.safety_warmup_steps:
            safety_state = self._encode_state(base_states, depths, self.actor_encoder)
            safety_state = safety_state if self.safety_end_to_end else safety_state.detach()
            g, h = self.safety_model(safety_state)
            logits = (g * actions).sum(dim=-1, keepdim=True) + h
            collision_target = ((dones > 0.5) & (collision_flags > 0.5)).float()
            safety_loss = self.safety_loss_coef * F.binary_cross_entropy_with_logits(logits, collision_target)

            self.safety_optimizer.zero_grad()
            safety_loss.backward()
            nn.utils.clip_grad_norm_(self.safety_params, self.grad_clip)
            self.safety_optimizer.step()
            safety_loss_value = float(safety_loss.item())
            with torch.no_grad():
                safety_violation_rate = float((torch.sigmoid(logits) > 0.5).float().mean().item())

        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None
        safety_penalty_value = None
        if self.total_it % self.policy_freq == 0:
            actor_state = self._encode_state(base_states, depths, self.actor_encoder)
            actions_pi_raw, log_prob = self.actor.action_log_prob(actor_state)
            safety_penalty = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if self.use_safety_layer:
                with torch.no_grad():
                    g_actor, h_actor = self.safety_model(actor_state)
                actions_pi, actor_violation = safety_project_actions(actions_pi_raw, g_actor, h_actor)
                actions_pi = actions_pi.clamp(-1.0, 1.0)
                safety_penalty = torch.clamp(actor_violation, min=0.0).mean()
            else:
                actions_pi = actions_pi_raw

            with torch.no_grad():
                critic_state_for_pi = self._encode_state(
                    base_states, depths, self.critic_encoder
                )
            q1_pi, q2_pi = self.critic(critic_state_for_pi, actions_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            actor_loss = (alpha * log_prob - min_q_pi).mean()
            actor_loss = actor_loss + self.safety_actor_penalty_coef * safety_penalty
            mean_log_prob_value = float(log_prob.mean().detach().item())
            q_pi_mean_value = float(min_q_pi.mean().detach().item())
            safety_penalty_value = float(safety_penalty.detach().item())

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
            "safety_loss": safety_loss_value,
            "safety_violation_rate": safety_violation_rate,
        }
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        if mean_log_prob_value is not None:
            result["mean_log_prob"] = mean_log_prob_value
        if q_pi_mean_value is not None:
            result["q_pi_mean"] = q_pi_mean_value
        if safety_penalty_value is not None:
            result["safety_penalty"] = safety_penalty_value
        if alpha_loss_value is not None:
            result["alpha_loss"] = alpha_loss_value
        if replay_info:
            result.update(replay_info)
        return result

    def _soft_update(self):
        super()._soft_update()
        for param, target_param in zip(self.safety_model.parameters(), self.safety_model_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        checkpoint = {
            "actor_encoder": self.actor_encoder.state_dict(),
            "critic_encoder": self.critic_encoder.state_dict(),
            "critic_encoder_target": self.critic_encoder_target.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "safety_model": self.safety_model.state_dict(),
            "safety_model_target": self.safety_model_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "safety_optimizer": self.safety_optimizer.state_dict(),
            "total_it": self.total_it,
            "alpha": self.alpha,
        }
        if self.auto_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        self.critic_encoder_target.load_state_dict(checkpoint.get("critic_encoder_target", checkpoint["critic_encoder"]))
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if "safety_model" in checkpoint:
            self.safety_model.load_state_dict(checkpoint["safety_model"])
        if "safety_model_target" in checkpoint:
            self.safety_model_target.load_state_dict(checkpoint["safety_model_target"])
        elif "safety_model" in checkpoint:
            self.safety_model_target.load_state_dict(checkpoint["safety_model"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "safety_optimizer" in checkpoint:
            self.safety_optimizer.load_state_dict(checkpoint["safety_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)
        self.alpha = checkpoint.get("alpha", self.alpha)
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


SACAgent = STSVimSACAgent
