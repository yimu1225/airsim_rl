import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..config_loader import get_algo_param
from .buffer import DualPrioritizedReplayBuffer
from .networks import VisualSubNetwork, BaseSubNetwork, GlobalActor, Critic


class DPERSVMSACAgent:
    """State-Decomposed ST-Vim SAC with Dual Prioritized Experience Replay."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.args.depth_shape = depth_shape
        self.base_dim = base_dim
        self.depth_shape = depth_shape
        self.action_dim = action_space.shape[0]

        self.max_action = np.asarray(action_space.high, dtype=np.float32)
        self.min_action = np.asarray(action_space.low, dtype=np.float32)
        self.action_scale = torch.as_tensor((self.max_action - self.min_action) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.as_tensor((self.max_action + self.min_action) / 2.0, dtype=torch.float32, device=self.device)

        # ---- Hyperparameters ----
        self.sub_visual_hidden = get_algo_param(args, "sub_visual_hidden", [256, 128])
        self.sub_base_hidden = get_algo_param(args, "sub_base_hidden", [256, 128])
        self.sub_out_dim = get_algo_param(args, "sub_out_dim", 2)

        # ---- Sub-networks ----
        self.visual_sub = VisualSubNetwork(args, self.sub_visual_hidden, self.sub_out_dim).to(self.device)

        self.base_sub = BaseSubNetwork(self.base_dim, self.sub_base_hidden, self.sub_out_dim).to(self.device)

        # ---- GlobalActor ----
        global_actor_input_dim = self.sub_out_dim * 2 + self.visual_sub.repr_dim + self.base_dim
        self.global_actor = GlobalActor(global_actor_input_dim, args.hidden_dim, self.action_dim).to(self.device)

        # ---- Critic (online + target, independent encoder) ----
        self.critic = Critic(args, self.base_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(args, self.base_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ---- Optimizer parameter groups ----
        self.actor_params = (
            list(self.visual_sub.parameters())
            + list(self.base_sub.parameters())
            + list(self.global_actor.parameters())
        )
        self.critic_params = list(self.critic.parameters())

        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        # ---- Entropy tuning ----
        self.ent_coef = get_algo_param(args, "ent_coef", 0.2)
        target_entropy = get_algo_param(args, "target_entropy", "auto")
        self.target_entropy = -float(self.action_dim) if target_entropy in (None, "auto") else float(target_entropy)
        self.log_alpha = None
        self.alpha_optimizer = None
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                if init_value <= 0:
                    raise ValueError("Initial ent_coef value must be greater than 0.")
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=float(get_algo_param(args, "alpha_lr", args.actor_lr)))
            self.alpha = float(init_value)
            self.auto_entropy_tuning = True
        else:
            self.alpha = float(self.ent_coef)
            self.auto_entropy_tuning = False

        # ---- Dual Prioritized Replay Buffer ----
        self.replay_buffer = DualPrioritizedReplayBuffer(
            args.buffer_size,
            success_capacity_ratio=get_algo_param(args, "dper_success_capacity_ratio", 0.3),
            success_sample_ratio=get_algo_param(args, "dper_success_sample_ratio", 0.5),
            alpha=get_algo_param(args, "dper_alpha", 0.6),
            eps=get_algo_param(args, "dper_eps", 1e-6),
            seed=seed,
        )
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.grad_clip = getattr(args, "grad_clip", 1.0)
        self.policy_freq = get_algo_param(args, "policy_freq", 1)
        self.target_update_interval = get_algo_param(args, "target_update_interval", 1)
        self.total_it = 0

    def _format_depth_sequence(self, depth_batch: torch.Tensor) -> torch.Tensor:
        if depth_batch.dim() == 2:
            depth_batch = depth_batch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif depth_batch.dim() == 3:
            depth_batch = depth_batch.unsqueeze(0).unsqueeze(2)
        elif depth_batch.dim() == 4:
            if depth_batch.size(0) == self.args.n_frames and depth_batch.size(1) == 1:
                depth_batch = depth_batch.unsqueeze(0)
            else:
                depth_batch = depth_batch.unsqueeze(2)
        elif depth_batch.dim() != 5:
            raise ValueError(f"Unsupported depth sequence shape: {tuple(depth_batch.shape)}")

        if depth_batch.size(2) != 1:
            raise ValueError(f"Expected single-channel sequence frames, got {tuple(depth_batch.shape)}")
        return depth_batch

    def _to_float_tensor(self, data):
        tensor = torch.as_tensor(data, device=self.device)
        return tensor if tensor.dtype == torch.float32 else tensor.float()

    def _dper_beta(self, progress_ratio=0.0) -> float:
        beta0 = float(get_algo_param(self.args, "dper_beta0", 0.4))
        beta1 = float(get_algo_param(self.args, "dper_beta1", 1.0))
        progress = float(np.clip(progress_ratio, 0.0, 1.0))
        return beta0 * (1.0 - progress) + beta1 * progress

    def _get_current_success_sample_ratio(self, progress_ratio: float) -> float:
        mu_low = float(get_algo_param(self.args, "dper_mu_low", 0.30))
        mu_mid = float(get_algo_param(self.args, "dper_mu_mid", 0.45))
        mu_high = float(get_algo_param(self.args, "dper_mu_high", 0.60))
        mu_step1 = float(get_algo_param(self.args, "dper_mu_step1", 0.25))
        mu_step2 = float(get_algo_param(self.args, "dper_mu_step2", 0.65))

        p = float(np.clip(progress_ratio, 0.0, 1.0))
        if p < mu_step1:
            mu = mu_low
        elif p < mu_step2:
            mu = mu_mid
        else:
            mu = mu_high
        return float(np.clip(mu, 0.0, 0.8))

    def _sample_replay(self, progress_ratio=0.0):
        dper_beta = self._dper_beta(progress_ratio)
        current_mu = self._get_current_success_sample_ratio(progress_ratio)
        self.replay_buffer.success_sample_ratio = current_mu
        out = self.replay_buffer.sample(self.batch_size, beta=dper_beta)
        if out is None:
            return None, None, None, {}
        samples, refs, weights, mix_info = out
        if isinstance(samples, tuple):
            stacked = samples
        else:
            stacked = tuple(np.stack(items, axis=0) for items in zip(*samples))
        replay_info = {
            "dper_beta": dper_beta,
            "replay/success_sample_ratio_target": current_mu,
            "replay/success_batch_fraction": mix_info["batch_success_fraction"],
            "replay/success_size": mix_info["success_size"],
            "replay/regular_size": mix_info["regular_size"],
        }
        return stacked, refs, weights, replay_info

    def _update_replay_priorities(self, refs, td_errors):
        self.replay_buffer.update_priorities(refs, np.asarray(td_errors, dtype=np.float32))

    def _set_critic_requires_grad(self, requires_grad: bool):
        for param in self.critic.parameters():
            param.requires_grad_(requires_grad)

    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False, progress_ratio=0.0):
        base = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        if depth_tensor.dim() < 5:
            depth_tensor = depth_tensor.unsqueeze(0)  # add batch dim for single sample

        with torch.no_grad():
            depth_seq = self._format_depth_sequence(depth_tensor)
            ao_visual, so_repr = self.visual_sub(depth_seq)
            ao_base = self.base_sub(base)
            if with_log_prob and not deterministic:
                action, log_prob = self.global_actor.action_log_prob(ao_visual, ao_base, so_repr, base)
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten(), log_prob.cpu().numpy()
            action = self.global_actor(ao_visual, ao_base, so_repr, base, deterministic=deterministic)
            real_action = self.action_scale * action + self.action_bias
            return real_action.cpu().numpy().flatten()

    def train(self, progress_ratio=0.0):
        if self.replay_buffer.size() < self.batch_size:
            return {}

        sample, replay_refs, replay_weights, replay_info = self._sample_replay(progress_ratio)
        if sample is None:
            return {}

        self.total_it += 1
        base_states, depths, actions, rewards, next_base_states, next_depths, dones = sample

        base_states = self._to_float_tensor(base_states)
        depths = self._to_float_tensor(depths)
        real_actions = self._to_float_tensor(actions)
        actions = ((real_actions - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)
        rewards = self._to_float_tensor(rewards).view(-1, 1)
        next_base_states = self._to_float_tensor(next_base_states)
        next_depths = self._to_float_tensor(next_depths)
        dones = self._to_float_tensor(dones).view(-1, 1)
        weights = self._to_float_tensor(replay_weights).view(-1, 1)

        depths_seq = self._format_depth_sequence(depths)
        next_depths_seq = self._format_depth_sequence(next_depths)

        # ----- Critic update -----
        with torch.no_grad():
            next_ao_visual, next_so_repr = self.visual_sub(next_depths_seq)
            next_ao_base = self.base_sub(next_base_states)
            next_actions, next_log_prob = self.global_actor.action_log_prob(
                next_ao_visual, next_ao_base, next_so_repr, next_base_states
            )
            next_q1, next_q2 = self.critic_target(next_depths_seq, next_base_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            target_q = rewards + (1.0 - dones) * self.gamma * (next_q - alpha * next_log_prob)

        current_q1, current_q2 = self.critic(depths_seq, base_states, actions)
        critic_loss_elements = 0.5 * (
            F.mse_loss(current_q1, target_q, reduction="none")
            + F.mse_loss(current_q2, target_q, reduction="none")
        )
        critic_loss = (critic_loss_elements * weights).mean()
        td_errors = 0.5 * ((current_q1 - target_q).abs() + (current_q2 - target_q).abs())
        target_q_mean_value = float(target_q.mean().detach().item())
        current_q_mean_value = float(torch.min(current_q1, current_q2).mean().detach().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip)
        self.critic_optimizer.step()
        self._update_replay_priorities(replay_refs, td_errors.detach().cpu().numpy().reshape(-1))

        # ----- Actor update -----
        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None
        if self.total_it % self.policy_freq == 0:
            ao_visual, so_repr = self.visual_sub(depths_seq)
            ao_base = self.base_sub(base_states)
            actions_pi, log_prob = self.global_actor.action_log_prob(ao_visual, ao_base, so_repr, base_states)

            with torch.no_grad():
                critic_so_repr = self.critic.encoder(depths_seq)

            self._set_critic_requires_grad(False)
            q1_pi, q2_pi = self.critic.q_from_repr(critic_so_repr, base_states, actions_pi)
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
            self._set_critic_requires_grad(True)
            actor_loss_value = float(actor_loss.item())

            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().detach().item())
                alpha_loss_value = float(alpha_loss.item())

        # ----- Soft updates -----
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
        result.update(replay_info)
        return result

    def _soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        checkpoint = {
            "visual_sub": self.visual_sub.state_dict(),
            "base_sub": self.base_sub.state_dict(),
            "global_actor": self.global_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "alpha": self.alpha,
        }
        if self.auto_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.visual_sub.load_state_dict(checkpoint["visual_sub"])
        self.base_sub.load_state_dict(checkpoint["base_sub"])
        self.global_actor.load_state_dict(checkpoint["global_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)
        self.alpha = checkpoint.get("alpha", self.alpha)
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
