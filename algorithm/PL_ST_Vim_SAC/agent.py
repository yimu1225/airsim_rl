import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..config_loader import get_algo_param
from ..state_adapter import StateAdapter
from .buffer import ReplayBuffer
from .networks import Actor, Critic, STVimEncoder


class PLSTVimSACAgent:
    """Privileged Learning ST-Vim-SAC agent.

    The actor uses base state + depth sequence only, while the critic additionally
    receives privileged distance-sensor information, following the asymmetric
    actor-critic (AAC) / privileged learning paradigm.
    """

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.args.depth_shape = depth_shape
        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
        self.critic_priv_dim = int(
            getattr(
                args,
                "critic_priv_dim",
                getattr(args, "distance_sensor_count", 108),
            )
        )
        self.depth_shape = depth_shape
        self.action_dim = action_space.shape[0]

        self.max_action = np.asarray(action_space.high, dtype=np.float32)
        self.min_action = np.asarray(action_space.low, dtype=np.float32)
        self.action_scale = torch.as_tensor((self.max_action - self.min_action) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.as_tensor((self.max_action + self.min_action) / 2.0, dtype=torch.float32, device=self.device)

        # Encoders
        self.actor_encoder = STVimEncoder(args).to(self.device)
        self.critic_encoder = STVimEncoder(args).to(self.device)
        self.critic_encoder_target = STVimEncoder(args).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())

        # State adapters
        self.actor_base_adapter = StateAdapter(base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter = StateAdapter(base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target = StateAdapter(base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target.load_state_dict(self.critic_base_adapter.state_dict())

        # State dimensions: actor only sees visual+base, critic also sees priv
        self.actor_state_dim = self.base_feature_dim + self.actor_encoder.repr_dim
        self.critic_state_dim = self.actor_state_dim + self.critic_priv_dim

        # Actor and Critic networks
        self.actor = Actor(self.actor_state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic = Critic(self.critic_state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.critic_state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizer parameter groups
        self.actor_params = (
            list(self.actor.parameters())
            + list(self.actor_encoder.parameters())
            + list(self.actor_base_adapter.parameters())
        )
        self.critic_params = (
            list(self.critic.parameters())
            + list(self.critic_encoder.parameters())
            + list(self.critic_base_adapter.parameters())
        )
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        # Entropy tuning
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

        self.replay_buffer = ReplayBuffer(args.buffer_size, seed=seed)
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.grad_clip = getattr(args, "grad_clip", 1.0)
        self.policy_freq = get_algo_param(args, "policy_freq", 1)
        self.target_update_interval = get_algo_param(args, "target_update_interval", 1)
        self.total_it = 0

    # ------------------------------------------------------------------
    #  Formatting helpers
    # ------------------------------------------------------------------
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

    def _prepare_priv(self, priv_batch: torch.Tensor) -> torch.Tensor:
        if priv_batch.dim() == 1:
            priv_batch = priv_batch.view(1, -1)
        elif priv_batch.dim() > 2:
            priv_batch = priv_batch.view(priv_batch.size(0), -1)

        if priv_batch.size(1) != self.critic_priv_dim:
            if priv_batch.size(1) > self.critic_priv_dim:
                priv_batch = priv_batch[:, : self.critic_priv_dim]
            else:
                pad = torch.zeros(
                    priv_batch.size(0),
                    self.critic_priv_dim - priv_batch.size(1),
                    device=priv_batch.device,
                    dtype=priv_batch.dtype,
                )
                priv_batch = torch.cat([priv_batch, pad], dim=1)
        return priv_batch

    # ------------------------------------------------------------------
    #  Encoding
    # ------------------------------------------------------------------
    def _encode_actor_state(self, base, depth):
        depth = self._format_depth_sequence(depth)
        base_features = self.actor_base_adapter(base)
        depth_features = self.actor_encoder(depth)
        return torch.cat([base_features, depth_features], dim=1)

    def _encode_critic_state(self, base, depth, priv):
        depth = self._format_depth_sequence(depth)
        base_features = self.critic_base_adapter(base)
        depth_features = self.critic_encoder(depth)
        priv_features = self._prepare_priv(priv)
        return torch.cat([base_features, depth_features, priv_features], dim=1)

    def _encode_critic_state_target(self, base, depth, priv):
        depth = self._format_depth_sequence(depth)
        base_features = self.critic_base_adapter_target(base)
        depth_features = self.critic_encoder_target(depth)
        priv_features = self._prepare_priv(priv)
        return torch.cat([base_features, depth_features, priv_features], dim=1)

    # ------------------------------------------------------------------
    #  Replay sampling (non-prioritized)
    # ------------------------------------------------------------------
    def _sample_replay(self):
        sample = self.replay_buffer.sample(self.batch_size)
        return sample, None, None, {}

    def _update_replay_priorities(self, refs, td_errors):
        return None

    # ------------------------------------------------------------------
    #  Action selection
    # ------------------------------------------------------------------
    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False, progress_ratio=0.0):
        base = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            state = self._encode_actor_state(base, depth_tensor)
            if with_log_prob and not deterministic:
                action, log_prob = self.actor.action_log_prob(state)
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten(), log_prob.cpu().numpy()
            action = self.actor(state, deterministic=deterministic)
            real_action = self.action_scale * action + self.action_bias
            return real_action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
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
            critic_privs,
            next_critic_privs,
        ) = sample

        base_states = torch.as_tensor(base_states, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        critic_privs = torch.as_tensor(critic_privs, dtype=torch.float32, device=self.device)
        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = ((real_actions - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_base_states, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depths, dtype=torch.float32, device=self.device)
        next_critic_privs = torch.as_tensor(next_critic_privs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        weights = None
        if replay_weights is not None:
            weights = torch.as_tensor(replay_weights, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            # Next actions from current actor (actor sees no priv)
            next_actor_state = self._encode_actor_state(next_base_states, next_depths)
            next_actions, next_log_prob = self.actor.action_log_prob(next_actor_state)

            # Target Q from target critic (critic sees priv)
            next_target_state = self._encode_critic_state_target(
                next_base_states, next_depths, next_critic_privs
            )
            next_q1, next_q2 = self.critic_target(next_target_state, next_actions)
            next_q = torch.min(next_q1, next_q2)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            target_q = rewards + (1.0 - dones) * self.gamma * (next_q - alpha * next_log_prob)

        # Current Q from critic (with priv)
        critic_state = self._encode_critic_state(base_states, depths, critic_privs)
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

        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None
        if self.total_it % self.policy_freq == 0:
            # Actor state (no priv)
            actor_state = self._encode_actor_state(base_states, depths)
            actions_pi, log_prob = self.actor.action_log_prob(actor_state)

            with torch.no_grad():
                # Critic evaluates with priv
                critic_state_for_pi = self._encode_critic_state(
                    base_states, depths, critic_privs
                )
            q1_pi, q2_pi = self.critic(critic_state_for_pi, actions_pi)
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
        if replay_info:
            result.update(replay_info)
        return result

    def _soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_base_adapter.parameters(), self.critic_base_adapter_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        checkpoint = {
            "actor_encoder": self.actor_encoder.state_dict(),
            "critic_encoder": self.critic_encoder.state_dict(),
            "critic_encoder_target": self.critic_encoder_target.state_dict(),
            "actor_base_adapter": self.actor_base_adapter.state_dict(),
            "critic_base_adapter": self.critic_base_adapter.state_dict(),
            "critic_base_adapter_target": self.critic_base_adapter_target.state_dict(),
            "actor": self.actor.state_dict(),
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
        self.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        self.critic_encoder_target.load_state_dict(checkpoint.get("critic_encoder_target", checkpoint["critic_encoder"]))
        self.actor_base_adapter.load_state_dict(checkpoint["actor_base_adapter"])
        self.critic_base_adapter.load_state_dict(checkpoint["critic_base_adapter"])
        self.critic_base_adapter_target.load_state_dict(
            checkpoint.get("critic_base_adapter_target", checkpoint["critic_base_adapter"])
        )
        self.actor.load_state_dict(checkpoint["actor"])
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
