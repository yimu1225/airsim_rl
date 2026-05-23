import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..state_adapter import StateAdapter
from ..config_loader import get_algo_param
from .networks import Actor, Critic, MambaEncoder
from .buffer import ReplayBuffer


class MambaSACAgent:
    """Mamba-SAC Agent: CNN spatial features + Mamba temporal modeling."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
        self.depth_shape = depth_shape
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape
        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.max_action_tensor = torch.from_numpy(self.max_action).float().to(self.device)
        self.min_action_tensor = torch.from_numpy(self.min_action).float().to(self.device)

        scale = (self.max_action - self.min_action) / 2.0
        bias = (self.max_action + self.min_action) / 2.0
        self.action_scale = torch.from_numpy(scale).float().to(self.device)
        self.action_bias = torch.from_numpy(bias).float().to(self.device)

        self.grad_clip = getattr(args, "grad_clip", 1.0)
        self.seq_len = getattr(args, "n_frames", 4)

        self.ent_coef = get_algo_param(args, "ent_coef", 0.2)
        self.target_entropy = get_algo_param(args, "target_entropy", None)
        if self.target_entropy is None or self.target_entropy == "auto":
            self.target_entropy = -float(self.action_dim)
        else:
            self.target_entropy = float(self.target_entropy)

        self.log_alpha = None
        self.alpha_optimizer = None
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                try:
                    init_value = float(self.ent_coef.split("_")[1])
                except Exception:
                    init_value = 1.0
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=float(get_algo_param(args, "alpha_lr", args.actor_lr)))
            self.alpha = float(init_value)
            self.auto_entropy_tuning = True
        else:
            if self.ent_coef is None:
                self.alpha = get_algo_param(args, "alpha", 0.2)
            else:
                self.alpha = float(self.ent_coef)
            self.auto_entropy_tuning = False

        # Encoders
        self.actor_encoder = MambaEncoder(self.args).to(self.device)
        self.critic_encoder = MambaEncoder(self.args).to(self.device)
        self.critic_encoder_target = MambaEncoder(self.args).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())

        # State adapters
        self.actor_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target.load_state_dict(self.critic_base_adapter.state_dict())

        # State dimension
        self.state_dim = self.base_feature_dim + self.actor_encoder.repr_dim

        # Actor and Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_params = list(self.actor.parameters()) + list(self.actor_encoder.parameters()) + list(self.actor_base_adapter.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)

        self.critic_params = list(self.critic.parameters()) + list(self.critic_encoder.parameters()) + list(self.critic_base_adapter.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = ReplayBuffer(args.buffer_size, self.seq_len, seed=seed)

        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.total_it = 0
        self.policy_freq = getattr(args, "policy_freq", 2)
        self.target_update_interval = get_algo_param(args, "target_update_interval", 1)

    def _concat_state(self, base, depth, encoder_net, base_adapter, detach_encoder=False):
        base_features = base_adapter(base)
        depth_features = encoder_net(depth)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base_features, depth_features], dim=1)

    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False, progress_ratio=0.0):
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            state = self._concat_state(base_tensor, depth_tensor, self.actor_encoder, self.actor_base_adapter)
            if with_log_prob and not deterministic:
                action, log_prob, _, _ = self.actor(state, with_log_prob=True)
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten(), log_prob.cpu().numpy()
            else:
                action = self.actor.get_action(state, deterministic=deterministic)
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten()

    def train(self, progress_ratio=0.0):
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return {}

        state, depth, action, reward, next_state, next_depth, dones = self.replay_buffer.sample(self.batch_size)

        base_states = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        real_actions = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        actions = (real_actions - self.action_bias) / self.action_scale

        rewards = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        # ============ Critic Update ============
        with torch.no_grad():
            next_actor_states = self._concat_state(
                next_base_states, next_depths, self.actor_encoder, self.actor_base_adapter
            )
            next_target_states = self._concat_state(
                next_base_states, next_depths, self.critic_encoder_target, self.critic_base_adapter_target
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
        base_features = self.critic_base_adapter(base_states)
        states = torch.cat([base_features, encoded_depths], dim=1)

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss_elements = 0.5 * (
            F.mse_loss(current_q1, target_q, reduction="none")
            + F.mse_loss(current_q2, target_q, reduction="none")
        )
        critic_loss = critic_loss_elements.mean()
        td_errors = 0.5 * ((current_q1 - target_q).abs() + (current_q2 - target_q).abs())
        target_q_mean_value = float(target_q.mean().detach().item())
        current_q_mean_value = float(torch.min(current_q1, current_q2).mean().detach().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()

        # ============ Actor and Alpha Update ============
        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None

        if self.total_it % self.policy_freq == 0:
            encoded_depths_actor = self.actor_encoder(depths)
            base_features_actor = self.actor_base_adapter(base_states)
            states_actor = torch.cat([base_features_actor, encoded_depths_actor], dim=1)

            sampled_actions, log_probs, _, _ = self.actor(states_actor, with_log_prob=True)

            with torch.no_grad():
                critic_states_for_pi = self._concat_state(
                    base_states, depths, self.critic_encoder, self.critic_base_adapter
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
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        if mean_log_prob_value is not None:
            result["mean_log_prob"] = mean_log_prob_value
        if q_pi_mean_value is not None:
            result["q_pi_mean"] = q_pi_mean_value
        if alpha_loss_value is not None:
            result["alpha_loss"] = alpha_loss_value
        return result

    def _soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_base_adapter.parameters(), self.critic_base_adapter_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'actor_base_adapter': self.actor_base_adapter.state_dict(),
            'critic_base_adapter': self.critic_base_adapter.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.actor_base_adapter.load_state_dict(checkpoint['actor_base_adapter'])
        self.critic_base_adapter.load_state_dict(checkpoint['critic_base_adapter'])

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        self.critic_base_adapter_target.load_state_dict(self.critic_base_adapter.state_dict())

        if self.auto_entropy_tuning and checkpoint.get('log_alpha') is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
