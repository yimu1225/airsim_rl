import copy
import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..config_loader import get_algo_param
from ..state_adapter import StateAdapter
from .buffer import EpisodeReplayBuffer
from .networks import Actor, Critic, DepthCNNEncoder, MambaHistoryEncoder


class MambaRSACAgent:
    """Recurrent SAC with Mamba history tokens [visual, base, previous_action]."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        if seed is not None:
            torch.manual_seed(seed)

        self.args = copy.deepcopy(args)
        self.base_dim = int(base_dim)
        self.depth_shape = tuple(depth_shape)
        self.args.depth_shape = self.depth_shape
        self.action_dim = int(action_space.shape[0])
        self.base_feature_dim = int(getattr(self.args, "base_feature_dim", 32))
        self.visual_feature_dim = int(get_algo_param(self.args, "visual_feature_dim", 64))
        self.sequence_length = int(get_algo_param(self.args, "sequence_length", getattr(self.args, "n_frames", 16)))
        self.burn_in = max(0, int(get_algo_param(self.args, "burn_in", 0)))
        self.history_dim = int(get_algo_param(self.args, "history_dim", 128))

        self.max_action = np.asarray(action_space.high, dtype=np.float32)
        self.min_action = np.asarray(action_space.low, dtype=np.float32)
        self.action_scale = torch.as_tensor((self.max_action - self.min_action) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.as_tensor((self.max_action + self.min_action) / 2.0, dtype=torch.float32, device=self.device)

        _, depth_h, depth_w = self._normalize_depth_shape(self.depth_shape)
        self.actor_encoder = DepthCNNEncoder(depth_h, depth_w, output_dim=self.visual_feature_dim).to(self.device)
        self.critic_encoder = DepthCNNEncoder(depth_h, depth_w, output_dim=self.visual_feature_dim).to(self.device)
        self.critic_encoder_target = DepthCNNEncoder(depth_h, depth_w, output_dim=self.visual_feature_dim).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())

        self.actor_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target.load_state_dict(self.critic_base_adapter.state_dict())

        token_dim = self.visual_feature_dim + self.base_feature_dim + self.action_dim
        self.actor_history = self._make_history_encoder(token_dim).to(self.device)
        self.critic_history = self._make_history_encoder(token_dim).to(self.device)
        self.critic_history_target = self._make_history_encoder(token_dim).to(self.device)
        self.critic_history_target.load_state_dict(self.critic_history.state_dict())

        repr_dim = self.visual_feature_dim + self.base_feature_dim + self.history_dim
        self.actor = Actor(repr_dim, action_space.shape, self.args.hidden_dim).to(self.device)
        self.critic = Critic(repr_dim, action_space.shape, self.args.hidden_dim).to(self.device)
        self.critic_target = Critic(repr_dim, action_space.shape, self.args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_params = (
            list(self.actor.parameters())
            + list(self.actor_encoder.parameters())
            + list(self.actor_base_adapter.parameters())
            + list(self.actor_history.parameters())
        )
        self.critic_params = (
            list(self.critic.parameters())
            + list(self.critic_encoder.parameters())
            + list(self.critic_base_adapter.parameters())
            + list(self.critic_history.parameters())
        )
        self.actor_optimizer = Adam(self.actor_params, lr=self.args.actor_lr)
        self.critic_optimizer = Adam(self.critic_params, lr=self.args.critic_lr)

        self.ent_coef = get_algo_param(self.args, "ent_coef", 0.2)
        target_entropy = get_algo_param(self.args, "target_entropy", "auto")
        self.target_entropy = -float(self.action_dim) if target_entropy in (None, "auto") else float(target_entropy)
        self.log_alpha = None
        self.alpha_optimizer = None
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_", 1)[1])
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=float(get_algo_param(self.args, "alpha_lr", self.args.actor_lr)))
            self.alpha = float(init_value)
            self.auto_entropy_tuning = True
        else:
            self.alpha = float(self.ent_coef)
            self.auto_entropy_tuning = False

        self.replay_buffer = self._make_replay_buffer(seed)
        self.gamma = float(self.args.gamma)
        self.tau = float(self.args.tau)
        self.batch_size = int(self.args.batch_size)
        self.grad_clip = float(getattr(self.args, "grad_clip", 1.0))
        self.policy_freq = int(get_algo_param(self.args, "policy_freq", 1))
        self.target_update_interval = int(get_algo_param(self.args, "target_update_interval", 1))
        self.total_it = 0

        self.reset_history()

    @staticmethod
    def _normalize_depth_shape(depth_shape):
        if len(depth_shape) == 2:
            return (1, int(depth_shape[0]), int(depth_shape[1]))
        if len(depth_shape) == 3:
            return (int(depth_shape[0]), int(depth_shape[1]), int(depth_shape[2]))
        raise ValueError(f"Unsupported depth_shape: {depth_shape}")

    def _make_history_encoder(self, token_dim: int):
        return MambaHistoryEncoder(
            token_dim=token_dim,
            history_dim=self.history_dim,
            n_layers=int(get_algo_param(self.args, "mamba_layers", 1)),
            d_state=int(get_algo_param(self.args, "mamba_d_state", 16)),
            d_conv=int(get_algo_param(self.args, "mamba_d_conv", 4)),
            expand=int(get_algo_param(self.args, "mamba_expand", 2)),
        )

    def _make_replay_buffer(self, seed=None):
        return EpisodeReplayBuffer(
            self.args.buffer_size,
            self.sequence_length,
            seed=seed,
            store_privileged=False,
        )

    def reset_history(self):
        self._history = collections.deque(maxlen=self.sequence_length)
        self._prev_action = np.zeros((self.action_dim,), dtype=np.float32)

    def observe_action(self, action):
        self._prev_action = np.asarray(action, dtype=np.float32).reshape(self.action_dim).copy()

    def current_prev_action(self):
        return self._prev_action.copy()

    def _normalize_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        return ((action - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)

    def _format_depth_sequence(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.dim() == 2:
            depth = depth.view(1, 1, 1, depth.size(0), depth.size(1))
        elif depth.dim() == 3:
            if depth.size(0) == 1:
                depth = depth.unsqueeze(0).unsqueeze(0)
            else:
                depth = depth.unsqueeze(0).unsqueeze(2)
        elif depth.dim() == 4:
            if depth.size(1) == 1:
                depth = depth.unsqueeze(1)
            else:
                depth = depth.unsqueeze(2)
        elif depth.dim() != 5:
            raise ValueError(f"Unsupported depth sequence shape: {tuple(depth.shape)}")
        if depth.size(2) != 1:
            raise ValueError(f"Expected single-channel depth frames, got {tuple(depth.shape)}")
        return depth

    def _encode_visual_sequence(self, depth: torch.Tensor, encoder) -> torch.Tensor:
        depth = self._format_depth_sequence(depth)
        batch_size, seq_len, channels, height, width = depth.shape
        frames = depth.reshape(batch_size * seq_len, channels, height, width)
        visual = encoder(frames).view(batch_size, seq_len, -1)
        return visual

    def _encode_state_sequence(self, base, depth, prev_actions, visual_encoder, base_adapter, history_encoder):
        visual = self._encode_visual_sequence(depth, visual_encoder)
        base_features = base_adapter(base)
        tokens = torch.cat([visual, base_features, prev_actions], dim=-1)
        history = history_encoder(tokens)
        return torch.cat([visual, base_features, history], dim=-1)

    def _learning_mask(self, replay_mask: torch.Tensor) -> torch.Tensor:
        mask = replay_mask.clone()
        burn_in = min(self.burn_in, max(mask.size(1) - 1, 0))
        if burn_in > 0:
            mask[:, :burn_in] = 0.0
        if float(mask.sum().detach().item()) <= 0.0:
            mask[:, -1:] = replay_mask[:, -1:]
        return mask.squeeze(-1) > 0.5

    @staticmethod
    def _select_learning(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return tensor[mask]

    def _sample_replay(self):
        return self.replay_buffer.sample(self.batch_size)

    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False, progress_ratio=0.0):
        del progress_ratio
        base_arr = np.asarray(base_state, dtype=np.float32).reshape(self.base_dim)
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3 and depth_arr.shape[0] == 1:
            depth_arr = depth_arr[0]

        prev_norm = self._normalize_action_tensor(
            torch.as_tensor(self._prev_action, dtype=torch.float32, device=self.device).view(1, -1)
        ).cpu().numpy().reshape(-1)
        self._history.append((base_arr.copy(), depth_arr.copy(), prev_norm.copy()))

        first = self._history[0]
        padded = [first] * (self.sequence_length - len(self._history)) + list(self._history)
        base_seq = torch.as_tensor(np.stack([item[0] for item in padded], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)
        depth_seq = torch.as_tensor(np.stack([item[1] for item in padded], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(2)
        prev_seq = torch.as_tensor(np.stack([item[2] for item in padded], axis=0), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            actor_seq = self._encode_state_sequence(
                base_seq,
                depth_seq,
                prev_seq,
                self.actor_encoder,
                self.actor_base_adapter,
                self.actor_history,
            )
            actor_state = actor_seq[:, -1, :]
            if with_log_prob and not deterministic:
                action, log_prob = self.actor.action_log_prob(actor_state)
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten(), log_prob.cpu().numpy()
            action = self.actor(actor_state, deterministic=deterministic)
            real_action = self.action_scale * action + self.action_bias
            return real_action.cpu().numpy().flatten()

    def train(self, progress_ratio=0.0):
        del progress_ratio
        self.total_it += 1
        if self.replay_buffer.size() < self.batch_size:
            return {}

        sample = self._sample_replay()
        if sample is None:
            return {}
        base, depth, prev_action, action, reward, next_base, next_depth, done, replay_mask = sample

        base = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        real_prev_action = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
        prev_action_norm = self._normalize_action_tensor(real_prev_action)
        real_action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action_norm = self._normalize_action_tensor(real_action)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_base = torch.as_tensor(next_base, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)
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
                next_depth,
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
            depth,
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

    def _soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_base_adapter.parameters(), self.critic_base_adapter_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_history.parameters(), self.critic_history_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        checkpoint = {
            "actor_encoder": self.actor_encoder.state_dict(),
            "critic_encoder": self.critic_encoder.state_dict(),
            "critic_encoder_target": self.critic_encoder_target.state_dict(),
            "actor_base_adapter": self.actor_base_adapter.state_dict(),
            "critic_base_adapter": self.critic_base_adapter.state_dict(),
            "critic_base_adapter_target": self.critic_base_adapter_target.state_dict(),
            "actor_history": self.actor_history.state_dict(),
            "critic_history": self.critic_history.state_dict(),
            "critic_history_target": self.critic_history_target.state_dict(),
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
        self.actor_history.load_state_dict(checkpoint["actor_history"])
        self.critic_history.load_state_dict(checkpoint["critic_history"])
        self.critic_history_target.load_state_dict(checkpoint.get("critic_history_target", checkpoint["critic_history"]))
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)
        self.alpha = float(checkpoint.get("alpha", self.alpha))
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
