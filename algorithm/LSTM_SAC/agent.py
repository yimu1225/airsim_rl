import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..config_loader import get_algo_param
from .buffer import ReplayBuffer
from .networks import PaperFeatureExtractor, LSTMSACActor, LSTMSACCritic


class LSTMSACAgent:
    """LSTM-SAC with the paper's self-supervised attention feature extractor."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.args = copy.deepcopy(args)
        self.base_dim = int(base_dim)
        self.depth_shape = depth_shape
        self.seq_len = int(getattr(self.args, "n_frames", 1))
        self.action_dim = action_space.shape[0]

        self.max_action = np.asarray(action_space.high, dtype=np.float32)
        self.min_action = np.asarray(action_space.low, dtype=np.float32)
        self.action_scale = torch.as_tensor((self.max_action - self.min_action) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.as_tensor((self.max_action + self.min_action) / 2.0, dtype=torch.float32, device=self.device)

        _, depth_h, depth_w = depth_shape
        feature_dim = int(get_algo_param(self.args, "lstm_sac_feature_dim", 64))
        hidden_dim = int(get_algo_param(self.args, "lstm_sac_hidden_dim", 512))
        self.feature_loss_weight = float(get_algo_param(self.args, "lstm_sac_feature_loss_weight", 1.0))
        self.kl_weight = float(get_algo_param(self.args, "lstm_sac_reconstruction_kl_weight", 0.5))

        self.feature_extractor = PaperFeatureExtractor(depth_h, depth_w, feature_dim=feature_dim).to(self.device)
        self.feature_extractor_target = copy.deepcopy(self.feature_extractor).to(self.device)
        self.base_norm = nn.LayerNorm(self.base_dim, elementwise_affine=False).to(self.device)
        self.state_dim = feature_dim + self.base_dim

        self.actor = LSTMSACActor(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_1 = LSTMSACCritic(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_2 = LSTMSACCritic(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2).to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.args.critic_lr,
        )
        self.feature_optimizer = Adam(
            self.feature_extractor.parameters(),
            lr=self.args.critic_lr,
        )

        self.ent_coef = get_algo_param(self.args, "ent_coef", 0.2)
        target_entropy = get_algo_param(self.args, "target_entropy", "auto")
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
            self.alpha_optimizer = Adam([self.log_alpha], lr=float(get_algo_param(self.args, "alpha_lr", self.args.actor_lr)))
            self.alpha = float(init_value)
            self.auto_entropy_tuning = True
        else:
            self.alpha = float(self.ent_coef)
            self.auto_entropy_tuning = False

        self.replay_buffer = ReplayBuffer(self.args.buffer_size, seed=seed)
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.batch_size = self.args.batch_size
        self.grad_clip = getattr(self.args, "grad_clip", 1.0)
        self.policy_freq = int(get_algo_param(self.args, "policy_freq", 1))
        self.target_update_interval = int(get_algo_param(self.args, "target_update_interval", 1))
        self.total_it = 0

        # ── Curriculum: paper-style 4-stage feature pre-training ──
        # progress_ratio thresholds: 0, 0.25, 0.50, 0.75
        # At each threshold, feature extractor is trained for
        # lstm_sac_feature_train_steps gradient updates, then frozen.
        self.learning_starts = int(getattr(self.args, "learning_starts", 0))
        self.feature_train_steps = int(
            get_algo_param(self.args, "lstm_sac_feature_train_steps", 2500)
        )
        self._current_stage = -1
        self._feature_steps_in_stage = 0
        self._feature_only_mode = False

    def _format_depth_sequence(self, depth_batch: torch.Tensor) -> torch.Tensor:
        if depth_batch.dim() == 2:
            depth_batch = depth_batch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif depth_batch.dim() == 3:
            depth_batch = depth_batch.unsqueeze(0).unsqueeze(2)
        elif depth_batch.dim() == 4:
            if depth_batch.size(0) == self.seq_len and depth_batch.size(1) == 1:
                depth_batch = depth_batch.unsqueeze(0)
            else:
                depth_batch = depth_batch.unsqueeze(2)
        elif depth_batch.dim() != 5:
            raise ValueError(f"Unsupported depth sequence shape: {tuple(depth_batch.shape)}")
        if depth_batch.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {depth_batch.size(1)}")
        if depth_batch.size(2) != 1:
            raise ValueError(f"Expected single-channel depth frames, got {tuple(depth_batch.shape)}")
        return depth_batch

    def _format_base_sequence(self, base: torch.Tensor, batch_size: int) -> torch.Tensor:
        if base.dim() == 1:
            return base.view(1, 1, self.base_dim).expand(batch_size, self.seq_len, self.base_dim)
        if base.dim() == 2:
            if base.shape == (self.seq_len, self.base_dim):
                return base.unsqueeze(0)
            if base.shape == (batch_size, self.base_dim):
                return base.unsqueeze(1).expand(batch_size, self.seq_len, self.base_dim)
        if base.dim() == 3 and base.shape == (batch_size, self.seq_len, self.base_dim):
            return base
        raise ValueError(f"Unsupported base sequence shape: {tuple(base.shape)}")

    def _state_sequence(self, base: torch.Tensor, depth: torch.Tensor, extractor=None, detach_features: bool = True):
        depth = self._format_depth_sequence(depth)
        base = self._format_base_sequence(base, depth.size(0))
        extractor = extractor if extractor is not None else self.feature_extractor
        if detach_features:
            with torch.no_grad():
                image_features = extractor(depth)
        else:
            image_features = extractor(depth)
        base_features = self.base_norm(base)
        if detach_features:
            base_features = base_features.detach()
        return torch.cat([image_features, base_features], dim=-1)

    def _feature_loss(self, depth: torch.Tensor):
        depth = self._format_depth_sequence(depth)
        batch_size, seq_len, channels, height, width = depth.shape
        frames = depth.reshape(batch_size * seq_len, channels, height, width)
        return self.feature_extractor.reconstruction_loss(frames, kl_weight=self.kl_weight)

    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False, progress_ratio=0.0):
        base = torch.as_tensor(base_state, dtype=torch.float32, device=self.device)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            state_seq = self._state_sequence(base, depth_tensor, detach_features=True)
            if with_log_prob and not deterministic:
                action, log_prob = self.actor.action_log_prob(state_seq)
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten(), log_prob.cpu().numpy()
            action = self.actor(state_seq, deterministic=deterministic)
            real_action = self.action_scale * action + self.action_bias
            return real_action.cpu().numpy().flatten()

    def _update_curriculum_stage(self, progress_ratio: float):
        """Determine curriculum stage and feature-training mode based on progress_ratio.

        Paper-style 4 stages:
          Stage 0: progress_ratio < 0.25  → train feature 2500 updates, then freeze
          Stage 1: 0.25 ≤ progress_ratio < 0.50 → train feature 2500 updates, then freeze
          Stage 2: 0.50 ≤ progress_ratio < 0.75 → train feature 2500 updates, then freeze
          Stage 3: 0.75 ≤ progress_ratio → train feature 2500 updates, then freeze

        Feature training only happens during the first ``feature_train_steps``
        gradient updates of each stage, AFTER ``learning_starts`` env steps.

        Returns dict with training-mode info.
        """
        # Map progress_ratio → stage (0,1,2,3)
        ratio = float(progress_ratio)
        new_stage = min(int(ratio / 0.25), 3) if ratio < 1.0 else 3

        stage_changed = new_stage != self._current_stage
        if stage_changed:
            self._feature_steps_in_stage = 0
        self._current_stage = new_stage

        # Feature-only mode: first feature_train_steps gradient updates of
        # this stage, but only after learning_starts env steps are done.
        in_feature_window = (
            self._feature_steps_in_stage < self.feature_train_steps
        )
        self._feature_only_mode = bool(in_feature_window)

        return {
            "stage": new_stage,
            "stage_changed": stage_changed,
            "feature_only": self._feature_only_mode,
        }

    def train(self, progress_ratio=0.0):
        self.total_it += 1
        if self.replay_buffer.size() < self.batch_size:
            return {}

        sample = self.replay_buffer.sample(self.batch_size)
        if sample is None:
            return {}
        base_states, depths, actions, rewards, next_base_states, next_depths, dones = sample

        base_states = torch.as_tensor(base_states, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = ((real_actions - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_base_states, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depths, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        # ── Paper-style curriculum stage ──
        cur = self._update_curriculum_stage(progress_ratio)
        feature_only = cur["feature_only"]

        # Feature extractor update (only backpropped during feature window)
        if self.feature_loss_weight > 0.0 and feature_only:
            feature_loss, recon_loss, kl_loss = self._feature_loss(depths)
            self.feature_optimizer.zero_grad()
            (self.feature_loss_weight * feature_loss).backward()
            nn.utils.clip_grad_norm_(
                self.feature_extractor.parameters(),
                self.grad_clip,
            )
            self.feature_optimizer.step()
        else:
            with torch.no_grad():
                feature_loss, recon_loss, kl_loss = self._feature_loss(depths)

        # ── Paper: freeze policy during feature-only phase ──
        if feature_only:
            self._feature_steps_in_stage += 1
            return {
                "stage": cur["stage"],
                "feature_only": 1.0,
                "feature_loss": float(feature_loss.detach().item()),
                "recon_loss": float(recon_loss.detach().item()),
                "kl_loss": float(kl_loss.detach().item()),
                "critic_loss": 0.0,
                "alpha": float(self.alpha),
                "target_q_mean": 0.0,
                "current_q_mean": 0.0,
            }

        # ── Normal policy update (policy_only_mode) ──
        with torch.no_grad():
            next_actor_state = self._state_sequence(next_base_states, next_depths, detach_features=True)
            next_actions, next_log_prob = self.actor.action_log_prob(next_actor_state)
            next_target_state = self._state_sequence(
                next_base_states,
                next_depths,
                extractor=self.feature_extractor_target,
                detach_features=True,
            )
            next_q1 = self.critic_1_target(next_target_state, next_actions)
            next_q2 = self.critic_2_target(next_target_state, next_actions)
            next_q = torch.min(next_q1, next_q2)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            target_q = rewards + (1.0 - dones) * self.gamma * (next_q - alpha * next_log_prob)

        critic_state = self._state_sequence(base_states, depths, detach_features=True)
        current_q1 = self.critic_1(critic_state, actions)
        current_q2 = self.critic_2(critic_state, actions)
        critic_loss = 0.5 * (
            F.mse_loss(current_q1, target_q)
            + F.mse_loss(current_q2, target_q)
        )
        target_q_mean_value = float(target_q.mean().detach().item())
        current_q_mean_value = float(torch.min(current_q1, current_q2).mean().detach().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.grad_clip,
        )
        self.critic_optimizer.step()

        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None
        if self.total_it % self.policy_freq == 0:
            actor_state = self._state_sequence(base_states, depths, detach_features=True)
            actions_pi, log_prob = self.actor.action_log_prob(actor_state)
            q1_pi = self.critic_1(actor_state, actions_pi)
            q2_pi = self.critic_2(actor_state, actions_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            alpha = self.log_alpha.exp().detach() if self.auto_entropy_tuning else torch.tensor(
                self.alpha, dtype=torch.float32, device=self.device
            )
            actor_loss = (alpha * log_prob - min_q_pi).mean()
            mean_log_prob_value = float(log_prob.mean().detach().item())
            q_pi_mean_value = float(min_q_pi.mean().detach().item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
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
            "stage": cur["stage"],
            "feature_only": 0.0,
            "critic_loss": float(critic_loss.item()),
            "alpha": float(self.alpha),
            "target_q_mean": target_q_mean_value,
            "current_q_mean": current_q_mean_value,
            "feature_loss": float(feature_loss.detach().item()),
            "recon_loss": float(recon_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
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
        for param, target_param in zip(self.feature_extractor.parameters(), self.feature_extractor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        checkpoint = {
            "feature_extractor": self.feature_extractor.state_dict(),
            "feature_extractor_target": self.feature_extractor_target.state_dict(),
            "base_norm": self.base_norm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "feature_optimizer": self.feature_optimizer.state_dict(),
            "total_it": self.total_it,
            "alpha": self.alpha,
            "curriculum_stage": self._current_stage,
        }
        if self.auto_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.feature_extractor_target.load_state_dict(
            checkpoint.get("feature_extractor_target", checkpoint["feature_extractor"])
        )
        self.base_norm.load_state_dict(checkpoint["base_norm"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_1_target.load_state_dict(checkpoint.get("critic_1_target", checkpoint["critic_1"]))
        self.critic_2_target.load_state_dict(checkpoint.get("critic_2_target", checkpoint["critic_2"]))
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "feature_optimizer" in checkpoint:
            self.feature_optimizer.load_state_dict(checkpoint["feature_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)
        self.alpha = checkpoint.get("alpha", self.alpha)
        self._current_stage = checkpoint.get("curriculum_stage", -1)
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


SACAgent = LSTMSACAgent
