import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..config_loader import get_algo_param
from .networks import STVimEncoder, Actor, Critic
from .buffer import DualPrioritizedReplayBuffer


class PLDPERSTVimTD3Agent:
    """
    PL + PER + ST-Vim TD3:
    - Visual encoder: STVimEncoder
    - Privileged-learning critic with distance-sensor inputs
    - Dual prioritized replay buffer (success/regular)
    """

    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"PL-DPER-Vim-TD3 Agent using device: {self.device}")
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.base_dim = base_dim
        self.depth_shape = depth_shape
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape

        self.seq_len = getattr(args, "n_frames", 4)

        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.action_scale = torch.from_numpy((self.max_action - self.min_action) / 2.0).float().to(self.device)
        self.action_bias = torch.from_numpy((self.max_action + self.min_action) / 2.0).float().to(self.device)

        self.actor_encoder = STVimEncoder(args).to(self.device)
        self.visual_feature_dim = self.actor_encoder.repr_dim
        self.actor_fused_feature_dim = self.visual_feature_dim + self.base_dim
        self.actor = Actor(
            feature_dim=self.actor_fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
        ).to(self.device)

        self.critic_encoder = STVimEncoder(args).to(self.device)
        if self.critic_encoder.repr_dim != self.visual_feature_dim:
            raise ValueError(
                f"Actor/Critic visual dims mismatch: {self.visual_feature_dim} vs {self.critic_encoder.repr_dim}"
            )

        self.critic_fused_feature_dim = self.visual_feature_dim + self.base_dim
        self.critic_1 = Critic(
            feature_dim=self.critic_fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
        ).to(self.device)
        self.critic_2 = Critic(
            feature_dim=self.critic_fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
        ).to(self.device)

        self.actor_encoder_target = copy.deepcopy(self.actor_encoder)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_encoder_target = copy.deepcopy(self.critic_encoder)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor_params = (
            list(self.actor_encoder.parameters())
           
            + list(self.actor.parameters())
        )
        self.critic_params = (
            list(self.critic_encoder.parameters())
           
            + list(self.critic_1.parameters())
            + list(self.critic_2.parameters())
        )
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = DualPrioritizedReplayBuffer(
            capacity=args.buffer_size,
            success_capacity_ratio=get_algo_param(args, "dper_td3_success_capacity_ratio", 0.3),
            success_sample_ratio=float(get_algo_param(args, "dper_td3_mu_low", 0.30)),
            alpha=get_algo_param(args, "dper_td3_alpha", 0.6),
            eps=get_algo_param(args, "dper_td3_priority_eps", 1e-6),
            seed=seed,
        )

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.grad_clip = getattr(args, "grad_clip", 1.0)
        self.batch_size = args.batch_size

        self.dper_beta_start = get_algo_param(args, "dper_td3_beta_start", 0.4)
        self.dper_beta_final = get_algo_param(args, "dper_td3_beta_final", 1.0)
        self.mu_low = float(get_algo_param(args, "dper_td3_mu_low", 0.30))
        self.mu_mid = float(get_algo_param(args, "dper_td3_mu_mid", 0.45))
        self.mu_high = float(get_algo_param(args, "dper_td3_mu_high", 0.60))
        self.mu_step1 = float(get_algo_param(args, "dper_td3_mu_step1", 0.25))
        self.mu_step2 = float(get_algo_param(args, "dper_td3_mu_step2", 0.65))

        self.exploration_noise = args.exploration_noise
        self.total_it = 0

    def _get_current_noise(self, progress_ratio: float) -> float:
        return max(float(self.exploration_noise), 1e-8)

    def _get_current_beta(self, progress_ratio: float) -> float:
        p = float(np.clip(progress_ratio, 0.0, 1.0))
        return self.dper_beta_start * (1.0 - p) + self.dper_beta_final * p

    def _get_current_success_sample_ratio(self, progress_ratio: float) -> float:
        p = float(np.clip(progress_ratio, 0.0, 1.0))
        if p < self.mu_step1:
            mu = self.mu_low
        elif p < self.mu_step2:
            mu = self.mu_mid
        else:
            mu = self.mu_high
        return float(np.clip(mu, 0.0, 0.8))

    def _to_float_tensor(self, data):
        tensor = torch.as_tensor(data, device=self.device)
        return tensor if tensor.dtype == torch.float32 else tensor.float()

    def select_action(self, base_state, depth_img, noise: bool = True, progress_ratio: float = 0.0):
        if isinstance(base_state, np.ndarray):
            base_state = torch.as_tensor(base_state, dtype=torch.float32, device=self.device)
        if isinstance(depth_img, np.ndarray):
            depth_img = torch.as_tensor(depth_img, dtype=torch.float32, device=self.device)

        depth_img = depth_img.unsqueeze(0)
        if base_state.dim() == 1:
            current_state = base_state.unsqueeze(0)
        else:
            current_state = base_state

        with torch.no_grad():
            visual_feat = self.actor_encoder(depth_img)
            actor_input = torch.cat([visual_feat, current_state], dim=-1)
            action = self.actor(actor_input).cpu().numpy().flatten()

        if noise:
            current_noise = self._get_current_noise(progress_ratio)
            action = action + self.rng.normal(0, current_noise, size=self.action_dim)

        action = np.clip(action, -1.0, 1.0)
        return action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()

    def train(self, progress_ratio: float = 0.0):
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return {}

        current_mu = self._get_current_success_sample_ratio(progress_ratio)
        self.replay_buffer.success_sample_ratio = current_mu
        beta = self._get_current_beta(progress_ratio)

        sampled = self.replay_buffer.sample(self.batch_size, beta=beta)
        if sampled is None:
            return {}

        samples, refs, importance_weights, mix_info = sampled
        if isinstance(samples, tuple):
            (
                state,
                depth,
                action,
                reward,
                next_state,
                next_depth,
                dones,
                critic_priv,
                next_critic_priv,
            ) = samples
        else:
            unpacked = zip(*samples)
            (
                state,
                depth,
                action,
                reward,
                next_state,
                next_depth,
                dones,
                critic_priv,
                next_critic_priv,
            ) = unpacked

        depth = self._to_float_tensor(depth)
        critic_priv = self._to_float_tensor(critic_priv)

        next_depth = self._to_float_tensor(next_depth)
        next_critic_priv = self._to_float_tensor(next_critic_priv)

        state = self._to_float_tensor(state)
        next_state = self._to_float_tensor(next_state)

        action = self._to_float_tensor(action)
        action = ((action - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)
        reward = self._to_float_tensor(reward).view(-1, 1)
        dones = self._to_float_tensor(dones).view(-1, 1)
        weights = self._to_float_tensor(importance_weights).view(-1, 1)

        with torch.no_grad():
            next_visual = self.actor_encoder_target(next_depth)
            next_actor_input = torch.cat([next_visual, next_state], dim=-1)
            next_action = self.actor_target(next_actor_input)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            target_visual = self.critic_encoder_target(next_critic_priv)
            target_input = torch.cat([target_visual, next_state], dim=-1)
            target_q1 = self.critic_1_target(target_input, next_action)
            target_q2 = self.critic_2_target(target_input, next_action)
            target_q = reward + (1.0 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_visual = self.critic_encoder(critic_priv)
        critic_input = torch.cat([current_visual, state], dim=-1)
        current_q1 = self.critic_1(critic_input, action)
        current_q2 = self.critic_2(critic_input, action)

        td_error = 0.5 * ((current_q1 - target_q).abs() + (current_q2 - target_q).abs())
        critic_loss = (
            weights
            * (
                F.mse_loss(current_q1, target_q, reduction="none")
                + F.mse_loss(current_q2, target_q, reduction="none")
            )
        ).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip)
        self.critic_optimizer.step()

        new_priorities = td_error.detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(refs, new_priorities)

        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:
            for param in self.critic_params:
                param.requires_grad_(False)

            actor_visual = self.actor_encoder(depth)
            actor_input = torch.cat([actor_visual, state], dim=-1)
            actor_action = self.actor(actor_input)

            with torch.no_grad():
                q_visual = self.critic_encoder(critic_priv)
            q_input = torch.cat([q_visual, state], dim=-1)
            actor_loss = -self.critic_1(q_input, actor_action).mean()
            actor_loss_value = float(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
            self.actor_optimizer.step()

            for param in self.critic_params:
                param.requires_grad_(True)

            self.soft_update(self.actor_encoder, self.actor_encoder_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic_encoder, self.critic_encoder_target, self.tau)
            self.soft_update(self.critic_1, self.critic_1_target, self.tau)
            self.soft_update(self.critic_2, self.critic_2_target, self.tau)

        result = {
            "critic_loss": float(critic_loss.item()),
            "dper_beta": float(beta),
            "replay/success_sample_ratio_target": float(current_mu),
            "replay/success_batch_fraction": mix_info["batch_success_fraction"],
            "replay/success_size": float(mix_info["success_size"]),
            "replay/regular_size": float(mix_info["regular_size"]),
        }
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        return result

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(
            {
                "actor_encoder": self.actor_encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "actor_encoder_target": self.actor_encoder_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_encoder_target": self.critic_encoder_target.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        if "actor_encoder_target" in checkpoint:
            self.actor_encoder_target.load_state_dict(checkpoint["actor_encoder_target"])
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"])
        if "critic_encoder_target" in checkpoint:
            self.critic_encoder_target.load_state_dict(checkpoint["critic_encoder_target"])
        if "critic_1_target" in checkpoint:
            self.critic_1_target.load_state_dict(checkpoint["critic_1_target"])
        if "critic_2_target" in checkpoint:
            self.critic_2_target.load_state_dict(checkpoint["critic_2_target"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "total_it" in checkpoint:
            self.total_it = checkpoint["total_it"]
