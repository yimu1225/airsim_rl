import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..state_adapter import StateAdapter
from .networks import STVimEncoder, Actor, Critic
from .buffer import ReplayBuffer
# “定位型”NaN/Inf监控


class AsymSTVimTD3Agent:
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"ST-Mamba-VimTokens-TD3 Agent using device: {self.device}")
        self.rng = np.random.default_rng(seed)

        # 设置 PyTorch 随机种子以确保网络初始化确定性
        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
        self.critic_priv_dim = int(
            getattr(
                args,
                "critic_priv_dim",
                getattr(args, "lidar_h_bins", 36) * getattr(args, "lidar_v_bins", 3),
            )
        )
        self.critic_priv_feature_dim = int(getattr(args, "critic_priv_feature_dim", 64))

        self.depth_shape = depth_shape
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape

        self.seq_len = getattr(args, "n_frames", 4)

        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.max_action_tensor = torch.from_numpy(self.max_action).float().to(self.device)
        self.min_action_tensor = torch.from_numpy(self.min_action).float().to(self.device)
        self.action_scale = torch.from_numpy((self.max_action - self.min_action) / 2.0).float().to(self.device)
        self.action_bias = torch.from_numpy((self.max_action + self.min_action) / 2.0).float().to(self.device)

        self.actor_encoder = STVimEncoder(args).to(self.device)
        self.actor_base_net = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.visual_feature_dim = self.actor_encoder.repr_dim
        self.actor_fused_feature_dim = self.visual_feature_dim + self.base_feature_dim
        self.actor = Actor(
            feature_dim=self.actor_fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.critic_encoder = STVimEncoder(args).to(self.device)
        self.critic_base_net = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_priv_net = StateAdapter(self.critic_priv_dim, self.critic_priv_feature_dim).to(self.device)
        if self.critic_encoder.repr_dim != self.visual_feature_dim:
            raise ValueError(
                f"Actor/Critic visual dims mismatch: {self.visual_feature_dim} vs {self.critic_encoder.repr_dim}"
            )

        self.critic_fused_feature_dim = self.visual_feature_dim + self.base_feature_dim + self.critic_priv_feature_dim
        self.critic_1 = Critic(
            feature_dim=self.critic_fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)
        self.critic_2 = Critic(
            feature_dim=self.critic_fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.actor_encoder_target = copy.deepcopy(self.actor_encoder)
        self.actor_base_net_target = copy.deepcopy(self.actor_base_net)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_encoder_target = copy.deepcopy(self.critic_encoder)
        self.critic_base_net_target = copy.deepcopy(self.critic_base_net)
        self.critic_priv_net_target = copy.deepcopy(self.critic_priv_net)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor_optimizer = Adam(
            list(self.actor_encoder.parameters()) + list(self.actor_base_net.parameters()) + list(self.actor.parameters()),
            lr=args.actor_lr
        )
        self.critic_optimizer = Adam(
            list(self.critic_encoder.parameters())
            + list(self.critic_base_net.parameters())
            + list(self.critic_priv_net.parameters())
            + list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            lr=args.critic_lr
        )

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.grad_clip = args.grad_clip
        self.exploration_noise = args.exploration_noise
        self.exploration_noise_final = getattr(args, "exploration_noise_final", 0.05)
        self.batch_size = args.batch_size

        self.batch_size = args.batch_size
        self.replay_buffer = ReplayBuffer(args.buffer_size, self.seq_len, seed=seed)
        self.total_it = 0

    def _assert_finite_tensor(self, name: str, tensor: torch.Tensor):
        if torch.is_tensor(tensor):
            finite_mask = torch.isfinite(tensor)
            if not finite_mask.all():
                total = tensor.numel()
                non_finite = total - int(finite_mask.sum().item())
                raise FloatingPointError(
                    f"[NaNMonitor][{name}] non-finite detected: {non_finite}/{total} elements. "
                    f"shape={tuple(tensor.shape)}"
                )

    def _assert_finite_array(self, name: str, array: np.ndarray):
        finite_mask = np.isfinite(array)
        if not finite_mask.all():
            total = array.size
            non_finite = total - int(finite_mask.sum())
            raise FloatingPointError(
                f"[NaNMonitor][{name}] non-finite detected: {non_finite}/{total} elements. "
                f"shape={array.shape}"
            )

    def _scale_action(self, action):
        return action * self.action_scale + self.action_bias

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

    def _get_current_noise(self, progress_ratio: float) -> float:
        success_rate = float(np.clip(progress_ratio, 0.0, 1.0))
        noise_max = max(float(self.exploration_noise), 1e-8)
        noise_min = min(max(float(self.exploration_noise_final), 1e-8), noise_max)

        if success_rate <= 0.5:
            return noise_max

        s_norm = float(np.clip((success_rate - 0.5) / 0.5, 0.0, 1.0))
        eta_g = noise_max / 2.0
        safe_term = max(1.0 - s_norm, 1e-6)
        noise = eta_g * (2.0 + np.log2(safe_term))
        return float(np.clip(noise, noise_min, noise_max))

    def select_action(self, base_state, depth_img, noise: bool = True, progress_ratio: float = 0.0):
        if isinstance(base_state, np.ndarray):
            base_state = torch.as_tensor(base_state, dtype=torch.float32, device=self.device)
        if isinstance(depth_img, np.ndarray):
            depth_img = torch.as_tensor(depth_img, dtype=torch.float32, device=self.device)

        # Add batch dimension to depth sequence
        depth_img = depth_img.unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W)

        # Ensure base_state has batch dimension
        if base_state.dim() == 1:
            current_state = base_state.unsqueeze(0)  # (dim,) -> (1, dim)
        else:
            current_state = base_state

        with torch.no_grad():
            visual_feat = self.actor_encoder(depth_img)
            self._assert_finite_tensor("select_action.visual_feat", visual_feat)
            base_feat = self.actor_base_net(current_state)
            actor_input = torch.cat([visual_feat, base_feat], dim=-1)
            action = self.actor(actor_input).cpu().numpy().flatten()
            self._assert_finite_array("select_action.actor_output", action)

        if noise:
            current_noise = self._get_current_noise(progress_ratio)
            noise = self.rng.normal(0, current_noise, size=self.action_dim)
            action = action + noise
            self._assert_finite_array("select_action.action_plus_noise", action)

        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        self._assert_finite_array("select_action.scaled_action", scaled_action)
        return scaled_action

    def train(self, progress_ratio: float = 0.0):
        self.total_it += 1

        (
            state,
            actor_depth,
            critic_depth,
            critic_priv,
            action,
            reward,
            next_state,
            next_actor_depth,
            next_critic_depth,
            next_critic_priv,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        actor_depth = torch.as_tensor(actor_depth, dtype=torch.float32, device=self.device)
        critic_depth = torch.as_tensor(critic_depth, dtype=torch.float32, device=self.device)
        critic_priv = torch.as_tensor(critic_priv, dtype=torch.float32, device=self.device)

        next_actor_depth = torch.as_tensor(next_actor_depth, dtype=torch.float32, device=self.device)
        next_critic_depth = torch.as_tensor(next_critic_depth, dtype=torch.float32, device=self.device)
        next_critic_priv = torch.as_tensor(next_critic_priv, dtype=torch.float32, device=self.device)

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = (action - self.action_bias) / self.action_scale
        action = action.clamp(-1.0, 1.0)

        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_visual = self.actor_encoder_target(next_actor_depth)
            self._assert_finite_tensor("train.next_visual", next_visual)
            next_base_actor = self.actor_base_net_target(next_state)
            next_actor_input = torch.cat([next_visual, next_base_actor], dim=-1)
            next_action = self.actor_target(next_actor_input)
            self._assert_finite_tensor("train.next_action_raw", next_action)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            self._assert_finite_tensor("train.next_action_noisy", next_action)

            target_visual = self.critic_encoder_target(next_critic_depth)
            self._assert_finite_tensor("train.target_visual", target_visual)
            target_base = self.critic_base_net_target(next_state)
            target_priv = self.critic_priv_net_target(self._prepare_priv(next_critic_priv))
            target_input = torch.cat([target_visual, target_base, target_priv], dim=-1)
            target_Q1 = self.critic_1_target(target_input, next_action)
            target_Q2 = self.critic_2_target(target_input, next_action)
            self._assert_finite_tensor("train.target_Q1", target_Q1)
            self._assert_finite_tensor("train.target_Q2", target_Q2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - dones) * self.gamma * target_Q
            self._assert_finite_tensor("train.target_Q", target_Q)

        current_visual = self.critic_encoder(critic_depth)
        self._assert_finite_tensor("train.current_visual", current_visual)
        current_base = self.critic_base_net(state)
        current_priv = self.critic_priv_net(self._prepare_priv(critic_priv))
        critic_input = torch.cat([current_visual, current_base, current_priv], dim=-1)
        current_Q1 = self.critic_1(critic_input, action)
        current_Q2 = self.critic_2(critic_input, action)
        self._assert_finite_tensor("train.current_Q1", current_Q1)
        self._assert_finite_tensor("train.current_Q2", current_Q2)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self._assert_finite_tensor("train.critic_loss", critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_encoder.parameters())
            + list(self.critic_base_net.parameters())
            + list(self.critic_priv_net.parameters())
            + list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            self.grad_clip,
        )
        
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        if self.total_it % self.policy_freq == 0:
            actor_visual = self.actor_encoder(actor_depth)
            self._assert_finite_tensor("train.actor_visual", actor_visual)
            actor_base = self.actor_base_net(state)
            actor_input = torch.cat([actor_visual, actor_base], dim=-1)
            actor_action = self.actor(actor_input)
            self._assert_finite_tensor("train.actor_action_raw", actor_action)

            critic_params = (
                list(self.critic_encoder.parameters())
                + list(self.critic_base_net.parameters())
                + list(self.critic_priv_net.parameters())
                + list(self.critic_1.parameters())
                + list(self.critic_2.parameters())
            )
            critic_requires_grad = [p.requires_grad for p in critic_params]
            for p in critic_params:
                p.requires_grad_(False)

            try:
                with torch.no_grad():
                    q_visual = self.critic_encoder(critic_depth)
                    q_base = self.critic_base_net(state)
                    q_priv = self.critic_priv_net(self._prepare_priv(critic_priv))
                self._assert_finite_tensor("train.q_visual", q_visual)
                q_input = torch.cat([q_visual, q_base, q_priv], dim=-1)
                actor_loss = -self.critic_1(q_input, actor_action).mean()
                self._assert_finite_tensor("train.actor_loss", actor_loss)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor_encoder.parameters()) + list(self.actor_base_net.parameters()) + list(self.actor.parameters()),
                    self.grad_clip
                )
                self.actor_optimizer.step()
            finally:
                for p, req in zip(critic_params, critic_requires_grad):
                    p.requires_grad_(req)

            self.soft_update(self.actor_encoder, self.actor_encoder_target, self.tau)
            self.soft_update(self.actor_base_net, self.actor_base_net_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic_encoder, self.critic_encoder_target, self.tau)
            self.soft_update(self.critic_base_net, self.critic_base_net_target, self.tau)
            self.soft_update(self.critic_priv_net, self.critic_priv_net_target, self.tau)
            self.soft_update(self.critic_1, self.critic_1_target, self.tau)
            self.soft_update(self.critic_2, self.critic_2_target, self.tau)

            actor_loss_value = actor_loss.item()

        return {}

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        # Store all components in a single checkpoint dictionary for convenience
        torch.save(
            {
                "actor_encoder": self.actor_encoder.state_dict(),
                "actor_base_net": self.actor_base_net.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic_base_net": self.critic_base_net.state_dict(),
                "critic_priv_net": self.critic_priv_net.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "actor_encoder_target": self.actor_encoder_target.state_dict(),
                "actor_base_net_target": self.actor_base_net_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_encoder_target": self.critic_encoder_target.state_dict(),
                "critic_base_net_target": self.critic_base_net_target.state_dict(),
                "critic_priv_net_target": self.critic_priv_net_target.state_dict(),
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
        if "actor_base_net" in checkpoint:
            self.actor_base_net.load_state_dict(checkpoint["actor_base_net"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        if "critic_base_net" in checkpoint:
            self.critic_base_net.load_state_dict(checkpoint["critic_base_net"])
        if "critic_priv_net" in checkpoint:
            self.critic_priv_net.load_state_dict(checkpoint["critic_priv_net"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        if "actor_encoder_target" in checkpoint:
            self.actor_encoder_target.load_state_dict(checkpoint["actor_encoder_target"])
        if "actor_base_net_target" in checkpoint:
            self.actor_base_net_target.load_state_dict(checkpoint["actor_base_net_target"])
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"])
        if "critic_encoder_target" in checkpoint:
            self.critic_encoder_target.load_state_dict(checkpoint["critic_encoder_target"])
        if "critic_base_net_target" in checkpoint:
            self.critic_base_net_target.load_state_dict(checkpoint["critic_base_net_target"])
        if "critic_priv_net_target" in checkpoint:
            self.critic_priv_net_target.load_state_dict(checkpoint["critic_priv_net_target"])
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


STVimTD3Agent = AsymSTVimTD3Agent
