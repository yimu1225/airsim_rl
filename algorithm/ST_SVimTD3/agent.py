import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..state_adapter import StateAdapter
from .networks import STVimEncoder, Actor, Critic, SafetyConstraintHead, safety_project_actions
from .buffer import ReplayBuffer


class STSVimTD3Agent:
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"ST-Mamba-VimTokens-Safety-TD3 Agent using device: {self.device}")
        self.rng = np.random.default_rng(seed)

        # 设置 PyTorch 随机种子以确保网络初始化确定性
        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
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
        self.fused_feature_dim = self.visual_feature_dim + self.base_feature_dim
        self.actor = Actor(
            feature_dim=self.fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.critic_encoder = STVimEncoder(args).to(self.device)
        self.critic_base_net = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        if self.critic_encoder.repr_dim != self.visual_feature_dim:
            raise ValueError(
                f"Actor/Critic visual dims mismatch: {self.visual_feature_dim} vs {self.critic_encoder.repr_dim}"
            )
        self.critic = Critic(
            feature_dim=self.fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.use_safety_layer = getattr(args, "use_vim_safety_layer", True)
        self.safety_model = SafetyConstraintHead(
            latent_dim=self.fused_feature_dim,
            action_dim=self.action_dim
        ).to(self.device)

        self.actor_encoder_target = copy.deepcopy(self.actor_encoder)
        self.actor_base_net_target = copy.deepcopy(self.actor_base_net)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_encoder_target = copy.deepcopy(self.critic_encoder)
        self.critic_base_net_target = copy.deepcopy(self.critic_base_net)
        self.critic_target = copy.deepcopy(self.critic)
        self.safety_model_target = copy.deepcopy(self.safety_model)

        self.actor_optimizer = Adam(
            list(self.actor_encoder.parameters()) + list(self.actor_base_net.parameters()) + list(self.actor.parameters()),
            lr=args.actor_lr
        )
        self.critic_optimizer = Adam(
            list(self.critic.parameters())
            + list(self.critic_encoder.parameters())
            + list(self.critic_base_net.parameters()),
            lr=args.critic_lr
        )

        self.safety_end_to_end = getattr(args, "safety_end_to_end", False)
        safety_params = list(self.safety_model.parameters())
        if self.safety_end_to_end:
            safety_params += list(self.actor_encoder.parameters())
            safety_params += list(self.actor_base_net.parameters())
        self.safety_optimizer = Adam(safety_params, lr=getattr(args, "safety_lr", args.actor_lr))

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.grad_clip = getattr(args, "grad_clip", 1.0)

        self.exploration_noise = args.exploration_noise
        self.exploration_noise_final = getattr(args, "exploration_noise_final", 0.05)

        self.safety_loss_coef = getattr(args, "safety_loss_coef", 1.0)
        self.safety_actor_penalty_coef = getattr(args, "safety_actor_penalty_coef", 0.05)
        self.safety_warmup_steps = getattr(args, "safety_warmup_steps", 0)
        self.safety_label_mode = getattr(args, "safety_label_mode", "collision")

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

    def _unscale_action(self, scaled_action):
        return (scaled_action - self.action_bias) / (self.action_scale + 1e-6)

    def _apply_safety_projection(self, raw_action, visual_feat, current_state):
        if not self.use_safety_layer:
            return raw_action, None
        safety_input = torch.cat([visual_feat, current_state], dim=-1)
        g, h = self.safety_model(safety_input)
        safe_action, violation = safety_project_actions(raw_action, g, h)
        safe_action = safe_action.clamp(-1.0, 1.0)
        return safe_action, violation

    def _get_current_noise(self, progress_ratio: float) -> float:
        current_noise = self.exploration_noise * (1 - progress_ratio) + self.exploration_noise_final * progress_ratio
        return current_noise

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
            base_feat = self.actor_base_net(current_state)
            self._assert_finite_tensor("select_action.visual_feat", visual_feat)
            self._assert_finite_tensor("select_action.base_feat", base_feat)

            actor_input = torch.cat([visual_feat, base_feat], dim=-1)
            raw_action = self.actor(actor_input)
            safe_action, _ = self._apply_safety_projection(raw_action, visual_feat, base_feat)
            action = safe_action.cpu().numpy().flatten()
            self._assert_finite_array("select_action.actor_output", action)

        if noise:
            current_noise = self._get_current_noise(progress_ratio)
            noise_arr = self.rng.normal(0, current_noise, size=self.action_dim)
            action = action + noise_arr
            self._assert_finite_array("select_action.action_plus_noise", action)

        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        self._assert_finite_array("select_action.scaled_action", scaled_action)
        return scaled_action

    def train(self, progress_ratio: float = 0.0):
        # progress_ratio parameter kept for uniform agent interface. Not used
        # in this implementation but may be utilized for noise scheduling or
        # safety adjustments in future.
        self.total_it += 1

        (state, depth, action, reward,
         next_state, next_depth, dones, collision_flag) = self.replay_buffer.sample(self.batch_size)

        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

        # No need for additional assignments, use state and next_state directly

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = self._unscale_action(action).clamp(-1.0, 1.0)

        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        if collision_flag is not None:
            collision_flag = torch.as_tensor(collision_flag, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_visual = self.actor_encoder_target(next_depth)
            next_base_actor = self.actor_base_net_target(next_state)
            self._assert_finite_tensor("train.next_visual", next_visual)
            self._assert_finite_tensor("train.next_base_actor", next_base_actor)
            next_actor_input = torch.cat([next_visual, next_base_actor], dim=-1)
            next_action_raw = self.actor_target(next_actor_input)

            if self.use_safety_layer:
                next_safety_input = torch.cat([next_visual, next_base_actor], dim=-1)
                next_g, next_h = self.safety_model_target(next_safety_input)
                next_action_raw, _ = safety_project_actions(next_action_raw, next_g, next_h)
                next_action_raw = next_action_raw.clamp(-1.0, 1.0)

            self._assert_finite_tensor("train.next_action_raw", next_action_raw)
            noise = (torch.randn_like(next_action_raw) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action_raw + noise).clamp(-1.0, 1.0)
            self._assert_finite_tensor("train.next_action_noisy", next_action)

            target_visual = self.critic_encoder_target(next_depth)
            target_base = self.critic_base_net_target(next_state)
            self._assert_finite_tensor("train.target_visual", target_visual)
            self._assert_finite_tensor("train.target_base", target_base)
            target_input = torch.cat([target_visual, target_base], dim=-1)
            target_Q1, target_Q2 = self.critic_target(target_input, next_action)
            self._assert_finite_tensor("train.target_Q1", target_Q1)
            self._assert_finite_tensor("train.target_Q2", target_Q2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - dones) * self.gamma * target_Q
            self._assert_finite_tensor("train.target_Q", target_Q)

        current_visual = self.critic_encoder(depth)
        current_base = self.critic_base_net(state)
        self._assert_finite_tensor("train.current_visual", current_visual)
        self._assert_finite_tensor("train.current_base", current_base)
        critic_input = torch.cat([current_visual, current_base], dim=-1)
        current_Q1, current_Q2 = self.critic(critic_input, action)
        self._assert_finite_tensor("train.current_Q1", current_Q1)
        self._assert_finite_tensor("train.current_Q2", current_Q2)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self._assert_finite_tensor("train.critic_loss", critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.parameters())
            + list(self.critic_encoder.parameters())
            + list(self.critic_base_net.parameters()),
            self.grad_clip
        )
        self.critic_optimizer.step()

        safety_loss_value = 0.0
        safety_violation_rate = 0.0
        if self.use_safety_layer and self.total_it >= self.safety_warmup_steps:
            safety_visual = self.actor_encoder(depth)
            safety_base = self.actor_base_net(state)
            safety_input = torch.cat([safety_visual, safety_base], dim=-1)
            safety_input = safety_input if self.safety_end_to_end else safety_input.detach()
            g, h = self.safety_model(safety_input)

            logits = (g * action).sum(dim=-1, keepdim=True) + h

            if collision_flag is None:
                raise ValueError("collision_flag is required for ST-VimTD3-Safety safety supervision.")
            collision_target = ((dones > 0.5) & (collision_flag > 0.5)).float()

            safety_loss = F.binary_cross_entropy_with_logits(logits, collision_target)
            safety_loss = self.safety_loss_coef * safety_loss

            self.safety_optimizer.zero_grad()
            safety_loss.backward()
            nn.utils.clip_grad_norm_(self.safety_model.parameters(), self.grad_clip)
            self.safety_optimizer.step()

            safety_loss_value = safety_loss.item()
            with torch.no_grad():
                safety_violation_rate = (torch.sigmoid(logits) > 0.5).float().mean().item()

        actor_loss_value = 0.0
        if self.total_it % self.policy_freq == 0:
            actor_visual = self.actor_encoder(depth)
            actor_base = self.actor_base_net(state)
            self._assert_finite_tensor("train.actor_visual", actor_visual)
            self._assert_finite_tensor("train.actor_base", actor_base)
            actor_input = torch.cat([actor_visual, actor_base], dim=-1)
            actor_action_raw = self.actor(actor_input)
            self._assert_finite_tensor("train.actor_action_raw", actor_action_raw)

            safety_penalty = torch.tensor(0.0, device=self.device)
            if self.use_safety_layer:
                with torch.no_grad():
                    safety_input_actor = torch.cat([actor_visual, actor_base], dim=-1)
                    g_actor, h_actor = self.safety_model(safety_input_actor)
                actor_action_safe, actor_violation = safety_project_actions(actor_action_raw, g_actor, h_actor)
                actor_action_safe = actor_action_safe.clamp(-1.0, 1.0)
                safety_penalty = torch.clamp(actor_violation, min=0.0).mean()
            else:
                actor_action_safe = actor_action_raw

            actor_action = actor_action_safe
            self._assert_finite_tensor("train.actor_action_scaled", actor_action)

            with torch.no_grad():
                q_visual = self.critic_encoder(depth)
                q_base = self.critic_base_net(state)
            self._assert_finite_tensor("train.q_visual", q_visual)
            self._assert_finite_tensor("train.q_base", q_base)
            q_input = torch.cat([q_visual, q_base], dim=-1)
            actor_q_loss = -self.critic(q_input, actor_action)[0].mean()
            actor_loss = actor_q_loss + self.safety_actor_penalty_coef * safety_penalty
            self._assert_finite_tensor("train.actor_loss", actor_loss)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor_encoder.parameters()) + list(self.actor_base_net.parameters()) + list(self.actor.parameters()),
                self.grad_clip
            )
            self.actor_optimizer.step()

            self.soft_update(self.actor_encoder, self.actor_encoder_target, self.tau)
            self.soft_update(self.actor_base_net, self.actor_base_net_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic_encoder, self.critic_encoder_target, self.tau)
            self.soft_update(self.critic_base_net, self.critic_base_net_target, self.tau)
            self.soft_update(self.critic, self.critic_target, self.tau)
            self.soft_update(self.safety_model, self.safety_model_target, self.tau)

            actor_loss_value = actor_loss.item()

        return {
            "safety_violation_rate": safety_violation_rate
        }

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(
            {
                "actor_encoder": self.actor_encoder.state_dict(),
                "actor_base_net": self.actor_base_net.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic_base_net": self.critic_base_net.state_dict(),
                "critic": self.critic.state_dict(),
                "safety_model": self.safety_model.state_dict(),
                "actor_encoder_target": self.actor_encoder_target.state_dict(),
                "actor_base_net_target": self.actor_base_net_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_encoder_target": self.critic_encoder_target.state_dict(),
                "critic_base_net_target": self.critic_base_net_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "safety_optimizer": self.safety_optimizer.state_dict(),
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
        
        # Handle loading old models with critic_1 and critic_2
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        elif "critic_1" in checkpoint and "critic_2" in checkpoint:
            # Old model format: only load critic_1 (new structure combines both)
            self.critic.load_state_dict(checkpoint["critic_1"])
            # Load target networks if available
            if "critic_1_target" in checkpoint:
                self.critic_target.load_state_dict(checkpoint["critic_1_target"])
        
        if "safety_model" in checkpoint:
            self.safety_model.load_state_dict(checkpoint["safety_model"])
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
        if "critic_target" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target"])
        elif "critic_1_target" in checkpoint:
            # Already handled above
            pass
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "safety_optimizer" in checkpoint:
            self.safety_optimizer.load_state_dict(checkpoint["safety_optimizer"])
        if "total_it" in checkpoint:
            self.total_it = checkpoint["total_it"]
