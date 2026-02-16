import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import STVimTokenMambaEncoder, Actor, Critic, SafetyConstraintHead, safety_project_actions
from .buffer import SequenceReplayBuffer


class ST_Mamba_VimTokens_Safety_Agent:
    def __init__(self, base_dim, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"ST-Mamba-VimTokens-Safety-TD3 Agent using device: {self.device}")

        self.args = args
        self.base_dim = base_dim
        self.depth_shape = depth_shape
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape

        self.seq_len = getattr(args, "seq_len", 16)

        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.max_action_tensor = torch.from_numpy(self.max_action).float().to(self.device)
        self.min_action_tensor = torch.from_numpy(self.min_action).float().to(self.device)
        self.action_scale = torch.from_numpy((self.max_action - self.min_action) / 2.0).float().to(self.device)
        self.action_bias = torch.from_numpy((self.max_action + self.min_action) / 2.0).float().to(self.device)

        self.actor_encoder = STVimTokenMambaEncoder(
            state_dim=self.base_dim,
            action_dim=None,
            args=args
        ).to(self.device)
        self.actor = Actor(
            feature_dim=args.st_mamba_embed_dim * self.seq_len + self.base_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.critic_encoder = STVimTokenMambaEncoder(
            state_dim=self.base_dim,
            action_dim=None,
            args=args
        ).to(self.device)
        self.critic_1 = Critic(
            feature_dim=args.st_mamba_embed_dim * self.seq_len + self.base_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)
        self.critic_2 = Critic(
            feature_dim=args.st_mamba_embed_dim * self.seq_len + self.base_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.use_safety_layer = getattr(args, "use_vim_safety_layer", True)
        self.safety_model = SafetyConstraintHead(
            latent_dim=args.st_mamba_embed_dim * self.seq_len,
            action_dim=self.action_dim
        ).to(self.device)

        self.actor_encoder_target = copy.deepcopy(self.actor_encoder)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_encoder_target = copy.deepcopy(self.critic_encoder)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.safety_model_target = copy.deepcopy(self.safety_model)

        self.actor_optimizer = Adam(
            list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
            lr=args.actor_lr
        )
        self.critic_optimizer = Adam(
            list(self.critic_encoder.parameters()) + list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=args.critic_lr
        )

        self.safety_end_to_end = getattr(args, "safety_end_to_end", False)
        safety_params = list(self.safety_model.parameters())
        if self.safety_end_to_end:
            safety_params += list(self.actor_encoder.parameters())
        self.safety_optimizer = Adam(safety_params, lr=getattr(args, "safety_lr", args.actor_lr))

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.grad_clip = getattr(args, "grad_clip", 1.0)

        self.exploration_noise = args.exploration_noise

        self.safety_loss_coef = getattr(args, "safety_loss_coef", 1.0)
        self.safety_actor_penalty_coef = getattr(args, "safety_actor_penalty_coef", 0.05)
        self.safety_collision_reward_threshold = getattr(args, "safety_collision_reward_threshold", -10.0)
        self.safety_warmup_steps = getattr(args, "safety_warmup_steps", 0)
        self.safety_label_mode = getattr(args, "safety_label_mode", "collision_then_reward")

        self.batch_size = args.batch_size
        self.replay_buffer = SequenceReplayBuffer(args.buffer_size, self.seq_len)
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

    def _normalize_depth(self, depth_tensor):
        if depth_tensor.dtype != torch.float32:
            depth_tensor = depth_tensor.float()
        max_val = depth_tensor.max().item() if depth_tensor.numel() > 0 else 1.0
        if max_val > 1.0:
            depth_tensor = depth_tensor / 255.0
        return depth_tensor.clamp(0.0, 1.0)

    def _scale_action(self, action):
        return action * self.action_scale + self.action_bias

    def _unscale_action(self, scaled_action):
        return (scaled_action - self.action_bias) / (self.action_scale + 1e-6)

    def _apply_safety_projection(self, raw_action, visual_feat):
        if not self.use_safety_layer:
            return raw_action, None
        g, h = self.safety_model(visual_feat)
        safe_action, violation = safety_project_actions(raw_action, g, h)
        safe_action = safe_action.clamp(-1.0, 1.0)
        return safe_action, violation

    def select_action(self, base_state, depth_img, noise: bool = True):
        if isinstance(base_state, np.ndarray):
            base_state = torch.as_tensor(base_state, dtype=torch.float32, device=self.device)
        if isinstance(depth_img, np.ndarray):
            depth_img = torch.as_tensor(depth_img, dtype=torch.float32, device=self.device)

        if depth_img.dim() == 3:
            depth_img = depth_img.unsqueeze(1)
        if depth_img.dim() == 4:
            depth_img = depth_img.unsqueeze(0)

        depth_img = self._normalize_depth(depth_img)
        if base_state.dim() == 1:
            current_state = base_state.unsqueeze(0)
        elif base_state.dim() == 2:
            current_state = base_state[-1, :].unsqueeze(0)
        else:
            current_state = base_state[:, -1, :]

        with torch.no_grad():
            visual_feat = self.actor_encoder(depth_img, current_state)
            self._assert_finite_tensor("select_action.visual_feat", visual_feat)

            actor_input = torch.cat([visual_feat, current_state], dim=-1)
            raw_action = self.actor(actor_input)
            safe_action, _ = self._apply_safety_projection(raw_action, visual_feat)
            action = safe_action.cpu().numpy().flatten()
            self._assert_finite_array("select_action.actor_output", action)

        if noise:
            noise_arr = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise_arr
            self._assert_finite_array("select_action.action_plus_noise", action)

        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        self._assert_finite_array("select_action.scaled_action", scaled_action)
        return scaled_action

    def train(self, replay_buffer=None, batch_size=None):
        self.total_it += 1

        if batch_size is None:
            batch_size = self.batch_size

        if replay_buffer is None:
            replay_buffer = self.replay_buffer

        sampled = replay_buffer.sample(batch_size)
        if sampled is None:
            return {"critic_loss": 0.0, "actor_loss": 0.0, "safety_loss": 0.0, "safety_violation_rate": 0.0}

        if len(sampled) == 8:
            (state, depth, action, reward,
             next_state, next_depth, done_flag, collision_flag) = sampled
        else:
            (state, depth, action, reward,
             next_state, next_depth, done_flag) = sampled
            collision_flag = None

        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)
        depth = self._normalize_depth(depth)
        next_depth = self._normalize_depth(next_depth)

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

        if state.dim() == 3:
            current_state = state[:, -1, :]
        else:
            current_state = state
        if next_state.dim() == 3:
            next_state_curr = next_state[:, -1, :]
        else:
            next_state_curr = next_state

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = self._unscale_action(action).clamp(-1.0, 1.0)

        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_flag = torch.as_tensor(done_flag, dtype=torch.float32, device=self.device)
        if collision_flag is not None:
            collision_flag = torch.as_tensor(collision_flag, dtype=torch.float32, device=self.device)

        reward = reward.view(-1, 1)

        done_flag = done_flag.view(-1, 1)

        if collision_flag is not None:
            collision_flag = collision_flag.view(-1, 1)

        not_done = 1.0 - done_flag

        with torch.no_grad():
            next_visual = self.actor_encoder_target(next_depth, next_state_curr)
            self._assert_finite_tensor("train.next_visual", next_visual)
            next_actor_input = torch.cat([next_visual, next_state_curr], dim=-1)
            next_action_raw = self.actor_target(next_actor_input)

            if self.use_safety_layer:
                next_g, next_h = self.safety_model_target(next_visual)
                next_action_raw, _ = safety_project_actions(next_action_raw, next_g, next_h)
                next_action_raw = next_action_raw.clamp(-1.0, 1.0)

            self._assert_finite_tensor("train.next_action_raw", next_action_raw)
            noise = (torch.randn_like(next_action_raw) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action_raw + noise).clamp(-1.0, 1.0)
            self._assert_finite_tensor("train.next_action_noisy", next_action)

            target_visual = self.critic_encoder_target(next_depth, next_state_curr)
            self._assert_finite_tensor("train.target_visual", target_visual)
            target_input = torch.cat([target_visual, next_state_curr], dim=-1)
            target_Q1 = self.critic_1_target(target_input, next_action)
            target_Q2 = self.critic_2_target(target_input, next_action)
            self._assert_finite_tensor("train.target_Q1", target_Q1)
            self._assert_finite_tensor("train.target_Q2", target_Q2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q
            self._assert_finite_tensor("train.target_Q", target_Q)

        current_visual = self.critic_encoder(depth, current_state)
        self._assert_finite_tensor("train.current_visual", current_visual)
        critic_input = torch.cat([current_visual, current_state], dim=-1)
        current_Q1 = self.critic_1(critic_input, action)
        current_Q2 = self.critic_2(critic_input, action)
        self._assert_finite_tensor("train.current_Q1", current_Q1)
        self._assert_finite_tensor("train.current_Q2", current_Q2)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self._assert_finite_tensor("train.critic_loss", critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_encoder.parameters()) + list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.grad_clip
        )
        self.critic_optimizer.step()

        safety_loss_value = 0.0
        safety_violation_rate = 0.0
        if self.use_safety_layer and self.total_it >= self.safety_warmup_steps:
            safety_visual = self.actor_encoder(depth, current_state)
            safety_input = safety_visual if self.safety_end_to_end else safety_visual.detach()
            g, h = self.safety_model(safety_input)

            logits = (g * action).sum(dim=-1, keepdim=True) + h

            reward_proxy_target = ((done_flag > 0.5) & (reward <= self.safety_collision_reward_threshold)).float()

            if self.safety_label_mode == "collision" and collision_flag is not None:
                collision_target = ((done_flag > 0.5) & (collision_flag > 0.5)).float()
            elif self.safety_label_mode == "reward_proxy" or collision_flag is None:
                collision_target = reward_proxy_target
            else:
                real_collision_target = ((done_flag > 0.5) & (collision_flag > 0.5)).float()
                collision_target = torch.maximum(real_collision_target, reward_proxy_target)

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
            actor_visual = self.actor_encoder(depth, current_state)
            self._assert_finite_tensor("train.actor_visual", actor_visual)
            actor_input = torch.cat([actor_visual, current_state], dim=-1)
            actor_action_raw = self.actor(actor_input)
            self._assert_finite_tensor("train.actor_action_raw", actor_action_raw)

            safety_penalty = torch.tensor(0.0, device=self.device)
            if self.use_safety_layer:
                with torch.no_grad():
                    g_actor, h_actor = self.safety_model(actor_visual)
                actor_action_safe, actor_violation = safety_project_actions(actor_action_raw, g_actor, h_actor)
                actor_action_safe = actor_action_safe.clamp(-1.0, 1.0)
                safety_penalty = torch.clamp(actor_violation, min=0.0).mean()
            else:
                actor_action_safe = actor_action_raw

            actor_action = actor_action_safe
            self._assert_finite_tensor("train.actor_action_scaled", actor_action)

            q_visual = self.critic_encoder(depth, current_state)
            self._assert_finite_tensor("train.q_visual", q_visual)
            q_input = torch.cat([q_visual, current_state], dim=-1)
            actor_q_loss = -self.critic_1(q_input, actor_action).mean()
            actor_loss = actor_q_loss + self.safety_actor_penalty_coef * safety_penalty
            self._assert_finite_tensor("train.actor_loss", actor_loss)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
                self.grad_clip
            )
            self.actor_optimizer.step()

            self.soft_update(self.actor_encoder, self.actor_encoder_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic_encoder, self.critic_encoder_target, self.tau)
            self.soft_update(self.critic_1, self.critic_1_target, self.tau)
            self.soft_update(self.critic_2, self.critic_2_target, self.tau)
            self.soft_update(self.safety_model, self.safety_model_target, self.tau)

            actor_loss_value = actor_loss.item()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_value,
            "safety_loss": safety_loss_value,
            "safety_violation_rate": safety_violation_rate
        }

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor_encoder.state_dict(), filename + "_actor_encoder")
        torch.save(self.actor.state_dict(), filename + "_actor_head")
        torch.save(self.critic_encoder.state_dict(), filename + "_critic_encoder")
        torch.save(self.critic_1.state_dict(), filename + "_critic_1_head")
        torch.save(self.critic_2.state_dict(), filename + "_critic_2_head")
        torch.save(self.safety_model.state_dict(), filename + "_safety_head")

    def load(self, filename):
        self.actor_encoder.load_state_dict(torch.load(filename + "_actor_encoder"))
        self.actor.load_state_dict(torch.load(filename + "_actor_head"))
        self.critic_encoder.load_state_dict(torch.load(filename + "_critic_encoder"))
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1_head"))
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2_head"))

        safety_path = filename + "_safety_head"
        try:
            self.safety_model.load_state_dict(torch.load(safety_path))
        except FileNotFoundError:
            print(f"[Safety] checkpoint not found: {safety_path}, skip loading safety head.")
