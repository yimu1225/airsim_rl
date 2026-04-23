import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .buffer import ReplayBuffer
from .networks import Actor, Critic, VimStateSeqEncoder


class VimStateSeqTD3Agent:
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Vim-StateSeq-TD3 Agent using device: {self.device}")
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.args = copy.deepcopy(args)
        self.base_dim = int(base_dim)
        self.depth_shape = depth_shape
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape
        self.args.base_dim = self.base_dim

        self.seq_len = int(getattr(self.args, "n_frames", 4))

        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.action_scale = torch.from_numpy((self.max_action - self.min_action) / 2.0).float().to(self.device)
        self.action_bias = torch.from_numpy((self.max_action + self.min_action) / 2.0).float().to(self.device)

        self.actor_encoder = VimStateSeqEncoder(self.args).to(self.device)
        self.feature_dim = self.actor_encoder.repr_dim
        self.actor = Actor(
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            hidden_dim=self.args.hidden_dim,
        ).to(self.device)

        self.critic_encoder = VimStateSeqEncoder(self.args).to(self.device)
        self.critic_1 = Critic(
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            hidden_dim=self.args.hidden_dim,
        ).to(self.device)
        self.critic_2 = Critic(
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            hidden_dim=self.args.hidden_dim,
        ).to(self.device)

        self.actor_encoder_target = copy.deepcopy(self.actor_encoder)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_encoder_target = copy.deepcopy(self.critic_encoder)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor_optimizer = Adam(
            list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
            lr=self.args.actor_lr,
        )
        self.critic_optimizer = Adam(
            list(self.critic_encoder.parameters())
            + list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            lr=self.args.critic_lr,
        )

        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.policy_noise = self.args.policy_noise
        self.noise_clip = self.args.noise_clip
        self.policy_freq = self.args.policy_freq
        self.grad_clip = self.args.grad_clip
        self.exploration_noise = self.args.exploration_noise
        self.exploration_noise_final = getattr(self.args, "exploration_noise_final", 0.05)

        self.batch_size = self.args.batch_size
        self.replay_buffer = ReplayBuffer(self.args.buffer_size, self.seq_len, seed=seed)
        self.total_it = 0

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

    def _prepare_depth_seq(self, depth_img: torch.Tensor) -> torch.Tensor:
        if depth_img.dim() == 4:
            depth_img = depth_img.unsqueeze(0)
        if depth_img.dim() != 5:
            raise ValueError(f"Expected depth sequence shape (B,T,C,H,W), got {tuple(depth_img.shape)}")
        if depth_img.shape[1] != self.seq_len:
            raise ValueError(f"Expected depth seq_len={self.seq_len}, got {depth_img.shape[1]}")
        return depth_img

    def _prepare_base_seq(self, base_state: torch.Tensor, batch_size: int) -> torch.Tensor:
        if base_state.dim() == 1:
            if base_state.shape[0] != self.base_dim:
                raise ValueError(f"Expected base dim {self.base_dim}, got {base_state.shape[0]}")
            return base_state.view(1, 1, self.base_dim).expand(batch_size, self.seq_len, self.base_dim)

        if base_state.dim() == 2:
            if base_state.shape == (self.seq_len, self.base_dim):
                if batch_size != 1:
                    raise ValueError(
                        f"Ambiguous base shape {tuple(base_state.shape)} for batch_size={batch_size}; expected (B, base_dim)."
                    )
                return base_state.unsqueeze(0)
            if base_state.shape == (batch_size, self.base_dim):
                return base_state.unsqueeze(1).expand(batch_size, self.seq_len, self.base_dim)
            raise ValueError(
                f"Invalid base shape {tuple(base_state.shape)}; expected (T,{self.base_dim}) or (B,{self.base_dim})."
            )

        if base_state.dim() == 3:
            expected = (batch_size, self.seq_len, self.base_dim)
            if tuple(base_state.shape) != expected:
                raise ValueError(f"Invalid base sequence shape {tuple(base_state.shape)}; expected {expected}.")
            return base_state

        raise ValueError(f"Unsupported base_state dim={base_state.dim()}")

    def select_action(self, base_state, depth_img, noise: bool = True, progress_ratio: float = 0.0):
        if isinstance(base_state, np.ndarray):
            base_state = torch.as_tensor(base_state, dtype=torch.float32, device=self.device)
        if isinstance(depth_img, np.ndarray):
            depth_img = torch.as_tensor(depth_img, dtype=torch.float32, device=self.device)

        depth_img = self._prepare_depth_seq(depth_img)
        base_seq = self._prepare_base_seq(base_state, depth_img.shape[0])

        with torch.no_grad():
            fused_feat = self.actor_encoder(depth_img, base_seq)
            action = self.actor(fused_feat).cpu().numpy().flatten()

        if noise:
            action = action + self.rng.normal(0, self._get_current_noise(progress_ratio), size=self.action_dim)

        action = np.clip(action, -1.0, 1.0)
        return action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()

    def train(self, progress_ratio: float = 0.0):
        del progress_ratio
        self.total_it += 1

        state, depth, action, reward, next_state, next_depth, dones = self.replay_buffer.sample(self.batch_size)

        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = ((action - self.action_bias) / self.action_scale).clamp(-1.0, 1.0)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_feat = self.actor_encoder_target(next_depth, next_state)
            next_action = self.actor_target(next_feat)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            target_feat = self.critic_encoder_target(next_depth, next_state)
            target_q1 = self.critic_1_target(target_feat, next_action)
            target_q2 = self.critic_2_target(target_feat, next_action)
            target_q = reward + (1.0 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_feat = self.critic_encoder(depth, state)
        current_q1 = self.critic_1(current_feat, action)
        current_q2 = self.critic_2(current_feat, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_encoder.parameters())
            + list(self.critic_1.parameters())
            + list(self.critic_2.parameters()),
            self.grad_clip,
        )
        self.critic_optimizer.step()

        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:
            actor_feat = self.actor_encoder(depth, state)
            actor_action = self.actor(actor_feat)

            with torch.no_grad():
                q_feat = self.critic_encoder(depth, state)
            actor_loss = -self.critic_1(q_feat, actor_action).mean()
            actor_loss_value = float(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
                self.grad_clip,
            )
            self.actor_optimizer.step()

            self.soft_update(self.actor_encoder, self.actor_encoder_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic_encoder, self.critic_encoder_target, self.tau)
            self.soft_update(self.critic_1, self.critic_1_target, self.tau)
            self.soft_update(self.critic_2, self.critic_2_target, self.tau)

        result = {
            "critic_loss": float(critic_loss.item()),
        }
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
        return result

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

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
        self.actor_encoder.load_state_dict(checkpoint["actor_encoder"], strict=False)
        self.actor.load_state_dict(checkpoint["actor"], strict=False)
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"], strict=False)
        self.critic_1.load_state_dict(checkpoint["critic_1"], strict=False)
        self.critic_2.load_state_dict(checkpoint["critic_2"], strict=False)

        if "actor_encoder_target" in checkpoint:
            self.actor_encoder_target.load_state_dict(checkpoint["actor_encoder_target"], strict=False)
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"], strict=False)
        if "critic_encoder_target" in checkpoint:
            self.critic_encoder_target.load_state_dict(checkpoint["critic_encoder_target"], strict=False)
        if "critic_1_target" in checkpoint:
            self.critic_1_target.load_state_dict(checkpoint["critic_1_target"], strict=False)
        if "critic_2_target" in checkpoint:
            self.critic_2_target.load_state_dict(checkpoint["critic_2_target"], strict=False)

        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "total_it" in checkpoint:
            self.total_it = checkpoint["total_it"]
