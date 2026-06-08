import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .buffer import ReplayBuffer
from .networks import Actor, Critic, MambaEncoder


class MambaTD3Agent:
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Mamba-TD3 Agent using device: {self.device}")
        self.rng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.args = copy.deepcopy(args)
        self.base_dim = base_dim
        self.depth_shape = depth_shape
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape

        self.seq_len = getattr(self.args, "n_frames", 4)

        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.action_scale = torch.from_numpy((self.max_action - self.min_action) / 2.0).float().to(self.device)
        self.action_bias = torch.from_numpy((self.max_action + self.min_action) / 2.0).float().to(self.device)

        self.actor_encoder = MambaEncoder(self.args).to(self.device)
        self.visual_feature_dim = self.actor_encoder.repr_dim
        self.fused_feature_dim = self.visual_feature_dim + self.base_dim
        self.actor = Actor(
            feature_dim=self.fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=self.args.hidden_dim,
        ).to(self.device)

        self.critic_encoder = MambaEncoder(self.args).to(self.device)
        self.critic_1 = Critic(
            feature_dim=self.fused_feature_dim,
            action_dim=self.action_dim,
            hidden_dim=self.args.hidden_dim,
        ).to(self.device)
        self.critic_2 = Critic(
            feature_dim=self.fused_feature_dim,
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
        self.batch_size = self.args.batch_size
        self.replay_buffer = ReplayBuffer(self.args.buffer_size, self.seq_len, seed=seed)
        self.total_it = 0

    def _get_current_noise(self, progress_ratio: float) -> float:
        return max(float(self.exploration_noise), 1e-8)
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
            next_visual = self.actor_encoder_target(next_depth)
            next_actor_input = torch.cat([next_visual, next_state], dim=-1)
            next_action = self.actor_target(next_actor_input)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            target_visual = self.critic_encoder_target(next_depth)
            target_input = torch.cat([target_visual, next_state], dim=-1)
            target_q1 = self.critic_1_target(target_input, next_action)
            target_q2 = self.critic_2_target(target_input, next_action)
            target_q = reward + (1.0 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_visual = self.critic_encoder(depth)
        critic_input = torch.cat([current_visual, state], dim=-1)
        current_q1 = self.critic_1(critic_input, action)
        current_q2 = self.critic_2(critic_input, action)
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
            actor_visual = self.actor_encoder(depth)
            actor_input = torch.cat([actor_visual, state], dim=-1)
            actor_action = self.actor(actor_input)

            with torch.no_grad():
                q_visual = self.critic_encoder(depth)
            q_input = torch.cat([q_visual, state], dim=-1)
            actor_loss = -self.critic_1(q_input, actor_action).mean()
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
