import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from .networks import STMambaEncoder, Actor, Critic
from .buffer import SequenceReplayBuffer


class ST_Mamba_Agent:
    def __init__(self, base_dim, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"ST-Mamba-TD3 Agent using device: {self.device}")
        self.rng = np.random.default_rng(seed)

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

        self.actor_encoder = STMambaEncoder(
            state_dim=self.base_dim,
            action_dim=None,
            args=args
        ).to(self.device)
        self.actor = Actor(
            feature_dim=args.st_mamba_embed_dim + self.base_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.critic_encoder = STMambaEncoder(
            state_dim=self.base_dim,
            action_dim=None,
            args=args
        ).to(self.device)
        self.critic = Critic(
            feature_dim=args.st_mamba_embed_dim + self.base_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        self.actor_encoder_target = copy.deepcopy(self.actor_encoder)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_encoder_target = copy.deepcopy(self.critic_encoder)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = Adam(
            list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
            lr=args.actor_lr
        )
        self.critic_optimizer = Adam(
            list(self.critic_encoder.parameters()) + list(self.critic.parameters()),
            lr=args.critic_lr
        )

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.grad_clip = getattr(args, "grad_clip", 1.0)

        self.exploration_noise = args.exploration_noise
        self.exploration_noise_final = getattr(args, "exploration_noise_final", 0.05)

        self.batch_size = args.batch_size
        self.replay_buffer = SequenceReplayBuffer(args.buffer_size, self.seq_len, seed=seed)
        self.total_it = 0

    def _scale_action(self, action):
        return action * self.action_scale + self.action_bias

    def _get_current_noise(self, progress_ratio: float) -> float:
        current_noise = self.exploration_noise * (1 - progress_ratio) + self.exploration_noise_final * progress_ratio
        return current_noise

    def select_action(self, base_state, depth_img, noise: bool = True, progress_ratio: float = 0.0):
        if isinstance(base_state, np.ndarray):
            base_state = torch.as_tensor(base_state, dtype=torch.float32, device=self.device)
        if isinstance(depth_img, np.ndarray):
            depth_img = torch.as_tensor(depth_img, dtype=torch.float32, device=self.device)

        if depth_img.dim() == 3:
            depth_img = depth_img.unsqueeze(1)
        if depth_img.dim() == 4:
            depth_img = depth_img.unsqueeze(0)

        if depth_img.dtype != torch.float32:
            depth_img = depth_img.float()
        if base_state.dim() == 1:
            current_state = base_state.unsqueeze(0)
        elif base_state.dim() == 2:
            current_state = base_state[-1, :].unsqueeze(0)
        else:
            current_state = base_state[:, -1, :]

        with torch.no_grad():
            visual_feat = self.actor_encoder(depth_img, current_state)
            actor_input = torch.cat([visual_feat, current_state], dim=-1)
            action = self.actor(actor_input).cpu().numpy().flatten()

        if noise:
            current_noise = self._get_current_noise(progress_ratio)
            noise = self.rng.normal(0, current_noise, size=self.action_dim)
            action = action + noise

        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        return scaled_action

    def train(self, replay_buffer=None, batch_size=None):
        self.total_it += 1

        if batch_size is None:
            batch_size = self.batch_size

        if replay_buffer is None:
            replay_buffer = self.replay_buffer

        (state, depth, action, reward,
         next_state, next_depth, done_flag) = replay_buffer.sample(batch_size)

        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        next_depth = torch.as_tensor(next_depth, dtype=torch.float32, device=self.device)

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
        action = (action - self.action_bias) / self.action_scale
        action = action.clamp(-1.0, 1.0)

        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done_flag = torch.as_tensor(done_flag, dtype=torch.float32, device=self.device)

        reward = reward.view(-1, 1)

        done_flag = done_flag.view(-1, 1)
        not_done = 1.0 - done_flag

        with torch.no_grad():
            next_visual = self.actor_encoder_target(next_depth, next_state_curr)
            next_actor_input = torch.cat([next_visual, next_state_curr], dim=-1)
            next_action = self.actor_target(next_actor_input)

            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            target_visual = self.critic_encoder_target(next_depth, next_state_curr)
            target_input = torch.cat([target_visual, next_state_curr], dim=-1)
            target_Q1, target_Q2 = self.critic_target(target_input, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        current_visual = self.critic_encoder(depth, current_state)
        critic_input = torch.cat([current_visual, current_state], dim=-1)
        current_Q1, current_Q2 = self.critic(critic_input, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic_encoder.parameters()) + list(self.critic.parameters()),
            self.grad_clip
        )
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        if self.total_it % self.policy_freq == 0:
            actor_visual = self.actor_encoder(depth, current_state)
            actor_input = torch.cat([actor_visual, current_state], dim=-1)
            actor_action = self.actor(actor_input)

            with torch.no_grad():
                q_visual = self.critic_encoder(depth, current_state)
            q_input = torch.cat([q_visual, current_state], dim=-1)
            q1_pi, _ = self.critic(q_input, actor_action)
            actor_loss = -q1_pi.mean()

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
            self.soft_update(self.critic, self.critic_target, self.tau)

            actor_loss_value = actor_loss.item()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_value}

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(
            {
                "actor_encoder": self.actor_encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_encoder_target": self.actor_encoder_target.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_encoder_target": self.critic_encoder_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
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
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_encoder_target" in checkpoint:
            self.actor_encoder_target.load_state_dict(checkpoint["actor_encoder_target"])
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"])
        if "critic_encoder_target" in checkpoint:
            self.critic_encoder_target.load_state_dict(checkpoint["critic_encoder_target"])
        if "critic_target" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "total_it" in checkpoint:
            self.total_it = checkpoint["total_it"]
