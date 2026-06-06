import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import SubNetwork1, SubNetwork2, GlobalActor, Critic
from .buffer import ReplayBuffer


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, mu, theta, sigma, dim, rng=None):
        self.mu = np.ones(dim, dtype=np.float32) * mu
        self.theta = theta
        self.sigma = sigma
        self.dim = dim
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(self.dim)
        self.state += dx
        return self.state


class SDDPGAgent:
    """
    State-Decomposition DDPG for UAV Navigation (SDDPG-NAV).

    论文: "A State-Decomposition DDPG Algorithm for UAV Autonomous Navigation
            in 3-D Complex Environments"

    架构:
      - SubNetwork1: depth(4帧) -> CNN 逐帧编码 -> [32,16] + Tanh -> ao
      - SubNetwork2: sg(4维)     -> [32,16] + Tanh -> ag
      - GlobalActor: [ao, ag, S] -> [400,300] + Tanh -> action
      - Critic:      独立 CNN 编码 depth,  concat[so_repr, sg, su, action] -> [400,300] -> Q
      其中 S = [so_repr(256), sg(4), su(7)] 为完整状态表征
    """

    # Indices within the 11-D base vector
    SG_RAW_INDICES = [0, 1, 2, 10]   # x_dist, y_dist, z_dist, r_yaw
    SU_INDICES = [3, 4, 5, 6, 7, 8, 9]  # altitude, body_x_vel, z_vel, yaw_rate, pitch, roll, yaw

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.base_dim = base_dim
        self.depth_shape = depth_shape  # (4, H, W)  env 返回 4 帧堆叠
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

        # ---- Hyperparameters from args / YAML ----
        self.sub1_hidden = getattr(args, "sub1_hidden", [32, 16])
        self.sub2_hidden = getattr(args, "sub2_hidden", [32, 16])
        self.global_actor_hidden = getattr(args, "global_actor_hidden", [400, 300])
        self.critic_hidden = getattr(args, "critic_hidden", [400, 300])

        self.sub1_out_dim = getattr(args, "sub1_out_dim", 16)
        self.sub2_out_dim = getattr(args, "sub2_out_dim", 16)
        self.encoder_output_dim = getattr(args, "encoder_output_dim", 64)

        # OU noise
        self.ou_theta = getattr(args, "ou_theta", 0.15)
        self.ou_sigma = getattr(args, "ou_sigma", 0.2)
        self.ou_mu = getattr(args, "ou_mu", 0.0)
        self.epsilon = float(getattr(args, "initial_noise_scale", 1.0))
        self.noise_decay = float(getattr(args, "noise_decay", 0.002))
        self.noise_min = float(getattr(args, "noise_min", 0.001))
        self.ou_noise = OrnsteinUhlenbeckNoise(
            mu=self.ou_mu,
            theta=self.ou_theta,
            sigma=self.ou_sigma,
            dim=self.action_dim,
            rng=self.rng,
        )

        self.use_smooth_l1 = bool(getattr(args, "use_smooth_l1", True))

        # ---- Dimensions ----
        self.sg_raw_dim = len(self.SG_RAW_INDICES)   # 4 raw -> converted to [dh, dv, ph, pv]
        self.su_dim = len(self.SU_INDICES)           # 7
        self.sg_dim = 4                              # [dh, dv, ph, pv]

        # ---- Networks ----
        # SubNetwork1 (perception / so): shared for actor & critic
        self.sub1 = SubNetwork1(
            depth_shape=self.depth_shape,
            hidden_dims=self.sub1_hidden,
            out_dim=self.sub1_out_dim,
            encoder_output_dim=self.encoder_output_dim,
        ).to(self.device)
        self.sub1_target = SubNetwork1(
            depth_shape=self.depth_shape,
            hidden_dims=self.sub1_hidden,
            out_dim=self.sub1_out_dim,
            encoder_output_dim=self.encoder_output_dim,
        ).to(self.device)
        self.sub1_target.load_state_dict(self.sub1.state_dict())

        # SubNetwork2 (target-related / sg)
        self.sub2 = SubNetwork2(
            sg_dim=self.sg_dim,
            hidden_dims=self.sub2_hidden,
            out_dim=self.sub2_out_dim,
        ).to(self.device)
        self.sub2_target = SubNetwork2(
            sg_dim=self.sg_dim,
            hidden_dims=self.sub2_hidden,
            out_dim=self.sub2_out_dim,
        ).to(self.device)
        self.sub2_target.load_state_dict(self.sub2.state_dict())

        # 完整状态 S = [so_repr(256), sg(4), su(7)] = 267 维
        self.so_repr_dim = self.sub1.cat_repr_dim          # 256 (4帧×64)
        self.state_repr_dim = self.so_repr_dim + self.sg_dim + self.su_dim  # 267
        self.global_actor_input_dim = self.sub1_out_dim + self.sub2_out_dim + self.state_repr_dim

        # Global Actor
        self.global_actor = GlobalActor(
            input_dim=self.global_actor_input_dim,
            hidden_dims=self.global_actor_hidden,
            action_dim=self.action_dim,
        ).to(self.device)
        self.global_actor_target = GlobalActor(
            input_dim=self.global_actor_input_dim,
            hidden_dims=self.global_actor_hidden,
            action_dim=self.action_dim,
        ).to(self.device)
        self.global_actor_target.load_state_dict(self.global_actor.state_dict())

        # Critic — 拥有独立 CNN，不与 Actor 共享
        self.critic = Critic(
            depth_shape=self.depth_shape,
            sg_dim=self.sg_dim,
            su_dim=self.su_dim,
            action_dim=self.action_dim,
            hidden_dims=self.critic_hidden,
            encoder_output_dim=self.encoder_output_dim,
        ).to(self.device)
        self.critic_target = Critic(
            depth_shape=self.depth_shape,
            sg_dim=self.sg_dim,
            su_dim=self.su_dim,
            action_dim=self.action_dim,
            hidden_dims=self.critic_hidden,
            encoder_output_dim=self.encoder_output_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ---- Optimizers ----
        self.actor_params = (
            list(self.sub1.parameters())
            + list(self.sub2.parameters())
            + list(self.global_actor.parameters())
        )
        self.actor_optimizer = Adam(self.actor_params, lr=getattr(args, "actor_lr", 3e-4))

        self.critic_params = list(self.critic.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=getattr(args, "critic_lr", 3e-4))

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(getattr(args, "buffer_size", 100000), seed=seed)

        self.gamma = getattr(args, "gamma", 0.99)
        self.tau = getattr(args, "tau", 0.005)
        self.batch_size = getattr(args, "batch_size", 256)

        self.total_it = 0

    # ------------------------------------------------------------------ #
    # State decomposition helpers
    # ------------------------------------------------------------------ #
    def _decompose_base(self, base: torch.Tensor):
        """
        Decompose base state vector into su and sg.

        Args:
            base: (B, 11) or (11,)
        Returns:
            su: (B, 7)
            sg: (B, 4)  [dh, dv, phi_h, phi_v]
        """
        if base.dim() == 1:
            base = base.unsqueeze(0)

        su = base[:, self.SU_INDICES]

        x_dist = base[:, 0]
        y_dist = base[:, 1]
        z_dist = base[:, 2]
        r_yaw = base[:, 10]

        dh = torch.sqrt(x_dist ** 2 + y_dist ** 2 + 1e-8)
        dv = z_dist
        phi_h = r_yaw
        phi_v = torch.atan2(z_dist, dh + 1e-8)

        sg = torch.stack([dh, dv, phi_h, phi_v], dim=-1)
        return su, sg

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_action(self, base_state, depth, noise: bool = True, progress_ratio: float = 0.0):
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(0)

        with torch.no_grad():
            su, sg = self._decompose_base(base_tensor)
            ao, so_repr = self.sub1(depth_tensor)          # 一次性获取 ao 和 so_repr
            ag = self.sub2(sg)
            # sg/su 原始维度直接送入，不经过 adapter 投影
            state_repr = torch.cat([so_repr, sg, su], dim=-1)
            action = self.global_actor(ao, ag, state_repr).cpu().numpy().flatten()

        if noise:
            ou_sample = self.ou_noise.sample()
            action = action + self.epsilon * ou_sample
            action = np.clip(action, -1.0, 1.0)

        real_action = self.action_scale.cpu().numpy() * action + self.action_bias.cpu().numpy()
        return real_action

    def on_episode_end(self):
        """Decay exploration noise epsilon per episode (paper Algorithm 1, line 23)."""
        self.epsilon = max(self.epsilon - self.noise_decay, self.noise_min)
        self.ou_noise.reset()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, progress_ratio=0.0):
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return {}

        base_states, depths, actions, rewards, next_base_states, next_depths, dones = self.replay_buffer.sample(self.batch_size)

        base_states = torch.as_tensor(base_states, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions_norm = (real_actions - self.action_bias) / self.action_scale

        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_base_states, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depths, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        # Decompose base states
        su, sg = self._decompose_base(base_states)
        next_su, next_sg = self._decompose_base(next_base_states)

        # ----- Critic update -----
        # Critic 内部有独立 CNN 编码 depth，sg/su 直接传入
        current_Q, _ = self.critic(depths, sg, su, actions_norm)

        with torch.no_grad():
            next_ao, next_so_repr = self.sub1_target(next_depths)
            next_ag = self.sub2_target(next_sg)
            next_state_repr = torch.cat([next_so_repr, next_sg, next_su], dim=-1)
            next_action = self.global_actor_target(next_ao, next_ag, next_state_repr).clamp(-1.0, 1.0)

            target_Q, _ = self.critic_target(next_depths, next_sg, next_su, next_action)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        if self.use_smooth_l1:
            critic_loss = F.smooth_l1_loss(current_Q, target_Q)
        else:
            critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()

        # ----- Actor update -----
        for p in self.critic_params:
            p.requires_grad = False

        # Actor 前向: 子网络 → GlobalActor → 动作
        ao, so_repr = self.sub1(depths)
        ag = self.sub2(sg)
        state_repr = torch.cat([so_repr, sg, su], dim=-1)
        pred_action = self.global_actor(ao, ag, state_repr)

        # ∇_aQ · ∇_θμ: DDPG 策略梯度，冻结 critic 的 CNN 和 sg/su 梯度
        q1, _ = self.critic(depths.detach(), sg.detach(), su.detach(), pred_action)
        actor_loss = -q1.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=self.grad_clip)
        self.actor_optimizer.step()

        for p in self.critic_params:
            p.requires_grad = True

        # ----- Soft update targets -----
        self._soft_update(self.sub1, self.sub1_target)
        self._soft_update(self.sub2, self.sub2_target)
        self._soft_update(self.global_actor, self.global_actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "epsilon": float(self.epsilon),
        }

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #
    def save(self, filename: str):
        torch.save(
            {
                "sub1": self.sub1.state_dict(),
                "sub2": self.sub2.state_dict(),
                "global_actor": self.global_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "sub1_target": self.sub1_target.state_dict(),
                "sub2_target": self.sub2_target.state_dict(),
                "global_actor_target": self.global_actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_it": self.total_it,
                "epsilon": self.epsilon,
            },
            filename,
        )

    def load(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device)
        self.sub1.load_state_dict(checkpoint["sub1"])
        self.sub2.load_state_dict(checkpoint["sub2"])
        self.global_actor.load_state_dict(checkpoint["global_actor"])
        self.critic.load_state_dict(checkpoint["critic"])

        self.sub1_target.load_state_dict(checkpoint.get("sub1_target", self.sub1.state_dict()))
        self.sub2_target.load_state_dict(checkpoint.get("sub2_target", self.sub2.state_dict()))
        self.global_actor_target.load_state_dict(checkpoint.get("global_actor_target", self.global_actor.state_dict()))
        self.critic_target.load_state_dict(checkpoint.get("critic_target", self.critic.state_dict()))

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon)


def make_agent(env, initial_obs, args, device=None):
    base_state = initial_obs["base"]
    depth = initial_obs["depth"]
    agent = SDDPGAgent(
        base_dim=base_state.shape[0],
        depth_shape=depth.shape,
        action_space=env.action_space,
        args=args,
        device=device,
    )
    return agent
