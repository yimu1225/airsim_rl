import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..config_loader import get_algo_param
from ..state_adapter import StateAdapter
from .buffer import RolloutBuffer
from .networks import Actor, Critic, STVimEncoder


class STVimPPOAgent:
    """SB3-style PPO adapted to base-state + ST-Vim/Mamba depth sequences."""

    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.args.depth_shape = depth_shape
        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
        self.depth_shape = depth_shape
        self.action_dim = action_space.shape[0]

        self.max_action = np.asarray(action_space.high, dtype=np.float32)
        self.min_action = np.asarray(action_space.low, dtype=np.float32)
        self.action_scale = torch.as_tensor((self.max_action - self.min_action) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.as_tensor((self.max_action + self.min_action) / 2.0, dtype=torch.float32, device=self.device)

        self.encoder = STVimEncoder(args).to(self.device)
        self.base_encoder = StateAdapter(base_dim, self.base_feature_dim).to(self.device)
        self.state_dim = self.base_feature_dim + self.encoder.repr_dim

        hidden_dim = getattr(args, "hidden_dim", 256)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim).to(self.device)

        lr = get_algo_param(args, "lr", getattr(args, "actor_lr", 3e-4))
        self.encoder_optimizer = Adam(self.encoder.parameters(), lr=lr)
        self.base_encoder_optimizer = Adam(self.base_encoder.parameters(), lr=lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        self.gamma = getattr(args, "gamma", 0.99)
        self.gae_lambda = get_algo_param(args, "gae_lambda", 0.95)
        buffer_size = get_algo_param(args, "rollout_buffer_size", 2048)
        buffer_depth_shape = (args.n_frames, depth_shape[-2], depth_shape[-1])
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size,
            base_dim=base_dim,
            depth_shape=buffer_depth_shape,
            action_dim=self.action_dim,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        self.ppo_epochs = get_algo_param(args, "ppo_epochs", 10)
        self.batch_size = get_algo_param(args, "ppo_batch_size", 64)
        self.clip_range = get_algo_param(args, "clip_range", 0.2)
        self.clip_range_vf = get_algo_param(args, "clip_range_vf", None)
        self.normalize_advantage = bool(get_algo_param(args, "normalize_advantage", True))
        self.vf_coef = get_algo_param(args, "vf_coef", 0.5)
        self.ent_coef = get_algo_param(args, "ent_coef", 0.0)
        self.max_grad_norm = get_algo_param(args, "max_grad_norm", 0.5)
        self.target_kl = get_algo_param(args, "target_kl", None)

        self.total_it = 0
        self.num_updates = 0
        self._last_policy_action = None

    def _format_depth_sequence(self, depth_batch: torch.Tensor) -> torch.Tensor:
        if depth_batch.dim() == 2:
            depth_batch = depth_batch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif depth_batch.dim() == 3:
            depth_batch = depth_batch.unsqueeze(0).unsqueeze(2)
        elif depth_batch.dim() == 4:
            if depth_batch.size(0) == self.args.n_frames and depth_batch.size(1) == 1:
                depth_batch = depth_batch.unsqueeze(0)
            else:
                depth_batch = depth_batch.unsqueeze(2)
        elif depth_batch.dim() != 5:
            raise ValueError(f"Unsupported depth sequence shape: {tuple(depth_batch.shape)}")

        if depth_batch.size(2) != 1:
            raise ValueError(f"Expected single-channel sequence frames, got {tuple(depth_batch.shape)}")
        return depth_batch

    def _encode_depth(self, depth):
        return self.encoder(self._format_depth_sequence(depth))

    def _concat_state(self, base, depth):
        base_features = self.base_encoder(base)
        depth_features = self._encode_depth(depth)
        return torch.cat([base_features, depth_features], dim=1)

    def get_state_representation(self, base, depth):
        return self._concat_state(base, depth)

    def select_action(self, base_state, depth, deterministic=False, progress_ratio=0.0):
        base = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            state = self._concat_state(base, depth_tensor)
            action, log_prob = self.actor(state, deterministic=deterministic)
            value = self.critic(state)

        raw_action = action.cpu().numpy().flatten().astype(np.float32)
        self._last_policy_action = raw_action.copy()
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        real_action = self.action_scale.cpu().numpy() * clipped_action + self.action_bias.cpu().numpy()
        return real_action, float(value.cpu().numpy().flatten()[0]), float(log_prob.cpu().numpy().flatten()[0])

    def store_transition(self, base_state, depth, action, reward, value, log_prob, done):
        cached_action = self._last_policy_action
        if cached_action is not None and np.asarray(cached_action).shape == np.asarray(action).shape:
            buffer_action = cached_action.astype(np.float32)
            self._last_policy_action = None
        else:
            buffer_action = (np.asarray(action, dtype=np.float32) - self.action_bias.cpu().numpy()) / self.action_scale.cpu().numpy()
            buffer_action = np.clip(buffer_action, -1.0, 1.0)
        self.rollout_buffer.add(base_state, depth, buffer_action, reward, value, log_prob, done)

    def finish_trajectory(self, last_base_state, last_depth, last_done):
        with torch.no_grad():
            base = torch.as_tensor(last_base_state, dtype=torch.float32, device=self.device).view(1, -1)
            depth = torch.as_tensor(last_depth, dtype=torch.float32, device=self.device)
            state = self._concat_state(base, depth)
            last_value = float(self.critic(state).cpu().numpy().flatten()[0])
        return self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    def train(self, progress_ratio=0.0):
        if self.rollout_buffer.size() == 0:
            return None
        self.total_it += 1
        return self._update_policy(self.rollout_buffer.get_trajectory())

    def update_policy(self, returns=None, advantages=None, epoch_pbar=None):
        return self._update_policy(self.rollout_buffer.get_trajectory(), epoch_pbar=epoch_pbar)

    def _update_policy(self, data, epoch_pbar=None):
        base_states = data["base_states"]
        depth_states = data["depth_states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        returns = data["returns"]
        advantages = data["advantages"]
        old_values = data["values"]

        n_samples = base_states.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0
        approx_kl_divs = []

        continue_training = True
        for epoch in range(self.ppo_epochs):
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb_indices = indices[start:end]

                states = self._concat_state(base_states[mb_indices], depth_states[mb_indices])
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]

                new_log_probs, entropy = self.actor.get_log_prob(states, mb_actions)
                values = self.critic(states).squeeze(-1)
                log_ratio = new_log_probs.squeeze(-1) - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                if self.normalize_advantage and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = mb_old_values + (values - mb_old_values).clamp(
                        -float(self.clip_range_vf),
                        float(self.clip_range_vf),
                    )
                value_loss = F.mse_loss(values_pred, mb_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    approx_kl_divs.append(float(approx_kl.item()))

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    if epoch_pbar is not None:
                        epoch_pbar.set_postfix_str(f"Early stop (KL: {approx_kl:.4f})")
                    break

                self.encoder_optimizer.zero_grad()
                self.base_encoder_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.base_encoder.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.encoder_optimizer.step()
                self.base_encoder_optimizer.step()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.mean().item())
                total_loss += float(loss.item())
                n_updates += 1

            if epoch_pbar is not None:
                epoch_pbar.update(1)
            if not continue_training:
                break

        self.rollout_buffer.after_update()
        self.num_updates += 1
        return {
            "policy_loss": total_policy_loss / n_updates if n_updates else 0.0,
            "value_loss": total_value_loss / n_updates if n_updates else 0.0,
            "total_loss": total_loss / n_updates if n_updates else 0.0,
            "entropy": total_entropy / n_updates if n_updates else 0.0,
            "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0,
        }

    def save(self, filename: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "base_encoder": self.base_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "base_encoder_optimizer": self.base_encoder_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "num_updates": self.num_updates,
        }, filename)

    def load(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.base_encoder.load_state_dict(checkpoint["base_encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "encoder_optimizer" in checkpoint:
            self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        if "base_encoder_optimizer" in checkpoint:
            self.base_encoder_optimizer.load_state_dict(checkpoint["base_encoder_optimizer"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_it = checkpoint.get("total_it", 0)
        self.num_updates = checkpoint.get("num_updates", 0)


def make_agent(env, initial_obs, args, device=None) -> STVimPPOAgent:
    base_state = initial_obs["base"]
    depth = initial_obs["depth"]
    args.depth_shape = (1, depth.shape[-2], depth.shape[-1])
    return STVimPPOAgent(
        base_dim=base_state.shape[0],
        depth_shape=args.depth_shape,
        action_space=env.action_space,
        args=args,
        device=device,
    )


PPOAgent = STVimPPOAgent
