import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..config_loader import get_algo_param
from .networks import Actor, Critic, Encoder
from .buffer import ReplayBuffer


class MambaCSJA_SACAgent:
    """Soft Actor-Critic (SAC) Agent with automatic entropy tuning.
    
    Features:
    - Stochastic policy with reparameterization trick
    - Twin Q-networks for reduced overestimation
    - Automatic temperature tuning for optimal exploration-exploitation trade-off
    - Separate encoders for actor and critic
    """
    
    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # RNG for this agent
        self.rng = np.random.default_rng(seed)
        
        # Set PyTorch random seed
        if seed is not None:
            torch.manual_seed(seed)

        self.args = args
        self.base_dim = base_dim
        self.depth_shape = depth_shape  # (C, H, W)
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape
        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.max_action_tensor = torch.from_numpy(self.max_action).float().to(self.device)
        self.min_action_tensor = torch.from_numpy(self.min_action).float().to(self.device)

        # Action scaling
        scale = (self.max_action - self.min_action) / 2.0
        bias = (self.max_action + self.min_action) / 2.0
        self.action_scale = torch.from_numpy(scale).float().to(self.device)
        self.action_bias = torch.from_numpy(bias).float().to(self.device)

        self.grad_clip = getattr(args, "grad_clip", 1.0)

        # Entropy temperature / entropy coefficient (ent_coef) handling
        # Support SB3-style: ent_coef can be a float or a string like 'auto' or 'auto_0.1'
        self.ent_coef = get_algo_param(args, "ent_coef", 0.2)
        # target entropy: fallback to -dim(A) heuristic when not provided or set to 'auto' / null
        self.target_entropy = get_algo_param(args, "target_entropy", None)
        if self.target_entropy is None or self.target_entropy == "auto":
            # heuristic default: -|A|
            self.target_entropy = -float(self.action_dim)
        else:
            self.target_entropy = float(self.target_entropy)

        # By default, no optimizer for ent_coef
        self.log_alpha = None
        self.alpha_optimizer = None
        # If ent_coef is a string starting with 'auto', learn log_alpha (initialize to provided value or 1.0)
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                try:
                    init_value = float(self.ent_coef.split("_")[1])
                except Exception:
                    init_value = 1.0
            # learn log_alpha (log(ent_coef))
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=float(get_algo_param(args, "alpha_lr", args.actor_lr)))
            # alpha property returns current scalar value via exp()
            self.alpha = float(init_value)
            self.auto_entropy_tuning = True
        else:
            # fixed ent_coef: prefer explicit ent_coef value, otherwise fall back to legacy 'alpha' config
            if self.ent_coef is None:
                self.alpha = get_algo_param(args, "alpha", 0.2)
            else:
                self.alpha = float(self.ent_coef)
            self.auto_entropy_tuning = False

        C, depth_h, depth_w = depth_shape
        self.depth_seq_len = max(1, int(C))
        visual_channels = self.depth_seq_len

        # Encoders
        self.actor_encoder = self._make_encoder(depth_h, depth_w, visual_channels).to(self.device)
        self.critic_encoder = self._make_encoder(depth_h, depth_w, visual_channels).to(self.device)

        # Target Encoder for Critic (soft update)
        self.critic_encoder_target = self._make_encoder(depth_h, depth_w, visual_channels).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        
        # State adapters
        
        # State dimension
        self.state_dim = self.base_dim + self._encoded_visual_dim()
        
        # Actor and Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        
        # Target Critic (for stability)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_params = list(self.actor.parameters()) + list(self.actor_encoder.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        self.critic_params = list(self.critic.parameters()) + list(self.critic_encoder.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = self._make_replay_buffer(args, seed)

        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        
        self.total_it = 0
        
        # SB3 SAC updates actor every gradient step by default.
        self.policy_freq = getattr(args, "policy_freq", 1)
        self.target_update_interval = get_algo_param(args, "target_update_interval", 1)

    def _make_encoder(self, depth_h: int, depth_w: int, visual_channels: int):
        return Encoder(input_height=depth_h, input_width=depth_w, input_channels=visual_channels)

    def _uses_sequence_encoder(self) -> bool:
        return False

    def _encoded_visual_dim(self) -> int:
        # MambaAttentionCNN encodes all stacked frames into a single vector of size repr_dim.
        return self.actor_encoder.repr_dim

    def _make_replay_buffer(self, args, seed=None):
        return ReplayBuffer(args.buffer_size, seed=seed)

    def _sample_replay(self):
        sample = self.replay_buffer.sample(self.batch_size)
        return sample, None, None, {}

    def _update_replay_priorities(self, refs, td_errors):
        return None

    def _encode(self, depth_batch: torch.Tensor, encoder_net) -> torch.Tensor:
        """Encode depth image. MambaAttentionCNN natively handles stacked frame inputs."""
        return encoder_net(depth_batch)

    def _concat_state(self, base: torch.Tensor, depth: torch.Tensor, encoder_net) -> torch.Tensor:
        """Concatenate raw base state and encoded depth features."""
        depth_features = self._encode(depth, encoder_net)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base_states, depth_features], dim=1)

    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False, progress_ratio=0.0):
        """Select action using the current policy.
        
        Args:
            base_state: base state vector
            depth: depth image
            deterministic: if True, use mean action (no sampling)
            with_log_prob: if True, return log probability of the action
        """
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            state = self._concat_state(base_tensor, depth_tensor, self.actor_encoder)
            
            if with_log_prob and not deterministic:
                action, log_prob, _, _ = self.actor(state, with_log_prob=True)
                # Scale to real action space
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten(), log_prob.cpu().numpy()
            else:
                action = self.actor.get_action(state, deterministic=deterministic)
                # Scale to real action space
                real_action = self.action_scale * action + self.action_bias
                return real_action.cpu().numpy().flatten()

    def train(self, progress_ratio=0.0):
        """Train the agent for one step."""
        self.total_it += 1

        if self.replay_buffer.size() < self.batch_size:
            return {}

        # Sample from replay buffer
        sample, replay_refs, replay_weights, replay_info = self._sample_replay()
        if sample is None:
            return {}
        base_states, depths, actions, rewards, next_base_states, next_depths, dones = sample

        # Convert to tensors
        base_states = torch.as_tensor(base_states, dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        # Actions from buffer are real actions, normalize them for training
        real_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = (real_actions - self.action_bias) / self.action_scale
        
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_base_states = torch.as_tensor(next_base_states, dtype=torch.float32, device=self.device)
        next_depths = torch.as_tensor(next_depths, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        weights = None
        if replay_weights is not None:
            weights = torch.as_tensor(replay_weights, dtype=torch.float32, device=self.device).view(-1, 1)

        # ============ Critic Update ============
        with torch.no_grad():
            # SB3-style: next action comes from the current actor,
            # target Q comes from the target critic.
            next_actor_states = self._concat_state(
                next_base_states,
                next_depths,
                self.actor_encoder,
            )
            next_target_states = self._concat_state(
                next_base_states,
                next_depths,
                self.critic_encoder_target,
            )
            
            # Sample next actions from current policy
            next_actions, next_log_probs, _, _ = self.actor(next_actor_states, with_log_prob=True)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_target_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term
            if self.auto_entropy_tuning:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = torch.as_tensor(self.alpha, dtype=torch.float32, device=self.device)
            target_q = target_q - alpha * next_log_probs
            
            # Bellman backup
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Encode current observations (Critic)
        encoded_depths = self._encode(depths, self.critic_encoder)
        states = torch.cat([base_states, encoded_depths], dim=1)
        
        # Compute current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss: SB3 uses 0.5 * sum(MSE) over the two critics.
        critic_loss_elements = 0.5 * (
            F.mse_loss(current_q1, target_q, reduction="none")
            + F.mse_loss(current_q2, target_q, reduction="none")
        )
        if weights is not None:
            critic_loss = (critic_loss_elements * weights).mean()
        else:
            critic_loss = critic_loss_elements.mean()
        td_errors = 0.5 * ((current_q1 - target_q).abs() + (current_q2 - target_q).abs())
        target_q_mean_value = float(target_q.mean().detach().item())
        current_q_mean_value = float(torch.min(current_q1, current_q2).mean().detach().item())

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()
        if replay_refs is not None:
            self._update_replay_priorities(replay_refs, td_errors.detach().cpu().numpy().reshape(-1))

        # ============ Actor and Alpha Update ============
        actor_loss_value = None
        alpha_loss_value = None
        mean_log_prob_value = None
        q_pi_mean_value = None
        
        if self.total_it % self.policy_freq == 0:
            # Encode current observations (Actor)
            encoded_depths_actor = self._encode(depths, self.actor_encoder)
            states_actor = torch.cat([base_states, encoded_depths_actor], dim=1)
            
            # Sample actions from policy
            sampled_actions, log_probs, _, _ = self.actor(states_actor, with_log_prob=True)
            
            with torch.no_grad():
                critic_states_for_pi = self._concat_state(
                    base_states,
                    depths,
                    self.critic_encoder,
                )

            # Compute Q-values for sampled actions.
            q1_new, q2_new = self.critic(critic_states_for_pi, sampled_actions)
            q_new = torch.min(q1_new, q2_new)
            
            # Actor loss: maximize Q - alpha * log_prob
            if self.auto_entropy_tuning:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = torch.as_tensor(self.alpha, dtype=torch.float32, device=self.device)
            
            actor_loss = (alpha * log_probs - q_new).mean()
            mean_log_prob_value = float(log_probs.mean().detach().item())
            q_pi_mean_value = float(q_new.mean().detach().item())

            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=self.grad_clip)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.item())

            # ============ Alpha Update ============
            if self.auto_entropy_tuning:
                # We want to minimize: -alpha * (log_prob + target_entropy)
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = float(self.log_alpha.exp().detach().item())
                
                # Update alpha value
                alpha_loss_value = float(alpha_loss.item())

        # ============ Soft Update Target Networks ============
        if self.total_it % self.target_update_interval == 0:
            self._soft_update()

        result = {
            "critic_loss": float(critic_loss.item()),
            "alpha": self.alpha if isinstance(self.alpha, float) else self.alpha.item(),
            "target_q_mean": target_q_mean_value,
            "current_q_mean": current_q_mean_value,
        }
        if replay_info:
            result.update(replay_info)
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
        """Soft update target networks."""
        # Update Critic Target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        
        # Copy to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        
        if self.auto_entropy_tuning and checkpoint.get('log_alpha') is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
