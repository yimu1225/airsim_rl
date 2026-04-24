import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..state_adapter import StateAdapter
from ..config_loader import get_algo_param
from .networks import Actor, Critic, Encoder
from .buffer import ReplayBuffer


class SACAgent:
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

        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, "base_feature_dim", 32)
        self.depth_shape = depth_shape  # (C, H, W)
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
        
        # Entropy temperature parameters
        self.auto_entropy_tuning = get_algo_param(args, "auto_entropy_tuning", True)
        self.target_entropy = -self.action_dim  # Heuristic: -dim(A)
        
        # Initialize temperature (alpha)
        if self.auto_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=args.actor_lr)
        else:
            self.alpha = get_algo_param(args, "alpha", 0.2)

        C, depth_h, depth_w = depth_shape
        self.depth_seq_len = max(1, int(C))
        visual_channels = 1

        # Encoders
        self.actor_encoder = Encoder(input_height=depth_h, input_width=depth_w, input_channels=visual_channels).to(self.device)
        self.critic_encoder = Encoder(input_height=depth_h, input_width=depth_w, input_channels=visual_channels).to(self.device)

        # Target Encoder for Critic (soft update)
        self.critic_encoder_target = Encoder(input_height=depth_h, input_width=depth_w, input_channels=visual_channels).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        
        # State adapters
        self.actor_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        self.critic_base_adapter_target.load_state_dict(self.critic_base_adapter.state_dict())
        
        # State dimension
        self.state_dim = self.base_feature_dim + self.actor_encoder.repr_dim * self.depth_seq_len
        
        # Actor and Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        
        # Target Critic (for stability)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_params = list(self.actor.parameters()) + list(self.actor_encoder.parameters()) + list(self.actor_base_adapter.parameters())
        self.actor_optimizer = Adam(self.actor_params, lr=args.actor_lr)
        
        self.critic_params = list(self.critic.parameters()) + list(self.critic_encoder.parameters()) + list(self.critic_base_adapter.parameters())
        self.critic_optimizer = Adam(self.critic_params, lr=args.critic_lr)

        self.replay_buffer = ReplayBuffer(args.buffer_size, seed=seed)

        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        
        self.total_it = 0
        
        # Policy update frequency (update actor less frequently than critic)
        self.policy_freq = getattr(args, "policy_freq", 1)

    def _encode(self, depth_batch: torch.Tensor, encoder_net) -> torch.Tensor:
        """Encode depth image."""
        if depth_batch.dim() == 2:
            depth_batch = depth_batch.unsqueeze(0).unsqueeze(0)
        elif depth_batch.dim() == 3:
            depth_batch = depth_batch.unsqueeze(0)
        elif depth_batch.dim() == 5:
            if depth_batch.size(2) != 1:
                raise ValueError(f"Expected single-channel frames, got {tuple(depth_batch.shape)}")
            depth_batch = depth_batch.squeeze(2)

        if depth_batch.dim() != 4:
            raise ValueError(f"Unsupported depth batch shape: {tuple(depth_batch.shape)}")

        batch_size, seq_len, height, width = depth_batch.shape
        frames = depth_batch.reshape(batch_size * seq_len, 1, height, width)
        frame_features = encoder_net(frames).view(batch_size, seq_len, -1)
        return frame_features.reshape(batch_size, seq_len * frame_features.size(-1))

    def _concat_state(self, base: torch.Tensor, depth: torch.Tensor, encoder_net, base_adapter, detach_encoder: bool = False) -> torch.Tensor:
        """Concatenate base features and encoded depth features."""
        base_features = base_adapter(base)
        depth_features = self._encode(depth, encoder_net)
        if detach_encoder:
            depth_features = depth_features.detach()
        return torch.cat([base_features, depth_features], dim=1)

    def select_action(self, base_state, depth, deterministic=False, with_log_prob=False):
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
            state = self._concat_state(base_tensor, depth_tensor, self.actor_encoder, self.actor_base_adapter)
            
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
        base_states, depths, actions, rewards, next_base_states, next_depths, dones = self.replay_buffer.sample(self.batch_size)

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

        # ============ Critic Update ============
        with torch.no_grad():
            # Encode next observations (Critic Target)
            next_encoded_depths = self._encode(next_depths, self.critic_encoder_target)
            next_base_features = self.critic_base_adapter_target(next_base_states)
            next_states = torch.cat([next_base_features, next_encoded_depths], dim=1)
            
            # Sample next actions from current policy
            next_actions, next_log_probs, _, _ = self.actor(next_states, with_log_prob=True)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term
            if self.auto_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha
            target_q = target_q - alpha * next_log_probs
            
            # Bellman backup
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Encode current observations (Critic)
        encoded_depths = self._encode(depths, self.critic_encoder)
        base_features = self.critic_base_adapter(base_states)
        states = torch.cat([base_features, encoded_depths], dim=1)
        
        # Compute current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=self.grad_clip)
        self.critic_optimizer.step()

        # ============ Actor and Alpha Update ============
        actor_loss_value = None
        alpha_loss_value = None
        
        if self.total_it % self.policy_freq == 0:
            # Encode current observations (Actor)
            encoded_depths_actor = self._encode(depths, self.actor_encoder)
            base_features_actor = self.actor_base_adapter(base_states)
            states_actor = torch.cat([base_features_actor, encoded_depths_actor], dim=1)
            
            # Sample actions from policy
            sampled_actions, log_probs, _, _ = self.actor(states_actor, with_log_prob=True)
            
            # Compute Q-values for sampled actions (use Q1)
            q1_new, q2_new = self.critic(states_actor.detach(), sampled_actions)
            q_new = torch.min(q1_new, q2_new)
            
            # Actor loss: maximize Q - alpha * log_prob
            if self.auto_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha
            
            actor_loss = (alpha * log_probs - q_new).mean()

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
                
                # Update alpha value
                self.alpha = self.log_alpha.exp().item()
                alpha_loss_value = float(alpha_loss.item())

        # ============ Soft Update Target Networks ============
        self._soft_update()

        result = {
            "critic_loss": float(critic_loss.item()),
            "alpha": self.alpha if isinstance(self.alpha, float) else self.alpha.item(),
        }
        if actor_loss_value is not None:
            result["actor_loss"] = actor_loss_value
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
            
        for param, target_param in zip(self.critic_base_adapter.parameters(), self.critic_base_adapter_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'actor_base_adapter': self.actor_base_adapter.state_dict(),
            'critic_base_adapter': self.critic_base_adapter.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.actor_base_adapter.load_state_dict(checkpoint['actor_base_adapter'])
        self.critic_base_adapter.load_state_dict(checkpoint['critic_base_adapter'])
        
        # Copy to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        self.critic_base_adapter_target.load_state_dict(self.critic_base_adapter.state_dict())
        
        if self.auto_entropy_tuning and checkpoint.get('log_alpha') is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            self.alpha = self.log_alpha.exp().item()
