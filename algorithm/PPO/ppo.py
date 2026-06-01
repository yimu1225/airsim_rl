import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from ..state_adapter import StateAdapter
from ..config_loader import get_algo_param
from .networks import Actor, Critic, Encoder
from .buffer import RolloutBuffer


class PPOAgent:
    """
    PPO Agent for continuous action spaces.
    Compatible with the same interface as TD3Agent.
    """
    
    def __init__(self, base_dim: int, depth_shape, action_space, args, device=None, seed=None):
        """
        Initialize PPO Agent.
        
        Args:
            base_dim: dimension of base state
            depth_shape: shape of depth image (C, H, W)
            action_space: gym action space
            args: configuration arguments
            device: torch device
            seed: random seed
        """
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        
        # 设置 PyTorch 随机种子以确保网络初始化确定性
        if seed is not None:
            torch.manual_seed(seed)
        
        self.base_dim = base_dim
        self.base_feature_dim = getattr(args, 'base_feature_dim', 32)
        self.depth_shape = depth_shape  # (C, H, W)
        self.args = args
        if not hasattr(self.args, "depth_shape"):
            self.args.depth_shape = depth_shape
        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.max_action_tensor = torch.from_numpy(self.max_action).float().to(self.device)
        self.min_action_tensor = torch.from_numpy(self.min_action).float().to(self.device)
        
        # Action normalization parameters
        scale = (self.max_action - self.min_action) / 2.0
        bias = (self.max_action + self.min_action) / 2.0
        self.action_scale = torch.from_numpy(scale).float().to(self.device)
        self.action_bias = torch.from_numpy(bias).float().to(self.device)
        
        # Encoder (shared feature extractor): frame-wise shared CNN.
        C, depth_h, depth_w = depth_shape
        self.depth_seq_len = max(1, int(C))
        visual_channels = 1

        self.encoder = self._make_encoder(depth_h, depth_w, visual_channels).to(self.device)

        self.base_encoder = StateAdapter(self.base_dim, self.base_feature_dim).to(self.device)
        
        # State dimension = projected base feature + concatenated per-frame visual features
        self.state_dim = self.base_feature_dim + self._encoded_visual_dim()
        
        # Actor and Critic
        hidden_dim = getattr(args, 'hidden_dim', 256)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim).to(self.device)
        
        # Optimizers - PPO uses a single learning rate (reuse actor_lr from config)
        lr = get_algo_param(args, 'lr', getattr(args, 'actor_lr', 3e-4))
        self.encoder_optimizer = Adam(self.encoder.parameters(), lr=lr)
        self.base_encoder_optimizer = Adam(self.base_encoder.parameters(), lr=lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        
        # Rollout buffer
        buffer_size = get_algo_param(args, 'rollout_buffer_size', 2048)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.gae_lambda = get_algo_param(args, 'gae_lambda', 0.95)
        
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size,
            base_dim=base_dim,
            depth_shape=depth_shape,
            action_dim=self.action_dim,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # PPO hyperparameters
        self.ppo_epochs = get_algo_param(args, 'ppo_epochs', 10)
        self.batch_size = get_algo_param(args, 'ppo_batch_size', 64)
        self.clip_range = get_algo_param(args, 'clip_range', 0.2)
        self.clip_range_vf = get_algo_param(args, 'clip_range_vf', None)
        self.normalize_advantage = bool(get_algo_param(args, 'normalize_advantage', True))
        self.vf_coef = get_algo_param(args, 'vf_coef', 0.5)
        self.ent_coef = get_algo_param(args, 'ent_coef', 0.0)
        self.max_grad_norm = get_algo_param(args, 'max_grad_norm', 0.5)
        self.target_kl = get_algo_param(args, 'target_kl', None)  # Optional early stopping
        
        self.total_it = 0
        self.num_updates = 0

    def _make_encoder(self, depth_h: int, depth_w: int, visual_channels: int):
        return Encoder(input_height=depth_h, input_width=depth_w, input_channels=visual_channels)

    def _uses_sequence_encoder(self) -> bool:
        return False

    def _encoded_visual_dim(self) -> int:
        if self._uses_sequence_encoder():
            return self.encoder.repr_dim
        return self.encoder.repr_dim * self.depth_seq_len
    
    def _encode(self, depth_batch: torch.Tensor) -> torch.Tensor:
        """Encode depth image."""
        if self._uses_sequence_encoder():
            if depth_batch.dim() == 2:
                depth_batch = depth_batch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif depth_batch.dim() == 3:
                depth_batch = depth_batch.unsqueeze(0).unsqueeze(2)
            elif depth_batch.dim() == 4:
                if depth_batch.size(1) == 1:
                    depth_batch = depth_batch.unsqueeze(0)
                else:
                    depth_batch = depth_batch.unsqueeze(2)
            elif depth_batch.dim() == 5:
                pass
            else:
                raise ValueError(f"Unsupported sequence depth batch shape: {tuple(depth_batch.shape)}")

            if depth_batch.size(2) != 1:
                raise ValueError(f"Expected single-channel sequence frames, got {tuple(depth_batch.shape)}")
            return self.encoder(depth_batch)

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
        frame_features = self.encoder(frames).view(batch_size, seq_len, -1)
        return frame_features.reshape(batch_size, seq_len * frame_features.size(-1))
    
    def _concat_state(self, base: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Concatenate projected base state and encoded depth features."""
        base_features = self.base_encoder(base)
        depth_features = self._encode(depth)
        return torch.cat([base_features, depth_features], dim=1)
    
    def get_state_representation(self, base: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Public method to get state representation for external use."""
        return self._concat_state(base, depth)
    
    def select_action(self, base_state, depth, deterministic=False, progress_ratio=0.0, critic_priv=None):
        """
        Select action using the current policy.
        
        Args:
            base_state: base state vector
            depth: depth image
            deterministic: if True, use mean action; otherwise sample
            progress_ratio: training progress (0.0 to 1.0), unused for PPO
            
        Returns:
            action: action in original action space scale
        """
        base_tensor = torch.as_tensor(base_state, dtype=torch.float32, device=self.device).view(1, -1)
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            state = self._concat_state(base_tensor, depth_tensor)
            action, log_prob = self.actor(state, deterministic=deterministic)
            
            # Get value estimate
            value = self.critic(state)
            
            # Convert to numpy
            action_np = action.cpu().numpy().flatten()
            self._last_policy_action = action_np.astype(np.float32, copy=True)
            clipped_action_np = np.clip(action_np, -1.0, 1.0)
            value_np = value.cpu().numpy().flatten()[0]
            log_prob_np = log_prob.cpu().numpy().flatten()[0] if log_prob is not None else 0.0
        
        # Scale action from [-1, 1] to original action space
        real_action = self.action_scale.cpu().numpy() * clipped_action_np + self.action_bias.cpu().numpy()
        
        return real_action, value_np, log_prob_np
    
    def store_transition(self, base_state, depth, action, reward, value, log_prob, done, critic_priv=None):
        """
        Store a transition in the rollout buffer.
        
        Args:
            base_state: base state
            depth: depth image
            action: action taken (in original scale)
            reward: reward received
            value: value estimate
            log_prob: log probability of action
            done: whether episode terminated
        """
        # Normalize action to [-1, 1] for storage
        cached_action = getattr(self, "_last_policy_action", None)
        if cached_action is not None and np.asarray(cached_action).shape == np.asarray(action).shape:
            # SB3 stores the raw Gaussian action in the rollout buffer and only
            # clips/scales a copy for the environment step.
            normalized_action = np.asarray(cached_action, dtype=np.float32)
            self._last_policy_action = None
        else:
            normalized_action = (action - self.action_bias.cpu().numpy()) / self.action_scale.cpu().numpy()
            normalized_action = np.clip(normalized_action, -1.0, 1.0)
        
        self.rollout_buffer.add(base_state, depth, normalized_action, reward, value, log_prob, done)
    
    def finish_trajectory(self, last_base_state, last_depth, last_done, critic_priv=None):
        """
        Finish the current trajectory and compute advantages.
        
        Args:
            last_base_state: last base state
            last_depth: last depth image
            last_done: whether last state is terminal
        """
        # Get value of last state
        with torch.no_grad():
            base_tensor = torch.as_tensor(last_base_state, dtype=torch.float32, device=self.device).view(1, -1)
            depth_tensor = torch.as_tensor(last_depth, dtype=torch.float32, device=self.device)
            state = self._concat_state(base_tensor, depth_tensor)
            last_value = self.critic(state).cpu().numpy().flatten()[0]
        
        return self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)
    
    def train(self, progress_ratio=0.0):
        """
        Train the policy using collected rollouts.
        
        Args:
            progress_ratio: training progress (unused for now)
            
        Returns:
            dict with training statistics
        """
        if self.rollout_buffer.size() == 0:
            return None
        
        self.total_it += 1
        
        return self.update_policy()
    
    def update_policy(self, returns=None, advantages=None, epoch_pbar=None):
        """
        Update policy with computed returns and advantages.
        This should be called after finish_trajectory.
        
        Args:
            returns: computed returns (optional, will use stored if None)
            advantages: computed advantages (optional, will use stored if None)
            epoch_pbar: optional tqdm progress bar for epochs (created in training script)
        """
        if (returns is None or advantages is None) and not self.rollout_buffer.returns_ready:
            raise RuntimeError(
                "PPO update requires computed returns/advantages. "
                "Call finish_trajectory() or rollout_buffer.compute_returns_and_advantages() first."
            )

        data = self.rollout_buffer.get_trajectory()
        if returns is not None:
            data['returns'] = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if advantages is not None:
            data['advantages'] = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        
        return self._update_policy(data, epoch_pbar=epoch_pbar)
    
    def _update_policy(self, data, epoch_pbar=None):
        """
        Internal policy update method.
        
        Args:
            data: trajectory data
            epoch_pbar: optional tqdm progress bar for epochs (created in training script)
        """
        base_states = data['base_states']
        depth_states = data['depth_states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        returns = data['returns']
        advantages = data['advantages']
        
        n_samples = base_states.shape[0]
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        n_updates = 0
        approx_kl_divs = []
        
        continue_training = True
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Generate random indices
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                mb_indices = indices[start:end]
                
                mb_base = base_states[mb_indices]
                mb_depth = depth_states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = data['values'][mb_indices]
                
                # Encode states
                states = self._encode_and_concat(mb_base, mb_depth)
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.get_log_prob(states, mb_actions)
                values = self.critic(states).squeeze(-1)
                
                # PPO loss
                log_ratio = new_log_probs.squeeze(-1) - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                if self.normalize_advantage and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss, optionally clipped like SB3 PPO.
                if self.clip_range_vf is None:
                    value_loss = F.mse_loss(values, mb_returns)
                else:
                    values_pred = mb_old_values + (values - mb_old_values).clamp(
                        -float(self.clip_range_vf),
                        float(self.clip_range_vf),
                    )
                    value_loss_unclipped = F.mse_loss(values, mb_returns, reduction="none")
                    value_loss_clipped = F.mse_loss(values_pred, mb_returns, reduction="none")
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Total loss
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
                
                # Optimize
                self.encoder_optimizer.zero_grad()
                self.base_encoder_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.base_encoder.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.encoder_optimizer.step()
                self.base_encoder_optimizer.step()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                n_updates += 1
            
            # Update progress bar if provided
            if epoch_pbar is not None:
                epoch_pbar.update(1)
            if not continue_training:
                break
        
        # Clear buffer after update
        self.rollout_buffer.after_update()
        self.num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'total_loss': total_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0,
            'approx_kl': np.mean(approx_kl_divs) if approx_kl_divs else 0,
        }
    
    def _encode_and_concat(self, base_states, depth_states):
        """Helper to encode and concatenate."""
        base_features = self.base_encoder(base_states)
        depth_features = self._encode(depth_states)
        return torch.cat([base_features, depth_features], dim=1)
    
    def save(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'base_encoder': self.base_encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'base_encoder_optimizer': self.base_encoder_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'num_updates': self.num_updates,
        }, filename)
    
    def load(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        if 'base_encoder' in checkpoint:
            self.base_encoder.load_state_dict(checkpoint['base_encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        if 'base_encoder_optimizer' in checkpoint:
            self.base_encoder_optimizer.load_state_dict(checkpoint['base_encoder_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint.get('total_it', 0)
        self.num_updates = checkpoint.get('num_updates', 0)


def make_agent(env, initial_obs, args, device=None) -> PPOAgent:
    """Factory function to create PPOAgent."""
    base_state = initial_obs['base']
    depth = initial_obs['depth']
    agent = PPOAgent(
        base_dim=base_state.shape[0],
        depth_shape=depth.shape,
        action_space=env.action_space,
        args=args,
        device=device,
    )
    return agent
