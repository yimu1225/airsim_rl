import numpy as np
import torch


class RolloutBuffer:
    """
    Rollout buffer for PPO (on-policy).
    Stores trajectories and computes advantages using GAE.
    """
    
    def __init__(self, buffer_size, base_dim, depth_shape, action_dim, device, 
                 gamma=0.99, gae_lambda=0.95):
        """
        Args:
            buffer_size: maximum number of steps to store
            base_dim: dimension of base state
            depth_shape: shape of depth image (C, H, W)
            action_dim: dimension of action
            device: torch device
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.base_dim = base_dim
        self.depth_shape = depth_shape
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Pre-allocate arrays
        self.base_states = np.zeros((buffer_size, base_dim), dtype=np.float32)
        self.depth_states = np.zeros((buffer_size, *depth_shape), dtype=np.float16)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.returns = np.zeros((buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        
    def add(self, base_state, depth, action, reward, value, log_prob, done):
        """
        Add a single step to the buffer.
        
        Args:
            base_state: base state vector
            depth: depth image
            action: action taken
            reward: reward received
            value: value estimate V(s)
            log_prob: log probability of the action
            done: whether episode terminated
        """
        idx = self.ptr % self.buffer_size
        
        self.base_states[idx] = base_state
        self.depth_states[idx] = depth.astype(np.float16)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_value, last_done):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: value estimate of the last state
            last_done: whether the last state is terminal
            
        Returns:
            returns: discounted returns
            advantages: GAE advantages
        """
        # Get the trajectory slice
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # Append last value for bootstrap
        values = np.append(values, last_value)
        dones = np.append(dones, last_done)
        
        n_steps = len(rewards)
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0
        
        # Compute GAE backwards
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values[:-1]
        
        # Store returns and advantages for later retrieval
        traj_slice = slice(self.path_start_idx, self.ptr)
        self.returns[traj_slice] = returns
        self.advantages[traj_slice] = advantages
        
        return returns, advantages
    
    def get_trajectory(self):
        """
        Get the complete trajectory from path_start_idx to ptr.
        Called after compute_returns_and_advantages.
        
        Returns:
            batch data as tensors
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        
        return {
            'base_states': torch.as_tensor(self.base_states[path_slice], dtype=torch.float32, device=self.device),
            'depth_states': torch.as_tensor(self.depth_states[path_slice].astype(np.float32), dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(self.actions[path_slice], dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(self.rewards[path_slice], dtype=torch.float32, device=self.device),
            'returns': torch.as_tensor(self.returns[path_slice], dtype=torch.float32, device=self.device),
            'advantages': torch.as_tensor(self.advantages[path_slice], dtype=torch.float32, device=self.device),
            'values': torch.as_tensor(self.values[path_slice], dtype=torch.float32, device=self.device),
            'log_probs': torch.as_tensor(self.log_probs[path_slice], dtype=torch.float32, device=self.device),
            'dones': torch.as_tensor(self.dones[path_slice], dtype=torch.float32, device=self.device),
        }
    
    def after_update(self):
        """
        Reset buffer after policy update.
        """
        self.ptr = 0
        self.path_start_idx = 0
    
    def size(self):
        """Return current buffer size."""
        return self.ptr - self.path_start_idx
    
    def is_full(self):
        """Check if buffer is full."""
        return self.ptr >= self.buffer_size
