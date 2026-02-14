import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import Actor, Critic, Encoder
from .buffer import MultiPoolAdaptiveBuffer
from ..ou_noise import OUNoise

class MPPTD3Agent:
    def __init__(self, base_dim, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.args = args
        
        self.base_dim = base_dim
        self.depth_shape = depth_shape
        self.action_dim = action_space.shape[0]
        self.max_action = np.array(action_space.high, dtype=np.float32)
        self.min_action = np.array(action_space.low, dtype=np.float32)
        self.action_scale = torch.FloatTensor((self.max_action - self.min_action) / 2.0).to(self.device)
        self.action_bias = torch.FloatTensor((self.max_action + self.min_action) / 2.0).to(self.device)
        
        # Encoders
        _, depth_h, depth_w = depth_shape
        self.actor_encoder = Encoder(depth_h, depth_w, args.feature_dim).to(self.device)
        self.critic_encoder = Encoder(depth_h, depth_w, args.feature_dim).to(self.device)
        self.actor_encoder_target = Encoder(depth_h, depth_w, args.feature_dim).to(self.device)
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        self.critic_encoder_target = Encoder(depth_h, depth_w, args.feature_dim).to(self.device)
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
        
        self.state_dim = self.base_dim + self.actor_encoder.repr_dim
        
        # Actor & Critic
        self.actor = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_space.shape, args.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = Adam(list(self.actor.parameters()) + list(self.actor_encoder.parameters()), lr=args.actor_lr)
        self.critic_optimizer = Adam(list(self.critic.parameters()) + list(self.critic_encoder.parameters()), lr=args.critic_lr)
        
        self.max_action_tensor = torch.FloatTensor(self.max_action).to(self.device)
        self.min_action_tensor = torch.FloatTensor(self.min_action).to(self.device)

        # Advanced Buffer
        self.replay_buffer = MultiPoolAdaptiveBuffer(
            total_capacity=args.buffer_size, 
            random_capacity=args.learning_starts
        )
        
        # Noise Systems
        self.exploration_noise = args.exploration_noise
        self.ou_noise = OUNoise(
            size=self.action_dim,
            mu=0.0,
            theta=getattr(args, "ou_theta", 0.15),
            sigma=getattr(args, "ou_sigma", 0.2),
            sigma_min=getattr(args, "ou_sigma_min", 0.01),
            dt=getattr(args, "ou_dt", 1.0)
        )
        self.ou_noise.reset()
        
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.batch_size = args.batch_size
        
        self.total_it = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_sampling_weights(self, progress_ratio):
        # Using exponential and sigmoid functions for differentiability and consistent convexity
        # Random: exponential decay from 1 at pr=0
        random_base = np.exp(-50 * progress_ratio)
        
        # Noisy: rise from 0 then fall, using exponentials and sigmoid for smooth transitions
        noisy_rise = 1 - np.exp(-20 * progress_ratio)
        noisy_fall = 1 - self._sigmoid(10 * (progress_ratio - 0.75))  # Smooth fall starting at ~0.75
        noisy_base = 0.4 * noisy_rise * noisy_fall
        
        # Clean: rise from 0 late in training using sigmoid
        clean_base = 0.5 * self._sigmoid(10 * (progress_ratio - 0.8))  # Smooth rise starting at ~0.8
        
        total = random_base + noisy_base + clean_base
        return [random_base/total, noisy_base/total, clean_base/total]

    def compute_adaptive_noise_std(self, obstacle_dist):
        max_dist = getattr(self.args, 'depth_max_distance', 10.0)
        normalized_dist = min(obstacle_dist / max_dist, 1.0)
        
        max_noise = 0.5
        min_noise = 0.05
        decay_factor = 2.0
        
        noise_ratio = np.exp(-decay_factor * normalized_dist)
        return min_noise + (max_noise - min_noise) * noise_ratio

    def select_action(self, state, exploration_prob, obstacle_dist, global_step):
        # Process state for NN
        base_state = torch.FloatTensor(state[0]).unsqueeze(0).to(self.device)
        depth_state = torch.FloatTensor(state[1]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.actor_encoder(depth_state)
            full_state = torch.cat([base_state, feat], dim=1)
            clean_action = self.actor(full_state).cpu().data.numpy().flatten()
            
        noise_type = 'clean'
        action = clean_action
        
        if global_step < self.args.learning_starts:
            action = np.random.uniform(self.min_action, self.max_action, size=self.action_dim)
            noise_type = 'random'
            clean_action = None
        elif np.random.rand() < exploration_prob:
            # Multi-layered noise
            # Gaussian only as requested
            action = clean_action + self.ou_noise.sample()
            noise_type = 'noisy'
            
        return np.clip(action, self.min_action, self.max_action), clean_action, noise_type

    def train(self, progress_ratio=0.0):
        self.total_it += 1

        self.ou_noise.scale_sigma(progress_ratio)
        
        # Beta scheduling for PER
        beta = 0.4 + (1.0 - 0.4) * progress_ratio
        
        # Sample using adaptive weights
        weights = self.get_sampling_weights(progress_ratio)
        samples, indices, importance_weights = self.replay_buffer.sample(self.batch_size, weights, beta)
        
        if samples is None: return {}
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to tensors
        # Assuming state is (base, depth)
        state_base = torch.FloatTensor(np.array([s[0] for s in states])).to(self.device)
        state_depth = torch.FloatTensor(np.array([s[1] for s in states])).to(self.device)
        action = torch.FloatTensor(np.array(actions)).to(self.device)
        action = (action - self.action_bias) / self.action_scale
        action = action.clamp(-1.0, 1.0)
        reward = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_state_base = torch.FloatTensor(np.array([s[0] for s in next_states])).to(self.device)
        next_state_depth = torch.FloatTensor(np.array([s[1] for s in next_states])).to(self.device)
        done = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        i_weights = torch.FloatTensor(importance_weights).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Target action with noise
            next_feat = self.actor_encoder_target(next_state_depth)
            next_full_state = torch.cat([next_state_base, next_feat], dim=1)
            next_action = self.actor_target(next_full_state)
            
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            
            target_q1, target_q2 = self.critic_target(next_full_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        # Current Q
        feat = self.critic_encoder(state_depth)
        full_state = torch.cat([state_base, feat], dim=1)
        current_q1, current_q2 = self.critic(full_state, action)
        
        # Critic loss with importance weights
        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q
        critic_loss = (i_weights * (F.mse_loss(current_q1, target_q, reduction='none') + 
                                   F.mse_loss(current_q2, target_q, reduction='none'))).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update PER priorities
        new_priorities = (torch.abs(td_error1) + torch.abs(td_error2)).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, new_priorities)
        
        result = {"critic_loss": critic_loss.item()}

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            feat_actor = self.actor_encoder(state_depth)
            full_state_actor = torch.cat([state_base, feat_actor], dim=1)
            q1_pi, _ = self.critic(full_state_actor, self.actor(full_state_actor))
            actor_loss = -q1_pi.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # Encoder target updates
            for param, target_param in zip(self.critic_encoder.parameters(), self.critic_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_encoder.parameters(), self.actor_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            result["actor_loss"] = actor_loss.item()
            
        return result

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_encoder_target.load_state_dict(self.actor_encoder.state_dict())
        self.critic_encoder_target.load_state_dict(self.critic_encoder.state_dict())
