import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from collections import deque
import copy

from .networks import STE_CNN_Encoder, Actor, Critic, StateMLP
from ..lstm_td3.buffer import SequenceReplayBuffer

class ST_CNN_Agent:
    """
    ST-CNN-TD3 Algorithm.
    Contrast Method: "Sequence Input -> CNN (Spatial) -> 1D Mamba (Temporal) -> Self-Attention (Spatial Refine) -> Pooling -> Concat"
    """
    def __init__(self, base_dim, depth_shape, action_space, args, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"ST-CNN-TD3 Agent using device: {self.device}")
        
        self.args = args
        self.base_dim = base_dim
        self.depth_shape = depth_shape # (C, H, W)
        self.seq_len = getattr(args, 'seq_len', 3) # Temporal sequence length
        
        self.action_dim = action_space.shape[0]
        self.max_action = float(action_space.high[0])
        
        # Action Scaling
        self.min_action = float(action_space.low[0])
        self.action_scale = torch.tensor((self.max_action - self.min_action) / 2.0, device=self.device)
        self.action_bias = torch.tensor((self.max_action + self.min_action) / 2.0, device=self.device)

        # Internal history buffer for inference - REMOVED, controlled by main_async
        # self.history = deque(maxlen=self.seq_len) 

        # --- Actor ---
        # 1. Visual Encoder (STE-CNN)
        self.actor_visual = STE_CNN_Encoder(
            img_size=(depth_shape[1], depth_shape[2]),
            in_chans=depth_shape[0], # Should be 1 typically for depth
            feature_dim=getattr(args, 'feature_dim', 128),
            args=args
        ).to(self.device)
        
        # 2. State Encoder
        state_hidden_dim = getattr(args, 'state_feature_dim', 64)
        self.actor_state_net = StateMLP(base_dim, state_hidden_dim).to(self.device)
        
        # Determine feature dim for head
        feature_dim = self.actor_visual.out_dim + state_hidden_dim
        
        # 3. Head
        self.actor_head = Actor(
            feature_dim=feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        # --- Critic ---
        self.critic_visual = STE_CNN_Encoder(
            img_size=(depth_shape[1], depth_shape[2]),
            in_chans=depth_shape[0],
            feature_dim=getattr(args, 'feature_dim', 128),
            args=args
        ).to(self.device)
        
        self.critic_state_net = StateMLP(base_dim, state_hidden_dim).to(self.device)
        
        self.critic_head = Critic(
            feature_dim=feature_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)

        # --- Target Networks ---
        self.actor_visual_target = copy.deepcopy(self.actor_visual)
        self.actor_state_net_target = copy.deepcopy(self.actor_state_net)
        self.actor_head_target = copy.deepcopy(self.actor_head)
        
        self.critic_visual_target = copy.deepcopy(self.critic_visual)
        self.critic_state_net_target = copy.deepcopy(self.critic_state_net)
        self.critic_head_target = copy.deepcopy(self.critic_head)

        # Optimizers
        lr = getattr(args, 'lr', 1e-4) # default lr
        self.actor_optimizer = Adam(
            list(self.actor_visual.parameters()) + 
            list(self.actor_state_net.parameters()) + 
            list(self.actor_head.parameters()), 
            lr=lr
        )
        self.critic_optimizer = Adam(
            list(self.critic_visual.parameters()) + 
            list(self.critic_state_net.parameters()) + 
            list(self.critic_head.parameters()), 
            lr=lr
        )

        self.gamma = getattr(args, 'gamma', 0.99)
        self.tau = getattr(args, 'tau', 0.005)
        self.policy_noise = getattr(args, 'policy_noise', 0.2) * self.max_action
        self.noise_clip = getattr(args, 'noise_clip', 0.5) * self.max_action
        self.policy_freq = getattr(args, 'policy_freq', 2)

        self.exploration_noise = getattr(args, 'exploration_noise', 0.1)

        self.total_it = 0
        
        # Replay Buffer
        self.replay_buffer = SequenceReplayBuffer(getattr(args, 'buffer_size', 100000), self.seq_len)

    def select_action(self, base_state, depth_img, noise: bool = True):
        # base_state: (Seq, BaseDim)
        # depth_img: (Seq, C, H, W)

        # Ensure correct shape
        if isinstance(base_state, np.ndarray):
            base_state = torch.FloatTensor(base_state).to(self.device)
        if isinstance(depth_img, np.ndarray):
            depth_img = torch.FloatTensor(depth_img).to(self.device)
        
        # Add batch dimension: (Seq, ...) -> (1, Seq, ...)
        if base_state.dim() == 2:
            base_state = base_state.unsqueeze(0)
        if depth_img.dim() == 4:
            depth_img = depth_img.unsqueeze(0)
            
        # Use only the latest base_state for the MLP part (unless we want to fuse sequence state too)
        # Architecture design usually fuses Visual Sequence + Current State
        current_base_state = base_state[:, -1, :] # (1, BaseDim)
        
        with torch.no_grad():
            visual_feat = self.actor_visual(depth_img) # (1, D)
            state_feat = self.actor_state_net(current_base_state) # (1, S)

            feat = torch.cat([visual_feat, state_feat], dim=1)
            action = self.actor_head(feat).cpu().numpy().flatten()

        if noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise

        action = np.clip(action, -1.0, 1.0)
        scaled_action = action * self.action_scale.cpu().numpy() + self.action_bias.cpu().numpy()
        return scaled_action

    def train(self, replay_buffer=None, batch_size=64):
        self.total_it += 1

        # Use internal buffer if not provided (Compatibility with generic training loop)
        if replay_buffer is None:
            replay_buffer = self.replay_buffer
            
       

        state, depth, action, reward, next_state, next_depth, not_done = replay_buffer.sample(batch_size)
        
        depth = torch.FloatTensor(depth).to(self.device)
        state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 3:
            current_state = state[:, -1, :]
        else:
            current_state = state
            
        action = torch.FloatTensor(action).to(self.device)
        if action.dim() == 3 and action.shape[1] == self.seq_len:
             action = action[:, -1, :]
            
        next_state = torch.FloatTensor(next_state).to(self.device)
        
        if next_state.dim() == 3:
            next_state_curr = next_state[:, -1, :]
        else:
            next_state_curr = next_state
            
        next_depth = torch.FloatTensor(next_depth).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)
        
        # Adjust shapes for Sequence Replay Buffer (B, T, ...) -> Take last step for TD3 update
        if reward.dim() > 1 and reward.shape[1] == self.seq_len:
            if reward.dim() == 2: # (B, T)
                reward = reward[:, -1].unsqueeze(1)
            elif reward.dim() == 3: # (B, T, 1)
                reward = reward[:, -1, :]
                
        if not_done.dim() > 1 and not_done.shape[1] == self.seq_len:
            if not_done.dim() == 2:
                not_done = not_done[:, -1].unsqueeze(1)
            elif not_done.dim() == 3:
                not_done = not_done[:, -1, :]

        with torch.no_grad():
            # Add noise to target action (in scaled space)
            # Standard TD3 adds noise to pi(s')
            # pi(s') = tanh(net(s')) * scale + bias
            # We add noise to the scaled action
            
            # (B, T, C, H, W) -> STE-CNN -> (B, D)
            next_vis_feat = self.actor_visual_target(next_depth)
            next_s_feat = self.actor_state_net_target(next_state_curr)
            next_feat = torch.cat([next_vis_feat, next_s_feat], dim=1)
            
            raw_next_action = self.actor_head_target(next_feat)
            scaled_next_action = raw_next_action * self.action_scale + self.action_bias
            
            noise = (torch.randn_like(scaled_next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (scaled_next_action + noise).clamp(self.min_action, self.max_action)

            # Target Q
            t_vis_feat_q = self.critic_visual_target(next_depth)
            t_s_feat_q = self.critic_state_net_target(next_state_curr)
            t_feat_q = torch.cat([t_vis_feat_q, t_s_feat_q], dim=1)
            
            target_Q1, target_Q2 = self.critic_head_target(t_feat_q, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.gamma * target_Q).detach()

        # Critic Update
        vis_feat_q = self.critic_visual(depth)
        s_feat_q = self.critic_state_net(current_state)
        feat_q = torch.cat([vis_feat_q, s_feat_q], dim=1)
        
        current_Q1, current_Q2 = self.critic_head(feat_q, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        if self.total_it % self.policy_freq == 0:
            vis_feat_pi = self.actor_visual(depth)
            s_feat_pi = self.actor_state_net(current_state)
            feat_pi = torch.cat([vis_feat_pi, s_feat_pi], dim=1)
            
            raw_pi = self.actor_head(feat_pi)
            scaled_pi = raw_pi * self.action_scale + self.action_bias
            
            actor_loss = -self.critic_head.q1_net(torch.cat([feat_q.detach(), scaled_pi], dim=1)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Update
            self.soft_update(self.actor_visual, self.actor_visual_target, self.tau)
            self.soft_update(self.actor_state_net, self.actor_state_net_target, self.tau)
            self.soft_update(self.actor_head, self.actor_head_target, self.tau)
            
            self.soft_update(self.critic_visual, self.critic_visual_target, self.tau)
            self.soft_update(self.critic_state_net, self.critic_state_net_target, self.tau)
            self.soft_update(self.critic_head, self.critic_head_target, self.tau)
            
            return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': 0.0}

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor_visual.state_dict(), filename + "_actor_visual")
        torch.save(self.actor_state_net.state_dict(), filename + "_actor_state")
        torch.save(self.actor_head.state_dict(), filename + "_actor_head")
        torch.save(self.critic_visual.state_dict(), filename + "_critic_visual")
        torch.save(self.critic_state_net.state_dict(), filename + "_critic_state")
        torch.save(self.critic_head.state_dict(), filename + "_critic_head")

    def load(self, filename):
        self.actor_visual.load_state_dict(torch.load(filename + "_actor_visual"))
        self.actor_state_net.load_state_dict(torch.load(filename + "_actor_state"))
        self.actor_head.load_state_dict(torch.load(filename + "_actor_head"))
        self.critic_visual.load_state_dict(torch.load(filename + "_critic_visual"))
        self.critic_state_net.load_state_dict(torch.load(filename + "_critic_state"))
        self.critic_head.load_state_dict(torch.load(filename + "_critic_head"))
