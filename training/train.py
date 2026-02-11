import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

def train(env, agent, args):
    # Ensure directories exist
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    writer = SummaryWriter(log_dir=f"./results/{args.algorithm_name}_{args.seed}_{int(time.time())}")
    
    # Check if agent is recurrent
    is_recurrent = args.recurrent_policy
    if args.algorithm_name == 'gru_td3': # Explicit check
        is_recurrent = True

    print(f"Start Training {args.algorithm_name} (Recurrent: {is_recurrent})...")

    # Reset env
    obs, _ = env.reset(seed=args.seed)
    depth_image, inform = obs
    
    # For Recurrent Policies: Initialize history buffers
    if is_recurrent:
        seq_len = args.seq_len
        # Deques to store history. 
        # Repeat the first frame to fill buffer
        base_hist = deque(maxlen=seq_len)
        depth_hist = deque(maxlen=seq_len)
        
        # Extract the latest frame (single frame) from the stack
        # depth_image shape is (stack_frames, H, W) e.g., (4, 112, 112)
        # We need (1, 112, 112) for the recurrent sequence
        current_frame = depth_image[-1:] 
        
        for _ in range(seq_len):
            base_hist.append(inform)
            depth_hist.append(current_frame)
            
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    total_timesteps = 0
    
    start_timesteps = args.learning_starts
    max_timesteps = args.max_timesteps
    
    state = depth_image # Current depth
    base = inform       # Current base info

    print("Start Training Loop...")

    while total_timesteps < max_timesteps:
        episode_timesteps += 1
        
        # Select action
        if total_timesteps < start_timesteps and args.load_model == "":
            action = env.action_space.sample()
        else:
            if is_recurrent:
                # Stack history to create sequence
                # Shape: (K, ...)
                base_seq = np.array(base_hist)
                depth_seq = np.array(depth_hist)
                action = agent.select_action(base_seq, depth_seq)
            else:
                action = agent.select_action(base, state)

        # Perform action
        # env.step returns (state, reward, terminated, truncated, info)
        # state is [depth, inform]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state, next_base = next_obs
        
        done_bool = float(done)
        
        if is_recurrent:
             # Extract single frames for GRU
             # state shape: (4, 112, 112) -> take last: (1, 112, 112)
             curr_frame = state[-1:]
             next_frame = next_state[-1:]
             
             agent.replay_buffer.add(base, curr_frame, action, reward, done_bool)
             
             # Update history
             base_hist.append(next_base)
             depth_hist.append(next_frame)
        else:
             agent.replay_buffer.add(base, state, action, reward, next_base, next_state, done_bool)

        state = next_state
        base = next_base
        episode_reward += reward

        # Train
        if total_timesteps >= start_timesteps:
            train_info = agent.train()
            if train_info:
                if total_timesteps % 100 == 0:
                     for k, v in train_info.items():
                        writer.add_scalar(f"loss/{k}", v, total_timesteps)

        if done:
            print(f"Total T: {total_timesteps+1} | Episode: {episode_num+1} | Steps: {episode_timesteps} | Reward: {episode_reward:.3f}")
            writer.add_scalar("reward/episode_reward", episode_reward, total_timesteps)
            writer.add_scalar("reward/avg_reward_per_step", episode_reward/episode_timesteps, total_timesteps)
            
            # Reset
            obs, _ = env.reset()
            state, base = obs
            
            if is_recurrent:
                 base_hist.clear()
                 depth_hist.clear()
                 # Extract single frame
                 current_frame = state[-1:]
                 for _ in range(seq_len):
                    base_hist.append(base)
                    depth_hist.append(current_frame)
            
            # Reset OU noise if available
            if hasattr(agent, 'ou_noise'):
                agent.ou_noise.reset()
                
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Save
        if (total_timesteps + 1) % args.eval_freq == 0:
            save_path = f"./models/{args.algorithm_name}_step_{total_timesteps+1}.pth"
            agent.save(save_path)
            print(f"Model saved to {save_path}")
            
        total_timesteps += 1

    writer.close()
