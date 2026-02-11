import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
import numpy as np
import torch
import argparse
from gym_airsim.envs.AirGym import AirSimEnv
from algorithm.td3.td3 import TD3Agent
from torch.utils.tensorboard import SummaryWriter
from config import get_config

def main():
    args = get_config()
    
    # 确保文件夹存在
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    # 创建环境
    print("Initialize AirSimEnv...")
    
    env = AirSimEnv(need_render=False, takeoff_height=args.takeoff_height, config=args) 

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # State dim & Action dim
    # env.reset() returns (obs_dict, info)
    obs, _ = env.reset()
    depth_image = obs['depth']
    inform = obs['base']
    
    base_dim = inform.shape[0]
    depth_shape = depth_image.shape
    action_space = env.action_space
    
    print(f"Observation shapes: Depth {depth_shape}, Info {base_dim}")
    print(f"Action space: {action_space}")

    # Initialize Agent
    agent = TD3Agent(base_dim, depth_shape, action_space, args)
    
    if args.load_model != "":
         print(f"Loading model: {args.load_model}")
         agent.load(args.load_model)
    
    writer = SummaryWriter(log_dir=f"./results/td3_{args.seed}_{int(time.time())}")

    # Training Loop
    total_timesteps = 0
    episode_num = 0
    
    # Reset env vars
    state = depth_image
    # inform is already set
    
    episode_reward = 0
    episode_timesteps = 0
    
    print("Start Training...")
    
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        
        # Select action
        if total_timesteps < args.learning_starts and args.load_model == "":
            action = env.action_space.sample()
        else:
            action = agent.select_action(inform, state) # noise is True by default

        # Perform action
        # env.step returns (state, reward, terminated, truncated, info)
        # state is [depth, inform]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = next_obs['depth']
        next_inform = next_obs['base']
        
        # Store data in replay buffer
        done_bool = float(done) 
        
        agent.replay_buffer.add(inform, state, action, reward, next_inform, next_state, done_bool)

        state = next_state
        inform = next_inform
        episode_reward += reward

        # Train (if enough samples)
        if total_timesteps >= args.learning_starts:
            train_info = agent.train()
            if train_info:
                if total_timesteps % 100 == 0:
                    for k, v in train_info.items():
                        writer.add_scalar(f"loss/{k}", v, total_timesteps)

        if done: 
            print(f"Total T: {total_timesteps+1} | Episode: {episode_num+1} | Steps: {episode_timesteps} | Reward: {episode_reward:.3f}")
            writer.add_scalar("reward/episode_reward", episode_reward, total_timesteps)
            writer.add_scalar("reward/avg_reward_per_step", episode_reward/episode_timesteps, total_timesteps)
            
            # Reset environment
            obs, _ = env.reset()
            state, inform = obs
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            
            # Reset noise if using OU noise (optional, some implementations reset noise on episode start)
            agent.ou_noise.reset()

        # Save model
        if (total_timesteps + 1) % args.eval_freq == 0:
            save_path = f"./models/td3_step_{total_timesteps+1}.pth"
            agent.save(save_path)
            print(f"Model saved to {save_path}")

        total_timesteps += 1

    writer.close()

if __name__ == "__main__":
    main()
