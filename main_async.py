#!/usr/bin/env python3
"""
Asynchronous Training Main Script for AirSim RL

This script implements asynchronous training where the agent collects a batch of
experiences (steps_per_update) before performing a single training update.
This is more efficient than updating after every step.

Usage:
python main_async.py --algorithm_name gru_td3 --steps_per_update 100 --max_timesteps 100000
"""

import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from config import get_config
from gym_airsim.envs.AirGym import AirSimEnv

def main():
    args = get_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize Environment
    print("Initialize AirSimEnv...")
    env = AirSimEnv(need_render=False, takeoff_height=args.takeoff_height, config=args)

    # State dim & Action dim
    obs, _ = env.reset(seed=args.seed)
    depth_image, inform = obs

    base_dim = inform.shape[0]
    depth_shape = depth_image.shape
    action_space = env.action_space

    print(f"Observation shapes: Depth {depth_shape}, Info {base_dim}")
    print(f"Action space: {action_space}")

    # Initialize Agent based on Algorithm
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    agent = None
    if args.algorithm_name == "td3":
        from algorithm.td3.td3 import TD3Agent
        agent = TD3Agent(base_dim, depth_shape, action_space, args, device=device)
    elif args.algorithm_name == "gru_td3":
        from algorithm.gru_td3.gru_td3 import GRUTD3Agent
        agent = GRUTD3Agent(base_dim, depth_shape, action_space, args, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm_name}")

    if args.load_model != "":
        print(f"Loading model: {args.load_model}")
        agent.load(args.load_model)

    # Check if agent is recurrent
    is_recurrent = args.recurrent_policy
    if args.algorithm_name == 'gru_td3':  # Explicit check
        is_recurrent = True

    print(f"Start Asynchronous Training {args.algorithm_name} (Recurrent: {is_recurrent})...")

    # Ensure directories exist
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    writer = SummaryWriter(log_dir=f"./results/{args.algorithm_name}_async_{args.seed}_{int(time.time())}")

    # For Recurrent Policies: Initialize history buffers
    if is_recurrent:
        seq_len = args.data_chunk_length
        base_hist = deque(maxlen=seq_len)
        depth_hist = deque(maxlen=seq_len)

        # Fill initial history with first observation
        current_frame = depth_image[-1:]  # Take last frame from stack
        for _ in range(seq_len):
            base_hist.append(inform)
            depth_hist.append(current_frame)

    # Training parameters
    max_timesteps = args.max_timesteps if hasattr(args, 'max_timesteps') else 100000
    steps_per_update = args.steps_per_update
    start_timesteps = args.start_timesteps if hasattr(args, 'start_timesteps') else 1000

    # Training loop
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0

    state = depth_image
    base = inform

    print("Start Asynchronous Training Loop...")

    while total_timesteps < max_timesteps:
        # Reset environment at start of episode
        if episode_timesteps == 0:
            obs, _ = env.reset(seed=args.seed + episode_num)
            depth_image, inform = obs
            state = depth_image
            base = inform
            episode_reward = 0

            if is_recurrent:
                # Reset history for new episode
                current_frame = depth_image[-1:]
                base_hist.clear()
                depth_hist.clear()
                for _ in range(seq_len):
                    base_hist.append(inform)
                    depth_hist.append(current_frame)

        # Collect steps_per_update experiences
        for step in range(steps_per_update):
            episode_timesteps += 1
            total_timesteps += 1

            # Select action
            if total_timesteps < start_timesteps and args.load_model == "":
                action = env.action_space.sample()
            else:
                if is_recurrent:
                    base_seq = np.array(base_hist)
                    depth_seq = np.array(depth_hist)
                    action = agent.select_action(base_seq, depth_seq)
                else:
                    action = agent.select_action(base, state)

            # Perform action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state, next_base = next_obs

            episode_reward += reward

            # Store transition in replay buffer
            done_bool = float(done)
            if is_recurrent:
                curr_frame = state[-1:]
                next_frame = next_state[-1:]
                agent.replay_buffer.add(base, curr_frame, action, reward, next_base, next_frame, done_bool)

                # Update history
                base_hist.append(next_base)
                depth_hist.append(next_frame)
            else:
                agent.replay_buffer.add(base, state, action, reward, next_base, next_state, done_bool)

            state = next_state
            base = next_base

            # Log episode end
            if done:
                writer.add_scalar('train/episode_reward', episode_reward, episode_num)
                writer.add_scalar('train/episode_length', episode_timesteps, episode_num)
                print(f"Episode {episode_num}: Reward {episode_reward:.2f}, Length {episode_timesteps}")
                episode_num += 1
                episode_reward = 0
                episode_timesteps = 0
                
                # Reset environment for next episode
                obs, _ = env.reset(seed=args.seed + episode_num)
                depth_image, inform = obs
                state = depth_image
                base = inform
                
                if is_recurrent:
                    current_frame = depth_image[-1:]
                    base_hist.clear()
                    depth_hist.clear()
                    for _ in range(seq_len):
                        base_hist.append(inform)
                        depth_hist.append(current_frame)

        # Perform training update after collecting steps_per_update
        if agent.replay_buffer.size() >= steps_per_update:
            agent.train()

        # Save model periodically
        if total_timesteps % 10000 == 0:
            agent.save(f"./models/{args.algorithm_name}_async_{total_timesteps}.pth")
            print(f"Model saved at timestep {total_timesteps}")

    # Final save
    agent.save(f"./models/{args.algorithm_name}_async_final.pth")
    print("Training completed. Final model saved.")

    env.close()
    writer.close()

if __name__ == "__main__":
    main()