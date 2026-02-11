#!/usr/bin/env python3
"""
Synchronous Training Main Script for AirSim RL
Supports all algorithms from UAV_Navigation(RLlib).
Updates agent every step.
"""

import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from config import get_config
from gym_airsim.envs.AirGym import AirSimEnv

# Algorithm Imports
from algorithm.td3.td3 import TD3Agent
from algorithm.aetd3.aetd3 import AETD3Agent
from algorithm.per_td3.per_td3 import PERTD3Agent
from algorithm.per_aetd3.per_aetd3 import PERAETD3Agent
from algorithm.gru_td3.gru_td3 import GRUTD3Agent
from algorithm.lstm_td3.lstm_td3 import LSTMTD3Agent
from algorithm.gru_aetd3.gru_aetd3 import GRUAETD3Agent
from algorithm.lstm_aetd3.lstm_aetd3 import LSTMAETD3Agent
from algorithm.cfc_td3.cfc_td3 import CFCTD3Agent
from algorithm.lgmd_gru_td3.lgmd_gru_td3 import LGMDGRUTD3Agent
from algorithm.lgmd_lstm_td3.lgmd_lstm_td3 import LGMDLSTMTD3Agent
from algorithm.lgmd_gru_aetd3.lgmd_gru_aetd3 import LGMDGRUAETD3Agent
from algorithm.lgmd_lstm_aetd3.lgmd_lstm_aetd3 import LGMDLSTMAETD3Agent
from algorithm.vmamba_td3.vmamba_td3 import VMambaTD3Agent
from algorithm.vmamba_td3_no_cross.vmamba_td3_no_cross import VMambaTD3NoCrossAgent
from algorithm.st_vmamba_td3.st_vmamba_td3 import ST_VMamba_Agent
from algorithm.st_mamba_td3.agent import ST_Mamba_Agent
from algorithm.ST_VimTD3.agent import ST_Mamba_VimTokens_Agent
from algorithm.st_cnn_td3.st_cnn_td3 import ST_CNN_Agent

def get_agent_class(algo_name):
    if algo_name == 'td3': return TD3Agent
    if algo_name == 'aetd3': return AETD3Agent
    if algo_name == 'per_td3': return PERTD3Agent
    if algo_name == 'per_aetd3': return PERAETD3Agent
    if algo_name == 'gru_td3': return GRUTD3Agent
    if algo_name == 'lstm_td3': return LSTMTD3Agent
    if algo_name == 'gru_aetd3': return GRUAETD3Agent
    if algo_name == 'lstm_aetd3': return LSTMAETD3Agent
    if algo_name == 'cfc_td3': return CFCTD3Agent
    if algo_name == 'lgmd_gru_td3': return LGMDGRUTD3Agent
    if algo_name == 'lgmd_lstm_td3': return LGMDLSTMTD3Agent
    if algo_name == 'lgmd_gru_aetd3': return LGMDGRUAETD3Agent
    if algo_name == 'lgmd_lstm_aetd3': return LGMDLSTMAETD3Agent
    if algo_name == 'vmamba_td3': return VMambaTD3Agent
    if algo_name == 'vmamba_td3_no_cross': return VMambaTD3NoCrossAgent
    if algo_name == 'st_vmamba_td3': return ST_VMamba_Agent
    if algo_name == 'st_mamba_td3': return ST_Mamba_Agent
    if algo_name == 'ST-VimTD3': return ST_Mamba_VimTokens_Agent
    if algo_name == 'st_cnn_td3': return ST_CNN_Agent
    raise ValueError(f"Unknown algorithm: {algo_name}")

def main():
    args = get_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine stack_frames based on algorithm
    recurrent_algos = [
        'gru_td3', 'lstm_td3', 'gru_aetd3', 'lstm_aetd3', 'cfc_td3',
        'lgmd_gru_td3', 'lgmd_lstm_td3', 'lgmd_gru_aetd3', 'lgmd_lstm_aetd3',
        'vmamba_td3', 'st_vmamba_td3', 'st_mamba_td3', 'ST-VimTD3', 'vmamba_td3_no_cross', 'st_cnn_td3'  # VMamba和ST算法也是时序算法
    ]
    lgmd_algos = ['lgmd_gru_td3', 'lgmd_lstm_td3', 'lgmd_gru_aetd3', 'lgmd_lstm_aetd3']
    
    is_recurrent = args.algorithm_name in recurrent_algos
    is_lgmd = args.algorithm_name in lgmd_algos
    
    stack_frames = args.stack_frames_recurrent if is_recurrent else args.stack_frames

    # Initialize Environment
    # We use stack_frames=1 for recurrent policies to allow manual history management
    print(f"Initialize AirSimEnv with stack_frames={stack_frames} for {args.algorithm_name}...")
    env = AirSimEnv(need_render=False, takeoff_height=args.takeoff_height, config=args, stack_frames=stack_frames)

    # Initial Reset
    obs, _ = env.reset(seed=args.seed)
    
    depth_image = obs['depth']
    base_state = obs['base']
    gray_image = obs['gray']

    base_dim = base_state.shape[0]
    depth_shape = depth_image.shape
    gray_shape = gray_image.shape if is_lgmd else None
    action_space = env.action_space

    print(f"Observation shapes: Depth {depth_shape}, Base {base_dim}")
    if is_lgmd:
        print(f"Gray shape: {gray_shape}")

    # Initialize Agent
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    AgentClass = get_agent_class(args.algorithm_name)
    
    if is_lgmd:
        agent = AgentClass(base_dim, depth_shape, gray_shape, action_space, args, device=device)
    else:
        agent = AgentClass(base_dim, depth_shape, action_space, args, device=device)

    if args.load_model != "":
        print(f"Loading model: {args.load_model}")
        agent.load(args.load_model)

    print(f"Start Training {args.algorithm_name}...")

    # Logging
    if not os.path.exists("./results"): os.makedirs("./results")
    if not os.path.exists("./models"): os.makedirs("./models")
    writer = SummaryWriter(log_dir=f"./results/{args.algorithm_name}_sync_{args.seed}_{int(time.time())}")

    # History Buffers
    if is_recurrent:
        seq_len = args.data_chunk_length
        base_hist = deque(maxlen=seq_len)
        depth_hist = deque(maxlen=seq_len)
        if is_lgmd:
            gray_hist = deque(maxlen=seq_len)

        for _ in range(seq_len):
            base_hist.append(base_state)
            depth_hist.append(depth_image)
            if is_lgmd:
                gray_hist.append(gray_image)

    max_timesteps = args.max_timesteps
    start_timesteps = args.learning_starts

    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0

    state = depth_image
    base = base_state
    gray = gray_image

    print("Start Synchronous Training Loop...")

    while total_timesteps < max_timesteps:
        episode_timesteps += 1
        total_timesteps += 1

        # Select Action
        if total_timesteps < start_timesteps and args.load_model == "":
            action = env.action_space.sample()
        else:
            if is_recurrent:
                base_seq = np.array(base_hist)
                depth_seq = np.array(depth_hist)
                if is_lgmd:
                    gray_seq = np.array(gray_hist)
                    action = agent.select_action(base_seq, depth_seq, gray_seq)
                else:
                    action = agent.select_action(base_seq, depth_seq)
            else:
                action = agent.select_action(base, state)

        # Step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = next_obs['depth']
        next_base = next_obs['base']
        next_gray = next_obs['gray']

        episode_reward += reward

        # Add to Buffer
        done_bool = float(done)
        
        if is_recurrent:
            if is_lgmd:
                agent.replay_buffer.add(base, state, gray, action, reward, done_bool)
            else:
                agent.replay_buffer.add(base, state, action, reward, done_bool)
            
            base_hist.append(next_base)
            depth_hist.append(next_state)
            if is_lgmd:
                gray_hist.append(next_gray)
        else:
            agent.replay_buffer.add(base, state, action, reward, next_base, next_state, done_bool)

        state = next_state
        base = next_base
        gray = next_gray

        # Train (Synchronous)
        if total_timesteps >= start_timesteps:
            # 计算训练进度比例，用于OU噪声衰减
            progress_ratio = total_timesteps / max_timesteps
            agent.train(progress_ratio)

        # Episode End
        if done:
            writer.add_scalar('train/episode_reward', episode_reward, episode_num)
            writer.add_scalar('train/episode_length', episode_timesteps, episode_num)
            print(f"[{args.algorithm_name.upper()}] Episode {episode_num}, Reward: {episode_reward:.2f}, Length: {episode_timesteps}, Total Steps: {total_timesteps}")
            
            episode_num += 1
            episode_reward = 0
            episode_timesteps = 0

            obs, _ = env.reset(seed=args.seed + episode_num)
            state = obs['depth']
            base = obs['base']
            gray = obs['gray']

            if is_recurrent:
                base_hist.clear()
                depth_hist.clear()
                if is_lgmd:
                    gray_hist.clear()
                
                for _ in range(seq_len):
                    base_hist.append(base)
                    depth_hist.append(state)
                    if is_lgmd:
                        gray_hist.append(gray)
        
        # Save
        if total_timesteps % 10000 == 0:
            agent.save(f"./models/{args.algorithm_name}_sync_{total_timesteps}.pth")

    agent.save(f"./models/{args.algorithm_name}_sync_final.pth")
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
