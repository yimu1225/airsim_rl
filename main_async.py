#!/usr/bin/env python3
"""
Asynchronous Training Main Script for AirSim RL
Supports multiple algorithms compatible with UAV_Navigation(RLlib) project.
"""
import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
import numpy as np
import torch
import csv
import cv2  # Added for visualization
import gc  # Added for memory management
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import tqdm

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
from algorithm.vmamba_td3.vmamba_td3 import VMambaTD3Agent
from algorithm.vmamba_td3_no_cross.vmamba_td3_no_cross import VMambaTD3NoCrossAgent
from algorithm.st_vmamba_td3.st_vmamba_td3 import ST_VMamba_Agent
from algorithm.st_mamba_td3.agent import ST_Mamba_Agent
from algorithm.ST_VimTD3.agent import ST_Mamba_VimTokens_Agent
from algorithm.ST_VimTD3_Safety.agent import ST_Mamba_VimTokens_Safety_Agent
from algorithm.st_cnn_td3.st_cnn_td3 import ST_CNN_Agent
from algorithm.gam_mamba_td3.td3 import GAMMambaTD3Agent


def _raise_if_non_finite(name, value, step_info=""):
    arr = np.asarray(value)
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        total = arr.size
        non_finite = total - int(finite_mask.sum())
        message = f"[NaNMonitor][{name}] non-finite detected: {non_finite}/{total} elements"
        if step_info:
            message = f"{message} | {step_info}"
        raise FloatingPointError(message)


def expand_algorithms(algo_str):
    """
    Expand algorithm string to list of individual algorithms.
    Supports comma-separated lists and predefined groups.
    """
    # Predefined algorithm groups
    groups = {
        'all': ['td3', 'aetd3', 'per_td3', 'per_aetd3', 'gru_td3', 'lstm_td3', 'gru_aetd3', 'lstm_aetd3', 'cfc_td3', 'vmamba_td3', 'vmamba_td3_no_cross', 'st_vmamba_td3', 'st_mamba_td3', 'ST-VimTD3', 'ST-VimTD3-Safety', 'st_cnn_td3', 'gam_mamba_td3'],
        'base': ['td3', 'aetd3', 'per_td3', 'per_aetd3'],
        'seq': ['gru_td3', 'lstm_td3', 'gru_aetd3', 'lstm_aetd3', 'cfc_td3', 'vmamba_td3', 'vmamba_td3_no_cross', 'st_vmamba_td3', 'st_mamba_td3', 'ST-VimTD3', 'ST-VimTD3-Safety', 'st_cnn_td3']
    }
    
    # Check if it's a predefined group
    if algo_str in groups:
        return groups[algo_str]
    
    # Check if it's comma-separated
    if ',' in algo_str:
        return [algo.strip() for algo in algo_str.split(',')]
    
    # Single algorithm
    return [algo_str]


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
    if algo_name == 'vmamba_td3': return VMambaTD3Agent
    if algo_name == 'vmamba_td3_no_cross': return VMambaTD3NoCrossAgent
    if algo_name == 'st_vmamba_td3': return ST_VMamba_Agent
    if algo_name == 'st_mamba_td3': return ST_Mamba_Agent
    if algo_name == 'ST-VimTD3': return ST_Mamba_VimTokens_Agent
    if algo_name == 'ST-VimTD3-Safety': return ST_Mamba_VimTokens_Safety_Agent
    if algo_name == 'st_cnn_td3': return ST_CNN_Agent
    if algo_name == 'gam_mamba_td3': return GAMMambaTD3Agent
    raise ValueError(f"Unknown algorithm: {algo_name}")

def main():
    args = get_config()
    seeds = args.seed if isinstance(args.seed, (list, tuple)) else [args.seed]

    # Expand algorithm names
    algorithms = expand_algorithms(args.algorithm_name)
    print(f"Training algorithms: {algorithms}")

    for seed in seeds:
        args.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Run training for each algorithm
        for algo_name in algorithms:
            print(f"\n{'='*50}")
            print(f"Training algorithm: {algo_name} (seed={seed})")
            print(f"{'='*50}")

            # Determine properties for this algorithm
            recurrent_algos = [
                'gru_td3', 'lstm_td3', 'gru_aetd3', 'lstm_aetd3', 'cfc_td3',
                'st_cnn_td3', 'vmamba_td3', 'st_vmamba_td3', 'st_mamba_td3', 'ST-VimTD3', 'ST-VimTD3-Safety', 'vmamba_td3_no_cross'  # vmamba_td3 也是时序算法
            ]
            
            is_recurrent = algo_name in recurrent_algos
            
            stack_frames = args.stack_frames_recurrent if is_recurrent else args.stack_frames

            # Initialize Environment
            print(f"Initialize AirSimEnv with stack_frames={stack_frames} for {algo_name} (seed={seed})...")
            env = AirSimEnv(need_render=args.need_render, takeoff_height=args.takeoff_height, config=args, stack_frames=stack_frames)

            # Initial Reset
            obs, _ = env.reset(seed=seed)
            # Obs is a dict keys: 'depth', 'base'
            
            depth_image = obs['depth']
            base_state = obs['base']

            # Dimensions
            base_dim = base_state.shape[0]
            depth_shape = depth_image.shape # (C, H, W)
            action_space = env.action_space

            print(f"Observation shapes: Depth {depth_shape}, Base {base_dim}")
            print(f"Action space: {action_space}")

            # Initialize Agent
            device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
            AgentClass = get_agent_class(algo_name)
            
            agent = AgentClass(base_dim, depth_shape, action_space, args, device=device)

            # Run training for this algorithm
            env = train_single_algorithm(env, agent, args, algo_name, is_recurrent, device, base_state, depth_image, stack_frames)


def train_single_algorithm(env, agent, args, algo_name, is_recurrent, device, base_state, depth_image, stack_frames):

    if args.load_model != "":
        print(f"Loading model: {args.load_model}")
        agent.load(args.load_model)

    print(f"Start Asynchronous Training {algo_name}...")

    # Restart interval for refreshing UE4 memory
    restart_interval = 200000
    next_restart = restart_interval

    # Logging
    if not os.path.exists("./results"): os.makedirs("./results")
    if not os.path.exists("./models"): os.makedirs("./models")
    
    # 移除时间戳，固定文件夹名称以实现覆盖 (Remove timestamp to use fixed folder for overwriting)
    run_name = f"{algo_name}_seed{args.seed}"
    log_dir = f"./results/{run_name}"
    
    # 如果文件夹已存在，则清理内容实现真正覆盖 (Clean directory if exists for true overwrite)
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
        
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize CSV logger
    csv_filename = os.path.join(log_dir, f"{algo_name}_seed{args.seed}_log.csv")
    with open(csv_filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['episode', 'total_timesteps', 'reward', 'episode_length', 'success_rate'])
    
    print(f"Logging to {csv_filename}")

    # History Buffers for Recurrent Policies
    if is_recurrent:
        seq_len = args.seq_len
        # Deques to store history
        base_hist = deque(maxlen=seq_len)
        depth_hist = deque(maxlen=seq_len)

        # Initialize history
        # Env returns (1, H, W) for recurrent.
        # We assume 'depth_image' is already (1, H, W).
        # We need to fill history.
        for _ in range(seq_len):
            base_hist.append(base_state)
            depth_hist.append(depth_image)

    # Training parameters
    max_timesteps = args.max_timesteps
    steps_per_update = args.steps_per_update
    start_timesteps = args.learning_starts

    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    action_hist = []

    state = depth_image
    base = base_state

    print("Start Asynchronous Training Loop...")

    if args.render_window:
        cv2.namedWindow("Depth View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth View", 256, 256)

    while total_timesteps < max_timesteps:
        # Collect steps_per_update
        for step in range(steps_per_update):
            episode_timesteps += 1
            total_timesteps += 1

            # Select Action
            if total_timesteps < start_timesteps and args.load_model == "":
                action = env.action_space.sample()
            else:
                if is_recurrent:
                    base_seq = np.array(base_hist) # (Seq, BaseDim)
                    depth_seq = np.array(depth_hist) # (Seq, 1, H, W)
                    
                    action = agent.select_action(base_seq, depth_seq)
                else:
                    # Non-recurrent: state is (4, H, W)
                    action = agent.select_action(base, state)

                _raise_if_non_finite(
                    "actor.action",
                    action,
                    f"algo={algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
                )

                # Track actions for distribution logging (per episode)
                action_hist.append(action)

            # Step
            try:
                next_obs, reward, terminated, truncated, step_info = env.step(action)
            except Exception as e:
                print(f"CRITICAL ERROR in env.step: {e}")
                print("Checking game status and attempting recovery...")
                
                # Force restart for robust recovery when UE process exists but window/sim is unhealthy
                if env.check_ue4_status(force_restart=True, reason="env_step_exception"):
                    # Force episode end and reset only after a restart
                    obs, _ = env.reset(seed=args.seed)
                    next_obs = obs  # Use fresh observation as 'next_obs' effectively
                    reward = 0.0
                    terminated = True  # End episode
                    truncated = False
                    step_info = {}
                else:
                    # If no restart was needed, still terminate the episode safely
                    reward = 0.0
                    terminated = True
                    truncated = False
                    step_info = {}
                
            done = terminated or truncated

            _raise_if_non_finite(
                "env.reward",
                reward,
                f"algo={algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
            )
            
            next_state = next_obs['depth']
            next_base = next_obs['base']

            _raise_if_non_finite(
                "env.next_depth",
                next_state,
                f"algo={algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
            )
            _raise_if_non_finite(
                "env.next_base",
                next_base,
                f"algo={algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
            )

            if args.render_window:
                vis_imgs = []
                if is_recurrent:
                    # 显示历史帧 + 当前帧 (Show history + current frame)
                    vis_imgs.extend(list(depth_hist))
                    vis_imgs.append(next_state)
                else:
                    # 显示所有堆叠帧 (Show all stacked frames)
                    if len(next_state.shape) == 3:
                        vis_imgs.extend([next_state[i] for i in range(next_state.shape[0])])
                    else:
                        vis_imgs.append(next_state)

                # 处理图像列表 (Process image list)
                processed_imgs = []
                for img in vis_imgs:
                    # 确保是 (H, W) 格式 (Ensure (H, W) format)
                    if len(img.shape) == 3:
                        img = img.squeeze(0) # Assume (1, H, W) -> (H, W)
                    
                    # 确保 uint8 类型 (Ensure uint8)
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    processed_imgs.append(img)
                
                # 水平拼接 (Horizontal concatenation)
                if processed_imgs:
                    vis_concat = np.hstack(processed_imgs)
                    # 动态调整窗口大小 (Dynamically resize window)
                    height, width = vis_concat.shape[:2]
                    cv2.resizeWindow("Depth View", width, height)
                    cv2.imshow("Depth View", vis_concat)
                    cv2.waitKey(1)

            episode_reward += reward

            # Add to Buffer
            done_bool = float(done)
            
            if is_recurrent:
                # Recurrent buffer usually expects single step data, sample() builds sequences.
                # Generic recurrent buffer: store current step only, next state is derived by shifting on sample.
                if algo_name == 'ST-VimTD3-Safety':
                    has_collided = float(step_info.get("has_collided", False)) if isinstance(step_info, dict) else 0.0
                    agent.replay_buffer.add(base, state, action, reward, done_bool, has_collided)
                else:
                    agent.replay_buffer.add(base, state, action, reward, done_bool)
                
                # Update history queues
                base_hist.append(next_base)
                depth_hist.append(next_state)
            else:
                # Non-recurrent buffer
                agent.replay_buffer.add(base, state, action, reward, next_base, next_state, done_bool)

            # State Update
            state = next_state
            base = next_base

            # Episode End Handling
            if done:
                # Calculate Success Rate based on recent history
                # env.success_deque is populated in env.step()
                success_rate = 0.0
                if len(env.success_deque) > 0:
                    success_rate = sum(env.success_deque) / len(env.success_deque)
                
                # Log to TensorBoard
                writer.add_scalar('train/episode_reward', episode_reward, episode_num)
                writer.add_scalar('train/episode_length', episode_timesteps, episode_num)
                writer.add_scalar('train/success_rate', success_rate, episode_num)

                # Log action distribution to TensorBoard (per episode)
                if len(action_hist) > 0:
                    action_arr = np.array(action_hist)
                    if action_arr.ndim == 1:
                        writer.add_histogram('train/action_distribution', action_arr, episode_num)
                    else:
                        # Record each action dimension separately
                        action_dim_names = ['forward_speed', 'vertical_speed', 'yaw_rate'] if action_arr.shape[1] == 3 else [f'action_dim_{i}' for i in range(action_arr.shape[1])]
                        for dim, name in enumerate(action_dim_names):
                            writer.add_histogram(f'train/{name}', action_arr[:, dim], episode_num)
                
                # Log to CSV
                with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([episode_num, total_timesteps, episode_reward, episode_timesteps, success_rate])

                print(f"[{algo_name.upper()}] Episode {episode_num}, Reward: {episode_reward:.2f}, Length: {episode_timesteps}, Success Rate: {success_rate:.2f}, Level: {env.level}, Total Timesteps: {total_timesteps}, Total Successes: {env.success_count}")
                
                episode_num += 1
                episode_reward = 0
                episode_timesteps = 0
                action_hist = []

                # Check if need to restart game
                if total_timesteps >= next_restart:
                    print(f"Restarting game to refresh UE4 memory at total_timesteps {total_timesteps}...")
                    # Save success stats and level before restart
                    old_success_count = env.success_count
                    old_success_deque = env.success_deque
                    old_level = env.level
                    old_game_config_handler = env.game_config_handler
                    # Close current environment
                    if hasattr(env, 'game_handler') and env.game_handler is not None:
                        env.game_handler.kill_game_in_editor()
                        time.sleep(2)  # Wait for kill
                    # Reinitialize environment
                    env = AirSimEnv(need_render=args.need_render, takeoff_height=args.takeoff_height, config=args, stack_frames=stack_frames)
                    # Restore success stats and level
                    env.success_count = old_success_count
                    env.success_deque = old_success_deque
                    env.level = old_level
                    env.game_config_handler = old_game_config_handler
                    next_restart += restart_interval

                # Reset
                obs, _ = env.reset(seed=args.seed + episode_num)
                state = obs['depth']
                base = obs['base']

                # Reset History
                if is_recurrent:
                    base_hist.clear()
                    depth_hist.clear()
                    
                    for _ in range(seq_len):
                        base_hist.append(base)
                        depth_hist.append(state)
                
                # Memory cleanup after episode end
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Training Update
        
        # Check buffer size property (some use .size(), some .size, we handle it generally)
        buffer_size = agent.replay_buffer.size if not callable(getattr(agent.replay_buffer, 'size', None)) else agent.replay_buffer.size()
        
        if buffer_size >= args.batch_size and total_timesteps >= start_timesteps:
            # Show progress bar for the update steps
            loss_info_list = []
            for _ in tqdm(range(steps_per_update), desc=f"Training ({total_timesteps})", leave=False):
                # Pass progress for schedulers if needed (conceptually)
                train_info = agent.train()
                if train_info:
                    for metric_name in ("actor_loss", "critic_loss"):
                        if metric_name in train_info:
                            _raise_if_non_finite(
                                f"train.{metric_name}",
                                train_info[metric_name],
                                f"algo={algo_name}, total_timesteps={total_timesteps}"
                            )
                    loss_info_list.append(train_info)
            
            # Log average loss every 100 training steps
            if loss_info_list and total_timesteps % 100 == 0:
                avg_actor_loss = sum(info.get('actor_loss', 0) for info in loss_info_list) / len(loss_info_list)
                avg_critic_loss = sum(info.get('critic_loss', 0) for info in loss_info_list) / len(loss_info_list)
                
                writer.add_scalar('loss/actor_loss', avg_actor_loss, total_timesteps)
                writer.add_scalar('loss/critic_loss', avg_critic_loss, total_timesteps)
            
            # Memory cleanup after training updates
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Checkpointing
        if total_timesteps % 100000 == 0:
            agent.save(f"./models/{algo_name}_async_{total_timesteps}.pth")
            print(f"Model saved at timestep {total_timesteps}")
            
            # Memory cleanup after checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    agent.save(f"./models/{algo_name}_async_final.pth")
    print("Training completed.")
    
    if args.render_window:
        cv2.destroyAllWindows()
        
    if hasattr(env, 'close'):
        env.close()
    writer.close()

    return env

if __name__ == "__main__":
    main()
