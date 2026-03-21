#!/usr/bin/env python3
"""
Asynchronous Training Main Script for AirSim RL
Supports multiple algorithms compatible with UAV_Navigation(RLlib) project.
"""
import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import time
import random
import numpy as np
import torch
import csv
import cv2  # Added for visualization
import gc  # Added for memory management
import inspect
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from gym_airsim.envs.AirGym import AirSimEnv

# Algorithm Imports
from algorithm.td3.td3 import TD3Agent
from algorithm.ddpg.ddpg import DDPGAgent
from algorithm.aetd3.aetd3 import AETD3Agent
from algorithm.per_td3.per_td3 import PERTD3Agent
from algorithm.per_aetd3.per_aetd3 import PERAETD3Agent

from algorithm.cfc_td3.cfc_td3 import CFCTD3Agent
from algorithm.st_mamba_td3.agent import ST_Mamba_Agent
from algorithm.ST_VimTD3.agent import STVimTD3Agent
from algorithm.ST_SVimTD3.agent import STSVimTD3Agent
from algorithm.st_cnn_td3.st_cnn_td3 import ST_CNN_Agent
from algorithm.gam_mamba_td3.td3 import GAMMambaTD3Agent
from algorithm.ST_3DVimTD3.agent import ST3DVimTD3Agent
from algorithm.st_dualvim_td3.agent import DualBranchVideoMambaTD3Agent



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


def _configure_reproducibility(seed: int, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(getattr(args, "cuda_deterministic", True))
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def expand_algorithms(algo_str):
    """
    Expand algorithm string to list of individual algorithms.
    Supports comma-separated lists and predefined groups.
    """
    # Predefined algorithm groups
    groups = {
        'all': ['td3', 'ddpg', 'aetd3', 'per_td3', 'per_aetd3', 'cfc_td3', 'st_mamba_td3', 'ST-VimTD3', 'ST-SVimTD3', 'st_cnn_td3', 'gam_mamba_td3', 'ST_3DVimTD3', 'ST-DualVimTD3'],
        'base': ['td3', 'ddpg', 'aetd3', 'per_td3', 'per_aetd3'],
        'seq': ['cfc_td3', 'st_mamba_td3', 'ST-VimTD3', 'ST-SVimTD3', 'st_cnn_td3', 'ST_3DVimTD3', 'ST-DualVimTD3']
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
    # 去掉 CL- 前缀（如果存在）
    if algo_name.startswith("CL-"):
        algo_name = algo_name[3:]
    
    agents = {
        'td3': TD3Agent,
        'ddpg': DDPGAgent,
        'aetd3': AETD3Agent,
        'per_td3': PERTD3Agent,
        'per_aetd3': PERAETD3Agent,
        'cfc_td3': CFCTD3Agent,
        'st_mamba_td3': ST_Mamba_Agent,
        'ST-VimTD3': STVimTD3Agent,
        'ST-SVimTD3': STSVimTD3Agent,
        'st_cnn_td3': ST_CNN_Agent,
        'gam_mamba_td3': GAMMambaTD3Agent,
        'ST_3DVimTD3': ST3DVimTD3Agent,
        'ST-DualVimTD3': DualBranchVideoMambaTD3Agent,
        # Backward-compatible alias
        'dual_videomamba_td3': DualBranchVideoMambaTD3Agent,
    }
    if algo_name in agents:
        return agents[algo_name]
    raise ValueError(f"Unknown algorithm: {algo_name}")

def main():
    args = get_config()
    seeds = args.seed if isinstance(args.seed, (list, tuple)) else [args.seed]

    # Expand algorithm names
    algorithms = expand_algorithms(args.algorithm_name)
    print(f"Training algorithms: {algorithms}")

    for seed in seeds:
        args.seed = seed
        _configure_reproducibility(seed, args)

        # Run training for each algorithm
        for algo_name in algorithms:
            print(f"\n{'='*50}")
            print(f"Training algorithm: {algo_name} (seed={seed})")
            print(f"{'='*50}")

            # 根据算法名判断是否使用课程学习
            # 算法名以 "CL-" 开头时启用课程学习
            if algo_name.startswith("CL-"):
                # 去掉 CL- 前缀获取实际算法名
                actual_algo_name = algo_name[3:]
                print(f"  [Curriculum Learning Enabled] {actual_algo_name}")
            else:
                actual_algo_name = algo_name
                print(f"  [Curriculum Learning Disabled] {algo_name}")

            # Determine properties for this algorithm
            recurrent_algos = [
                'cfc_td3', 'st_cnn_td3', 'st_mamba_td3', 'ST-VimTD3', 'ST-SVimTD3', 'ST_3DVimTD3'
            ]
            
            is_recurrent = actual_algo_name in recurrent_algos
            
            n_frames = args.n_frames

            # Initialize Environment
            print(f"Initialize AirSimEnv with n_frames={n_frames} for {algo_name} (seed={seed})...")
            env = AirSimEnv(takeoff_height=args.takeoff_height, config=args, stack_frames=n_frames)
            if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)

            # Initial Reset
            obs, _ = env.reset(seed=seed)
            # Obs is a dict keys: 'depth', 'base'
            
            depth_image = obs['depth']
            base_state = obs['base']

            # Dimensions
            base_dim = base_state.shape[0]
            depth_shape = depth_image.shape # (C, H, W)
            action_space = env.action_space

            # Recurrent algorithms consume env stacked frames as temporal sequence.
            # Most sequence models process each frame as single-channel (C=1)
            if is_recurrent:
                model_depth_shape = (1, depth_shape[-2], depth_shape[-1])
            else:
                model_depth_shape = depth_shape

            print(f"Observation shapes: Depth {depth_shape}, Base {base_dim}")
            print(f"Action space: {action_space}")

            # Initialize Agent
            device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
            AgentClass = get_agent_class(algo_name)
            
            # pass seed to agent so it can create its own RNG
            agent = AgentClass(base_dim, model_depth_shape, action_space, args, device=device, seed=seed)

            # Run training for this algorithm
            env = train_single_algorithm(env, agent, args, algo_name, is_recurrent, device, base_state, depth_image, n_frames)

            # Close AirSim after training this algorithm/seed
            if hasattr(env, 'game_handler') and env.game_handler is not None:
                print(f"Closing AirSim for {algo_name} (seed={seed})...")
                env.game_handler.kill_game_in_editor()
                time.sleep(2)  # Wait for shutdown
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"Memory cleaned for {algo_name} (seed={seed}).")


def train_single_algorithm(env, agent, args, algo_name, is_recurrent, device, base_state, depth_image, n_frames):

    if args.load_model != "":
        print(f"Loading model: {args.load_model}")
        agent.load(args.load_model)

    # 根据是否使用课程学习修改算法显示名称（用于日志和绘图）
    # algo_name 已经包含了 CL- 前缀（如果启用课程学习），直接使用即可
    display_algo_name = algo_name
    print(f"Start Asynchronous Training {display_algo_name}...")

    # Restart interval for refreshing UE4 memory
    restart_interval = 200000
    next_restart = restart_interval

    # Logging
    if not os.path.exists("./results"): os.makedirs("./results")
    if not os.path.exists("./models"): os.makedirs("./models")
    
    # 移除时间戳，固定文件夹名称以实现覆盖 (Remove timestamp to use fixed folder for overwriting)
    # 使用 display_algo_name（带 CL- 前缀）用于日志目录和文件名
    run_name = f"{display_algo_name}_seed{args.seed}"
    log_dir = f"./results/{run_name}"
    
    # 如果文件夹已存在，则清理内容实现真正覆盖 (Clean directory if exists for true overwrite)
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
        
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize CSV logger
    csv_filename = os.path.join(log_dir, f"{display_algo_name}_seed{args.seed}_log.csv")
    with open(csv_filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['episode', 'total_timesteps', 'reward', 'episode_length', 'success_rate'])
    
    print(f"Logging to {csv_filename}")

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

            if is_recurrent:
                depth_seq = state
                if depth_seq.ndim == 3:
                    depth_seq = np.expand_dims(depth_seq, axis=1)

            # Select Action
            if total_timesteps < start_timesteps and args.load_model == "":
                action = env.action_space.sample()
                # print(f"Random action at timestep {total_timesteps}: {action}")
            else:
                progress_ratio = total_timesteps / max_timesteps
                if is_recurrent:
                    action = agent.select_action(base, depth_seq, progress_ratio=progress_ratio)
                else:
                    # Non-recurrent: state is (4, H, W)
                    action = agent.select_action(base, state, progress_ratio=progress_ratio)

                _raise_if_non_finite(
                    "actor.action",
                    action,
                    f"algo={display_algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
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
                    obs, _ = env.reset(seed=args.seed + episode_num)
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
                f"algo={display_algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
            )
            
            next_state = next_obs['depth']
            next_base = next_obs['base']

            _raise_if_non_finite(
                "env.next_depth",
                next_state,
                f"algo={display_algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
            )
            _raise_if_non_finite(
                "env.next_base",
                next_base,
                f"algo={display_algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
            )

            if args.render_window:
                vis_imgs = []
                if is_recurrent:
                    recurrent_vis = depth_seq
                    if recurrent_vis.ndim == 4 and recurrent_vis.shape[1] == 1:
                        recurrent_vis = recurrent_vis[:, 0]
                    vis_imgs.extend(list(recurrent_vis))
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
                next_depth_seq = next_state
                if next_depth_seq.ndim == 3:
                    next_depth_seq = np.expand_dims(next_depth_seq, axis=1)

                if algo_name == 'ST-SVimTD3':
                    has_collided = float(step_info.get("has_collided", False)) if isinstance(step_info, dict) else 0.0
                    agent.replay_buffer.add(
                        base,
                        depth_seq,
                        action,
                        reward,
                        next_base,
                        next_depth_seq,
                        done_bool,
                        has_collided
                    )
                else:
                    agent.replay_buffer.add(
                        base,
                        depth_seq,
                        action,
                        reward,
                        next_base,
                        next_depth_seq,
                        done_bool
                    )
            else:
                # Non-recurrent buffer
                add_fn = getattr(agent.replay_buffer, "add", None)
                supports_success_flag = False
                if callable(add_fn):
                    try:
                        supports_success_flag = "is_success" in inspect.signature(add_fn).parameters
                    except (TypeError, ValueError):
                        supports_success_flag = False

                if supports_success_flag:
                    is_success = float(step_info.get("is_success", False)) if isinstance(step_info, dict) else 0.0
                    agent.replay_buffer.add(base, state, action, reward, next_base, next_state, done_bool, is_success)
                else:
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
                
                # Log to TensorBoard (use total_timesteps as x-axis)
                writer.add_scalar('train/episode_reward', episode_reward, total_timesteps)
                writer.add_scalar('train/episode_length', episode_timesteps, total_timesteps)
                writer.add_scalar('train/success_rate', success_rate, total_timesteps)

                # Log action distribution to TensorBoard (per episode)
                if len(action_hist) > 0:
                    action_arr = np.array(action_hist)
                    if action_arr.ndim == 1:
                        writer.add_histogram('train/action_distribution', action_arr, episode_num)
                    else:
                        # Record each action dimension separately - only keep histograms
                        action_dim_names = ['forward_speed', 'vertical_speed', 'yaw_rate'] if action_arr.shape[1] == 3 else [f'action_dim_{i}' for i in range(action_arr.shape[1])]
                        for dim, name in enumerate(action_dim_names):
                            writer.add_histogram(f'train/{name}', action_arr[:, dim], episode_num)
                
                # Log to CSV
                with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([episode_num, total_timesteps, episode_reward, episode_timesteps, success_rate])

                print(f"[{display_algo_name.upper()}] Episode {episode_num}, Reward: {episode_reward:.2f}, Length: {episode_timesteps}, Success Rate: {success_rate:.2f}, Level: {env.level}, Total Timesteps: {total_timesteps}, Total Successes: {env.success_count}")
                
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
                    env = AirSimEnv(takeoff_height=args.takeoff_height, config=args, stack_frames=n_frames)
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

                # Memory cleanup after episode end
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Training Update
        
        if agent.replay_buffer.size() >= args.batch_size and total_timesteps >= start_timesteps:
            # 计算实际的梯度更新次数：收集步数 * gradient_steps 倍数
            n_updates = int(steps_per_update * args.gradient_steps)
            n_updates = max(1, n_updates)  # 至少更新1次
            
            # Show progress bar for the update steps
            loss_info_list = []
            for _ in tqdm(range(n_updates), desc=f"Training ({total_timesteps})", leave=False):
                # Pass progress for schedulers if needed (conceptually)
                progress_ratio = total_timesteps / max_timesteps
                train_info = agent.train(progress_ratio=progress_ratio)
                if train_info:
                    for metric_name in ("actor_loss", "critic_loss"):
                        if metric_name in train_info:
                            _raise_if_non_finite(
                                f"train.{metric_name}",
                                train_info[metric_name],
                                f"algo={display_algo_name}, total_timesteps={total_timesteps}"
                            )
                    loss_info_list.append(train_info)
            
            # Log average loss every 100 training steps
            if loss_info_list and total_timesteps % 100 == 0:
                avg_actor_loss = sum(info.get('actor_loss', 0) for info in loss_info_list) / len(loss_info_list)
                avg_critic_loss = sum(info.get('critic_loss', 0) for info in loss_info_list) / len(loss_info_list)
                
                writer.add_scalar('loss/actor_loss', avg_actor_loss, total_timesteps)
                writer.add_scalar('loss/critic_loss', avg_critic_loss, total_timesteps)

                meta_metric_map = {
                    'adaptive/meta_weight_entropy': 'meta_weight_entropy',
                    'adaptive/meta_weight_max': 'meta_weight_max',
                    'adaptive/reg': 'adaptive_reg',
                }
                for tb_tag, key in meta_metric_map.items():
                    values = [info[key] for info in loss_info_list if key in info]
                    if values:
                        mean_value = float(np.mean(values))
                        _raise_if_non_finite(
                            f"train.{key}",
                            mean_value,
                            f"algo={display_algo_name}, total_timesteps={total_timesteps}"
                        )
                        writer.add_scalar(tb_tag, mean_value, total_timesteps)

                per_metric_map = {
                    'per/beta': 'per_beta',
                    'per/success_sample_ratio_target': 'replay/success_sample_ratio_target',
                    'per/success_batch_fraction': 'replay/success_batch_fraction',
                    'per/success_size': 'replay/success_size',
                    'per/regular_size': 'replay/regular_size',
                }
                for tb_tag, key in per_metric_map.items():
                    values = [info[key] for info in loss_info_list if key in info]
                    if values:
                        mean_value = float(np.mean(values))
                        _raise_if_non_finite(
                            f"train.{key}",
                            mean_value,
                            f"algo={display_algo_name}, total_timesteps={total_timesteps}"
                        )
                        writer.add_scalar(tb_tag, mean_value, total_timesteps)
            
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
