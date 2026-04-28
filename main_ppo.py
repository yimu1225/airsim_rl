#!/usr/bin/env python3
"""
On-Policy Training Main Script for AirSim RL
Supports PPO and other on-policy algorithms.
"""
import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import time
import random
import copy
import numpy as np
import torch
import csv
import cv2
import gc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from algorithm.config_loader import apply_algorithm_params
from algo_name_utils import (
    expand_algorithm_spec,
    is_curriculum_algorithm,
    split_curriculum_prefix,
    to_internal_algorithm_name,
    to_internal_core_algorithm_name,
)
import gymnasium as gym
import gym_airsim  # noqa: F401 - ensure env ids are registered
from gym_airsim.envs import AirSimEnv

# On-Policy Algorithm Imports
from algorithm.ppo.ppo import PPOAgent
from algorithm.ST_Vim_PPO.agent import STVimPPOAgent


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


def _to_scalar_float(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.item())
    return None


def _log_train_metrics_per_update(writer, train_info, update_step, algo_name, total_timesteps):
    if not isinstance(train_info, dict):
        return

    for key, value in train_info.items():
        scalar_value = _to_scalar_float(value)
        if scalar_value is None:
            continue

        _raise_if_non_finite(
            f"train.{key}",
            scalar_value,
            f"algo={algo_name}, update_step={update_step}, total_timesteps={total_timesteps}",
        )

        key_str = str(key)
        if "loss" in key_str.lower():
            writer.add_scalar(f"loss/{key_str}", scalar_value, total_timesteps)
        else:
            writer.add_scalar(f"update/{key_str}", scalar_value, total_timesteps)


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
    return expand_algorithm_spec(algo_str)


def get_agent_class(algo_name):
    """Get agent class by algorithm name."""
    core_algo_name = to_internal_core_algorithm_name(algo_name)
    
    agents = {
        'ppo': PPOAgent,
        'st_vim_ppo': STVimPPOAgent,
    }
    if core_algo_name in agents:
        return agents[core_algo_name]
    raise ValueError(f"Unknown on-policy algorithm: {algo_name}")


def create_env_from_name(args, n_frames):
    """
    Build environment from --env_name.
    Supports both registered Gym ids and local aliases.
    """
    env_name = str(getattr(args, "env_name", "AirSimEnv-v42")).strip()

    # Local aliases (instantiate directly to avoid wrapper surprises).
    env_aliases = {
        "AirSimEnv-v42": AirSimEnv,
    }
    env_cls = env_aliases.get(env_name)
    if env_cls is not None:
        return env_cls(takeoff_height=args.takeoff_height, config=args, stack_frames=n_frames)

    # Fallback: try Gym registry id.
    try:
        return gym.make(
            env_name,
            takeoff_height=args.takeoff_height,
            config=args,
            stack_frames=n_frames,
        )
    except Exception as e:
        supported_aliases = ", ".join(sorted(env_aliases.keys()))
        raise ValueError(
            f"Unsupported --env_name '{env_name}'. "
            f"Supported aliases: {supported_aliases}; or pass a valid gymnasium env id."
        ) from e


def _get_env_core(env):
    """
    Return the underlying custom env even when gym wrappers are present.
    """
    return env.unwrapped if hasattr(env, "unwrapped") else env


def train_ppo_algorithm(env, agent, args, algo_name, device, base_state, depth_image, n_frames):
    """
    Training loop for PPO algorithm.
    """
    if args.load_model != "":
        print(f"Loading model: {args.load_model}")
        agent.load(args.load_model)

    display_algo_name = algo_name
    print(f"Start PPO Training {display_algo_name}...")

    # Restart interval for refreshing UE4 memory
    restart_interval = 200000
    next_restart = restart_interval

    # Logging
    if not os.path.exists("./results"): 
        os.makedirs("./results")
    if not os.path.exists("./models"): 
        os.makedirs("./models")
    
    run_name = f"{display_algo_name}_seed{args.seed}"
    log_dir = f"./results/{run_name}"
    
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
    
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    update_step = 0

    state = depth_image
    base = base_state

    print("Start PPO Training Loop...")
    print(f"Rollout buffer size: {args.rollout_buffer_size}")
    print(f"PPO epochs: {args.ppo_epochs}, Batch size: {args.ppo_batch_size}")

    if args.render_window:
        cv2.namedWindow("Depth View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth View", 256, 256)

    while total_timesteps < max_timesteps:
        # Collect rollout data until buffer is full or episode ends
        rollout_started = False
        
        while not agent.rollout_buffer.is_full() and total_timesteps < max_timesteps:
            episode_timesteps += 1
            total_timesteps += 1
            rollout_started = True

            # Select Action (PPO returns action, value, log_prob)
            progress_ratio = total_timesteps / max_timesteps
            action, value, log_prob = agent.select_action(base, state, deterministic=False)

            _raise_if_non_finite(
                "actor.action",
                action,
                f"algo={display_algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
            )

            # Step environment
            try:
                next_obs, reward, terminated, truncated, step_info = env.step(action)
            except Exception as e:
                print(f"CRITICAL ERROR in env.step: {e}")
                print("Checking game status and attempting recovery...")
                
                if env.check_ue4_status(force_restart=True, reason="env_step_exception"):
                    obs, _ = env.reset(seed=args.seed)
                    next_obs = obs
                    reward = 0.0
                    terminated = True
                    truncated = False
                    step_info = {}
                else:
                    reward = 0.0
                    terminated = True
                    truncated = False
                    step_info = {}
                
            done = terminated or truncated

            _raise_if_non_finite("env.reward", reward, f"algo={display_algo_name}, total_timesteps={total_timesteps}")
            
            next_state = next_obs['depth']
            next_base = next_obs['base']

            _raise_if_non_finite("env.next_depth", next_state, f"algo={display_algo_name}, total_timesteps={total_timesteps}")
            _raise_if_non_finite("env.next_base", next_base, f"algo={display_algo_name}, total_timesteps={total_timesteps}")

            # Visualization
            if args.render_window:
                vis_imgs = []
                if len(next_state.shape) == 3:
                    vis_imgs.extend([next_state[i] for i in range(next_state.shape[0])])
                else:
                    vis_imgs.append(next_state)

                processed_imgs = []
                for img in vis_imgs:
                    if len(img.shape) == 3:
                        img = img.squeeze(0)
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    processed_imgs.append(img)
                
                if processed_imgs:
                    vis_concat = np.hstack(processed_imgs)
                    height, width = vis_concat.shape[:2]
                    cv2.resizeWindow("Depth View", width, height)
                    cv2.imshow("Depth View", vis_concat)
                    cv2.waitKey(1)

            episode_reward += reward

            # Store transition in rollout buffer
            done_bool = float(done)
            agent.store_transition(base, state, action, reward, value, log_prob, done_bool)

            # State update
            state = next_state
            base = next_base

            # Episode end handling
            if done:
                # Calculate success rate
                success_rate = 0.0
                env_core = _get_env_core(env)
                if len(env_core.success_deque) > 0:
                    success_rate = sum(env_core.success_deque) / len(env_core.success_deque)
                
                # Log to TensorBoard
                writer.add_scalar('train/episode_reward', episode_reward, total_timesteps)
                writer.add_scalar('train/episode_length', episode_timesteps, total_timesteps)
                writer.add_scalar('train/success_rate', success_rate, total_timesteps)

                # Log to CSV
                with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([episode_num, total_timesteps, episode_reward, episode_timesteps, success_rate])

                print(f"[{display_algo_name.upper()}] Episode {episode_num}, Reward: {episode_reward:.2f}, Length: {episode_timesteps}, Success Rate: {success_rate:.2f}, Level: {env_core.level}, Total Timesteps: {total_timesteps}, Total Successes: {env_core.success_count}")
                
                episode_num += 1
                episode_reward = 0
                episode_timesteps = 0

                # Check restart
                if total_timesteps >= next_restart:
                    print(f"Restarting game to refresh UE4 memory at total_timesteps {total_timesteps}...")
                    env_core = _get_env_core(env)
                    old_success_count = env_core.success_count
                    old_success_deque = env_core.success_deque
                    old_level = env_core.level
                    old_game_config_handler = env_core.game_config_handler
                    
                    if hasattr(env_core, 'game_handler') and env_core.game_handler is not None:
                        env_core.game_handler.kill_game_in_editor()
                        time.sleep(2)
                    
                    env = create_env_from_name(args, n_frames)
                    new_env_core = _get_env_core(env)
                    new_env_core.success_count = old_success_count
                    new_env_core.success_deque = old_success_deque
                    new_env_core.level = old_level
                    new_env_core.game_config_handler = old_game_config_handler
                    next_restart += restart_interval

                # Reset environment
                obs, _ = env.reset(seed=args.seed + episode_num)
                state = obs['depth']
                base = obs['base']

                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Training update when buffer is full
        if rollout_started and agent.rollout_buffer.size() > 0:
            # Finish trajectory with bootstrap value
            with torch.no_grad():
                base_tensor = torch.as_tensor(base, dtype=torch.float32, device=device).view(1, -1)
                depth_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                # Use agent's method to get last value
                last_state = agent.get_state_representation(base_tensor, depth_tensor)
                last_value = agent.critic(last_state).cpu().numpy().flatten()[0]
            
            # Compute returns and advantages (stored in buffer)
            agent.rollout_buffer.compute_returns_and_advantages(last_value, 0.0)
            
            # Update policy with progress bar for epochs
            from tqdm import tqdm
            epoch_pbar = tqdm(range(args.ppo_epochs), desc=f"Training ({total_timesteps})", leave=False)
            train_info = agent.update_policy(epoch_pbar=epoch_pbar)
            epoch_pbar.close()
            
            if train_info:
                update_step += 1
                _log_train_metrics_per_update(
                    writer=writer,
                    train_info=train_info,
                    update_step=update_step,
                    algo_name=display_algo_name,
                    total_timesteps=total_timesteps,
                )
                print(f"  Entropy: {train_info.get('entropy', 0):.4f}, "
                      f"Approx KL: {train_info.get('approx_kl', 0):.4f}")
            
            # Memory cleanup after training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Checkpointing
        if total_timesteps % 100000 == 0:
            agent.save(f"./models/{algo_name}_async_{total_timesteps}.pth")
            print(f"Model saved at timestep {total_timesteps}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Final save
    agent.save(f"./models/{algo_name}_async_final.pth")
    print("Training completed.")
    
    if args.render_window:
        cv2.destroyAllWindows()
        
    if hasattr(env, 'close'):
        env.close()
    writer.close()

    return env


def main():
    base_args = get_config()
    seeds = base_args.seed if isinstance(base_args.seed, (list, tuple)) else [base_args.seed]

    # Expand algorithm names
    algorithms = expand_algorithms(base_args.algorithm_name)
    print(f"Training on-policy algorithms: {algorithms}")

    for seed in seeds:
        seed_args = copy.deepcopy(base_args)
        seed_args.seed = seed
        _configure_reproducibility(seed, seed_args)

        for algo_name in algorithms:
            algo_name = to_internal_algorithm_name(algo_name)
            core_algo_name = to_internal_core_algorithm_name(algo_name)
            if core_algo_name not in {"ppo", "st_vim_ppo"}:
                print(f"Skipping unsupported on-policy algorithm in main_ppo.py: {algo_name}")
                continue

            args = copy.deepcopy(seed_args)
            args.algorithm_name = algo_name
            params_path, loaded_keys = apply_algorithm_params(args, algo_name)
            print(f"\n{'='*50}")
            print(f"Training algorithm: {algo_name} (seed={seed})")
            print(f"{'='*50}")
            if loaded_keys:
                print(f"  [Algo Params] Loaded {len(loaded_keys)} params from {params_path}")
            else:
                print(f"  [Algo Params] Loaded empty params from {params_path}")

            # Handle curriculum learning prefix
            if is_curriculum_algorithm(algo_name):
                actual_algo_name = split_curriculum_prefix(algo_name)[1]
                print(f"  [Curriculum Learning Enabled] {actual_algo_name}")
            else:
                actual_algo_name = algo_name
                print(f"  [Curriculum Learning Disabled] {algo_name}")

            # ST-Vim-PPO consumes the stacked depth frames as a temporal sequence.
            is_recurrent = core_algo_name == "st_vim_ppo"
            n_frames = args.n_frames

            # Initialize Environment
            print(
                f"Initialize env '{args.env_name}' with n_frames={n_frames} "
                f"for {algo_name} (seed={seed})..."
            )
            env = create_env_from_name(args, n_frames)
            if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)

            # Initial Reset
            obs, _ = env.reset(seed=seed)
            
            depth_image = obs['depth']
            base_state = obs['base']

            # Dimensions
            base_dim = base_state.shape[0]
            depth_shape = depth_image.shape
            model_depth_shape = (1, depth_shape[-2], depth_shape[-1]) if is_recurrent else depth_shape
            if is_recurrent:
                args.depth_shape = model_depth_shape
            action_space = env.action_space

            print(f"Observation shapes: Depth {depth_shape}, Base {base_dim}")
            print(f"Action space: {action_space}")

            # Initialize Agent
            device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
            AgentClass = get_agent_class(algo_name)
            
            agent = AgentClass(base_dim, model_depth_shape, action_space, args, device=device, seed=seed)

            # Run training
            env = train_ppo_algorithm(env, agent, args, algo_name, device, base_state, depth_image, n_frames)

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


if __name__ == "__main__":
    main()
