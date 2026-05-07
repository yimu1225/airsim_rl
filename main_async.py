#!/usr/bin/env python3
"""
Asynchronous Training Main Script for AirSim RL
Supports multiple algorithms compatible with UAV_Navigation(RLlib) project.
"""
import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  


# 设置环境变量，获得更详细的错误信息

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

import time
import random
import copy
import collections
import numpy as np
import torch
import csv
import cv2  # Added for visualization
import gc  # Added for memory management
import inspect
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

# Algorithm Imports
from algorithm.TD3.td3 import TD3Agent
from algorithm.DDPG.ddpg import DDPGAgent
from algorithm.PER_TD3.per_td3 import PERTD3Agent

from algorithm.ST_Vim_TD3.agent import STVimTD3Agent
from algorithm.STV_Patch_TD3.agent import VimPatchTD3Agent
from algorithm.Vim_TD3.agent import VimTD3Agent
from algorithm.ST_Seq_Vim_TD3.agent import StateSeqVimTD3Agent
from algorithm.STV_Seq_Vim_TD3.agent import VimStateSeqTD3Agent
from algorithm.PER_ST_Vim_TD3.agent import PERVimTD3Agent
from algorithm.ST_SVim_TD3.agent import STSVimTD3Agent
from algorithm.Mamba_TD3.agent import MambaTD3Agent
from algorithm.ST_DualVim_TD3.agent import DualBranchVideoMambaTD3Agent
from algorithm.SAC.agent import SACAgent
from algorithm.PL_SAC.agent import PLSACAgent
from algorithm.PL_PER_ST_Vim_SAC.agent import PLPERSTVimSACAgent
from algorithm.PL_PER_ST_Vim_TD3.agent import PLPERSTVimTD3Agent
from algorithm.LSTM_SAC.agent import LSTMSACAgent
from algorithm.ST_Vim_SAC.agent import STVimSACAgent
from algorithm.PER_ST_Vim_SAC.agent import PERSTVimSACAgent
from algorithm.PL_TD3.pl_td3 import PLTD3Agent
from algorithm.PL_PER_TD3.pl_per_td3 import PLPERTD3Agent
from algorithm.PL_ST_Vim_TD3.agent import PLSTVimTD3Agent
from algorithm.AETD3.aetd3 import AETD3Agent



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

    compatibility_tags = {
        "meta_weight_entropy": "adaptive/meta_weight_entropy",
        "meta_weight_max": "adaptive/meta_weight_max",
        "adaptive_reg": "adaptive/reg",
        "per_beta": "per/beta",
        "replay/success_sample_ratio_target": "per/success_sample_ratio_target",
        "replay/success_batch_fraction": "per/success_batch_fraction",
        "replay/success_size": "per/success_size",
        "replay/regular_size": "per/regular_size",
    }

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

        if key_str in compatibility_tags:
            writer.add_scalar(compatibility_tags[key_str], scalar_value, total_timesteps)


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
    core_algo_name = to_internal_core_algorithm_name(algo_name)
    
    agents = {
        'TD3': TD3Agent,
        'PL_TD3': PLTD3Agent,
        'DDPG': DDPGAgent,
        'PER_TD3': PERTD3Agent,
        'PL_PER_TD3': PLPERTD3Agent,
        'PL_PER_ST_Vim_TD3': PLPERSTVimTD3Agent,
        'ST_Vim_TD3': STVimTD3Agent,
        'STV_Patch_TD3': VimPatchTD3Agent,
        'Vim_TD3': VimTD3Agent,
        'ST_Seq_Vim_TD3': StateSeqVimTD3Agent,
        'STV_Seq_Vim_TD3': VimStateSeqTD3Agent,
        'PER_ST_Vim_TD3': PERVimTD3Agent,
        'PL_ST_Vim_TD3': PLSTVimTD3Agent,
        'ST_SVim_TD3': STSVimTD3Agent,
        'Mamba_TD3': MambaTD3Agent,
        'ST_DualVim_TD3': DualBranchVideoMambaTD3Agent,
        'AETD3': AETD3Agent,
        'SAC': SACAgent,
        'PL_SAC': PLSACAgent,
        'PL_PER_ST_Vim_SAC': PLPERSTVimSACAgent,
        'LSTM_SAC': LSTMSACAgent,
        'ST_Vim_SAC': STVimSACAgent,
        'PER_ST_Vim_SAC': PERSTVimSACAgent,
    }
    if core_algo_name in agents:
        return agents[core_algo_name]
    raise ValueError(f"Unknown algorithm: {algo_name}")


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


def _pause_env_simulation(env):
    """
    Keep UE/AirSim time stopped while the learner is updating from replay.
    """
    env_core = _get_env_core(env)
    try:
        airgym = getattr(env_core, "airgym", None)
        client = getattr(airgym, "client", None)
        if client is not None:
            client.simPause(True)
    except Exception as exc:
        print(f"WARNING: failed to pause simulator before training update: {exc}")


def _is_pl_algorithm(algo_name: str) -> bool:
    core_name = to_internal_core_algorithm_name(algo_name)
    return core_name in {
        "PL_TD3", "PL_PER_TD3", "PL_ST_Vim_TD3", "PL_SAC",
        "PL_PER_ST_Vim_SAC", "PL_PER_ST_Vim_TD3", "PL_ST_Vim_PPO",
    }


def _extract_last_depth_frame(depth_tensor):
    arr = np.asarray(depth_tensor, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[-1, 0]
    if arr.ndim == 3:
        return arr[-1]
    if arr.ndim == 2:
        return arr
    # 极端fallback：使用极小尺寸，实际分辨率由正常数据决定
    return np.zeros((1, 1), dtype=np.float32)


def _resize_depth_frame(frame, target_hw):
    """不再强制resize，直接使用原始分辨率（仅做维度检查和clip）"""
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 2:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        return np.zeros((target_h, target_w), dtype=np.float32)
    return np.clip(arr, 0.0, 255.0).astype(np.float32)


def _extract_clean_depth_sequence(env_core, fallback_depth):
    fallback_frame = _extract_last_depth_frame(fallback_depth)

    for attr in ("clean_depth_stack", "depth_stack_clean"):
        stack = getattr(env_core, attr, None)
        if stack is None or len(stack) <= 0:
            continue
        arr = np.asarray(stack, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] > 0:
            return arr

    airgym = getattr(env_core, "airgym", None)

    raw_clean = getattr(airgym, "last_depth_clean", None)
    if raw_clean is not None:
        arr = np.asarray(raw_clean, dtype=np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        if arr.ndim == 3 and arr.shape[0] > 0:
            return arr

    raw = getattr(airgym, "last_img", None)
    if raw is not None:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        if arr.ndim == 3 and arr.shape[0] > 0:
            return arr

    return fallback_frame[None, ...]


def _align_clean_depth_sequence(clean_seq, target_len, target_hw):
    clean = np.asarray(clean_seq, dtype=np.float32)
    if clean.ndim == 2:
        clean = clean[None, ...]
    if clean.ndim != 3 or clean.shape[0] <= 0:
        frame = np.zeros((int(target_hw[0]), int(target_hw[1])), dtype=np.float32)
        clean = frame[None, ...]

    # 不再强制resize每一帧，直接使用原始分辨率做序列长度对齐
    if clean.shape[0] >= target_len:
        return clean[-target_len:]

    pad_count = target_len - clean.shape[0]
    pad = np.repeat(clean[:1], pad_count, axis=0)
    return np.concatenate([pad, clean], axis=0).astype(np.float32)


def _build_critic_depth_like(env_core, actor_depth):
    actor_arr = np.asarray(actor_depth, dtype=np.float32)
    target_hw = actor_arr.shape[-2:] if actor_arr.ndim >= 2 else getattr(env_core, "depth_shape", (1, 128, 128))[-2:]
    clean_seq = _extract_clean_depth_sequence(env_core, actor_arr)

    if actor_arr.ndim == 2:
        return _align_clean_depth_sequence(clean_seq, target_len=1, target_hw=target_hw)[-1]
    if actor_arr.ndim == 3:
        return _align_clean_depth_sequence(
            clean_seq,
            target_len=max(1, int(actor_arr.shape[0])),
            target_hw=target_hw,
        ).astype(np.float32)
    if actor_arr.ndim == 4 and actor_arr.shape[1] == 1:
        aligned = _align_clean_depth_sequence(
            clean_seq,
            target_len=max(1, int(actor_arr.shape[0])),
            target_hw=target_hw,
        )
        return aligned[:, None, ...].astype(np.float32)
    return actor_arr


def _extract_critic_privileged_distance_sensor(env_core):
    scan = getattr(env_core, "last_distance_sensor_scan_distance", None)
    if scan is None:
        cfg = getattr(env_core, "config", None)
        count = int(getattr(env_core, "distance_sensor_count", getattr(cfg, "distance_sensor_count", 36)))
        max_distance = getattr(env_core, "last_distance_sensor_max_distance", None)
        if max_distance is not None:
            max_distance = np.asarray(max_distance, dtype=np.float32).reshape(-1)
            if max_distance.size == count:
                return max_distance.copy()
        return np.ones((count,), dtype=np.float32)

    arr = np.asarray(scan, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)

def main():
    base_args = get_config()
    seeds = base_args.seed if isinstance(base_args.seed, (list, tuple)) else [base_args.seed]

    # Expand algorithm names
    algorithms = expand_algorithms(base_args.algorithm_name)
    print(f"Training algorithms: {algorithms}")

    for seed in seeds:
        seed_args = copy.deepcopy(base_args)
        seed_args.seed = seed
        _configure_reproducibility(seed, seed_args)

        # Run training for each algorithm
        for algo_name in algorithms:
            algo_name = to_internal_algorithm_name(algo_name)
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

            # 根据算法名判断是否使用课程学习
            # 算法名以 "CL-" 开头时启用课程学习
            if is_curriculum_algorithm(algo_name):
                actual_algo_name = split_curriculum_prefix(algo_name)[1]
                print(f"  [Curriculum Learning Enabled] {actual_algo_name}")
            else:
                actual_algo_name = algo_name
                print(f"  [Curriculum Learning Disabled] {algo_name}")

            # Determine properties for this algorithm
            recurrent_algos = {
                'Mamba_TD3',
                'ST_Vim_TD3',
                'STV_Patch_TD3',
                'Vim_TD3',
                'ST_Seq_Vim_TD3',
                'STV_Seq_Vim_TD3',
                'PER_ST_Vim_TD3',
                'PL_ST_Vim_TD3',
                'PL_PER_ST_Vim_SAC',
                'PL_PER_ST_Vim_TD3',
                'ST_SVim_TD3',
                'ST_DualVim_TD3',
                'LSTM_SAC',
                'ST_Vim_SAC',
                'PER_ST_Vim_SAC',
            }
            
            is_recurrent = actual_algo_name in recurrent_algos
            
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

            if _is_pl_algorithm(algo_name):
                env_core = _get_env_core(env)
                inferred_priv = _extract_critic_privileged_distance_sensor(env_core)
                setattr(args, "critic_priv_dim", int(np.asarray(inferred_priv).reshape(-1).shape[0]))
            
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
    core_algo_name = to_internal_core_algorithm_name(algo_name)
    is_pl_algo = _is_pl_algorithm(algo_name)
    print(f"Start Asynchronous Training {display_algo_name}...")

    # Restart interval for refreshing UE4 memory
    restart_interval = 200000
    next_restart = restart_interval

    # Logging
    if not os.path.exists("./results"): os.makedirs("./results")
    if not os.path.exists("./models"): os.makedirs("./models")

    # Save checkpoints by algorithm and seed: ./models/<algorithm>/seed<seed>/
    model_dir = os.path.join("./models", display_algo_name, f"seed{args.seed}")
    os.makedirs(model_dir, exist_ok=True)
    
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
    print(f"Model checkpoints will be saved to {model_dir}")

    # Training parameters
    max_timesteps = args.max_timesteps
    steps_per_update = args.steps_per_update
    start_timesteps = args.learning_starts

    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    update_step = 0

    state = depth_image
    base = base_state
    base_seq_algos = {"ST_Seq_Vim_TD3", "STV_Seq_Vim_TD3", "LSTM_SAC"}
    use_base_sequence = bool(is_recurrent and core_algo_name in base_seq_algos)
    base_seq_deque = None
    base_seq = None
    if use_base_sequence:
        base_seq_deque = collections.deque(maxlen=n_frames)
        init_base = np.asarray(base, dtype=np.float32)
        for _ in range(n_frames):
            base_seq_deque.append(init_base.copy())
        base_seq = np.stack(list(base_seq_deque), axis=0).astype(np.float32)

    depth_view_scale = max(float(getattr(args, "depth_view_scale", 2.5)), 1.0)
           

    print("Start Asynchronous Training Loop...")

    if args.render_window:
        cv2.namedWindow("Depth View", cv2.WINDOW_NORMAL)
        init_side = int(256 * depth_view_scale)
        cv2.resizeWindow("Depth View", init_side, init_side)

    while total_timesteps < max_timesteps:
        # Collect steps_per_update
        for step in range(steps_per_update):
            episode_timesteps += 1
            total_timesteps += 1

            if is_recurrent:
                depth_seq = state
                if depth_seq.ndim == 3:
                    depth_seq = np.expand_dims(depth_seq, axis=1)
                actor_depth_current = depth_seq
                actor_base_current = base_seq if use_base_sequence else base
            else:
                actor_depth_current = state
                actor_base_current = base

            critic_depth_current = None
            critic_priv_current = None
            if is_pl_algo:
                env_core = _get_env_core(env)
                critic_depth_current = _build_critic_depth_like(env_core, actor_depth_current)
                critic_priv_current = _extract_critic_privileged_distance_sensor(env_core)

            env_core_for_signal = _get_env_core(env)
            success_rate_signal = 0.0
            if len(env_core_for_signal.success_deque) > 0:
                success_rate_signal = sum(env_core_for_signal.success_deque) / len(env_core_for_signal.success_deque)

            # Select Action
            if total_timesteps < start_timesteps and args.load_model == "":
                action = env.action_space.sample()
                # print(f"Random action at timestep {total_timesteps}: {action}")
            else:
                if is_recurrent:
                    action = agent.select_action(actor_base_current, depth_seq, progress_ratio=success_rate_signal)
                else:
                    # Non-recurrent: state is (4, H, W)
                    action = agent.select_action(base, state, progress_ratio=success_rate_signal)

                _raise_if_non_finite(
                    "actor.action",
                    action,
                    f"algo={display_algo_name}, total_timesteps={total_timesteps}, episode={episode_num}, episode_step={episode_timesteps}"
                )

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
                    render_depth = next_state
                    # 显示所有堆叠帧 (Show all stacked frames)
                    if len(render_depth.shape) == 3:
                        vis_imgs.extend([render_depth[i] for i in range(render_depth.shape[0])])
                    else:
                        vis_imgs.append(render_depth)

                # 处理图像列表 (Process image list)
                processed_imgs = []
                for img in vis_imgs:
                    # 确保是 (H, W) 格式 (Ensure (H, W) format)
                    if len(img.shape) == 3:
                        img = img.squeeze(0) # Assume (1, H, W) -> (H, W)
                    
                    # 确保 uint8 类型 (Ensure uint8)
                    if img.dtype != np.uint8:
                        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
                    processed_imgs.append(img)
                
                # 水平拼接 (Horizontal concatenation)
                if processed_imgs:
                    vis_concat = np.hstack(processed_imgs)
                    if depth_view_scale != 1.0:
                        vis_show = cv2.resize(
                            vis_concat,
                            None,
                            fx=depth_view_scale,
                            fy=depth_view_scale,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        vis_show = vis_concat
                    # 动态调整窗口大小 (Dynamically resize window)
                    height, width = vis_show.shape[:2]
                    cv2.resizeWindow("Depth View", width, height)
                    cv2.imshow("Depth View", vis_show)
                    cv2.waitKey(1)

            episode_reward += reward

            # Add to Buffer
            done_bool = float(done)

            if is_recurrent:
                next_depth_seq = next_state
                if next_depth_seq.ndim == 3:
                    next_depth_seq = np.expand_dims(next_depth_seq, axis=1)
                actor_depth_next = next_depth_seq
            else:
                actor_depth_next = next_state

            next_base_seq = None
            next_base_seq_deque = None
            if use_base_sequence:
                next_base_seq_deque = collections.deque(base_seq_deque, maxlen=n_frames)
                next_base_seq_deque.append(np.asarray(next_base, dtype=np.float32))
                next_base_seq = np.stack(list(next_base_seq_deque), axis=0).astype(np.float32)

            base_for_buffer = base_seq if use_base_sequence else base
            next_base_for_buffer = next_base_seq if use_base_sequence else next_base

            critic_depth_next = None
            critic_priv_next = None
            if is_pl_algo:
                env_core = _get_env_core(env)
                critic_depth_next = _build_critic_depth_like(env_core, actor_depth_next)
                critic_priv_next = _extract_critic_privileged_distance_sensor(env_core)
            
            if is_recurrent:
                if core_algo_name == 'ST_SVim_TD3':
                    has_collided = float(step_info.get("has_collided", False)) if isinstance(step_info, dict) else 0.0
                    agent.replay_buffer.add(
                        base_for_buffer,
                        depth_seq,
                        action,
                        reward,
                        next_base_for_buffer,
                        next_depth_seq,
                        done_bool,
                        has_collided
                    )
                elif is_pl_algo:
                    add_kwargs = dict(
                        critic_priv=critic_priv_current,
                        next_critic_priv=critic_priv_next,
                        critic_depth=critic_depth_current,
                        next_critic_depth=critic_depth_next,
                    )
                    if core_algo_name in {"PL_PER_ST_Vim_SAC", "PL_PER_ST_Vim_TD3"}:
                        add_kwargs["is_success"] = (
                            float(step_info.get("is_success", False)) if isinstance(step_info, dict) else 0.0
                        )
                    agent.replay_buffer.add(
                        base_for_buffer,
                        depth_seq,
                        action,
                        reward,
                        next_base_for_buffer,
                        next_depth_seq,
                        done_bool,
                        **add_kwargs,
                    )
                else:
                    add_fn = getattr(agent.replay_buffer, "add", None)
                    supports_success_flag = False
                    if callable(add_fn):
                        try:
                            supports_success_flag = "is_success" in inspect.signature(add_fn).parameters
                        except (TypeError, ValueError):
                            supports_success_flag = False

                    if supports_success_flag:
                        is_success = float(step_info.get("is_success", False)) if isinstance(step_info, dict) else 0.0
                        agent.replay_buffer.add(
                            base_for_buffer,
                            depth_seq,
                            action,
                            reward,
                            next_base_for_buffer,
                            next_depth_seq,
                            done_bool,
                            is_success,
                        )
                    else:
                        agent.replay_buffer.add(
                            base_for_buffer,
                            depth_seq,
                            action,
                            reward,
                            next_base_for_buffer,
                            next_depth_seq,
                            done_bool
                        )
            else:
                if is_pl_algo:
                    if core_algo_name == "PL_PER_TD3":
                        is_success = float(step_info.get("is_success", False)) if isinstance(step_info, dict) else 0.0
                        agent.replay_buffer.add(
                            base,
                            state,
                            action,
                            reward,
                            next_base,
                            next_state,
                            done_bool,
                            is_success,
                            critic_priv=critic_priv_current,
                            next_critic_priv=critic_priv_next,
                            critic_depth=critic_depth_current,
                            next_critic_depth=critic_depth_next,
                        )
                    else:
                        agent.replay_buffer.add(
                            base,
                            state,
                            action,
                            reward,
                            next_base,
                            next_state,
                            done_bool,
                            critic_priv=critic_priv_current,
                            next_critic_priv=critic_priv_next,
                            critic_depth=critic_depth_current,
                            next_critic_depth=critic_depth_next,
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
            if use_base_sequence:
                base_seq_deque = next_base_seq_deque
                base_seq = next_base_seq

            # Episode End Handling
            if done:
                # Calculate Success Rate based on recent history
                # env.success_deque is populated in env.step()
                success_rate = 0.0
                env_core = _get_env_core(env)
                if len(env_core.success_deque) > 0:
                    success_rate = sum(env_core.success_deque) / len(env_core.success_deque)
                
                # Log to TensorBoard (use total_timesteps as x-axis)
                writer.add_scalar('train/episode_reward', episode_reward, total_timesteps)
                writer.add_scalar('train/episode_length', episode_timesteps, total_timesteps)
                writer.add_scalar('train/success_rate', success_rate, total_timesteps)

                # Log to CSV
                with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([episode_num, total_timesteps, episode_reward, episode_timesteps, success_rate])

                print(f"[{display_algo_name}] Episode {episode_num}, Reward: {episode_reward:.2f}, Length: {episode_timesteps}, Success Rate: {success_rate:.3f}, Level: {env_core.level}, Total Timesteps: {total_timesteps}, Total Successes: {env_core.success_count}")
                
                # Periodic CUDA memory cleanup
                if episode_num % 50 == 0:
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'memory_summary'):
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"[Memory] CUDA cache cleared. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
                episode_num += 1
                episode_reward = 0
                episode_timesteps = 0

                # Check if need to restart game
                if total_timesteps >= next_restart:
                    print(f"Restarting game to refresh UE4 memory at total_timesteps {total_timesteps}...")
                    # Save success stats and level before restart
                    env_core = _get_env_core(env)
                    old_success_count = env_core.success_count
                    old_success_deque = env_core.success_deque
                    old_level = env_core.level
                    old_game_config_handler = env_core.game_config_handler
                    # Close current environment
                    if hasattr(env_core, 'game_handler') and env_core.game_handler is not None:
                        env_core.game_handler.kill_game_in_editor()
                        time.sleep(2)  # Wait for kill
                    # Reinitialize environment
                    env = create_env_from_name(args, n_frames)
                    # Restore success stats and level
                    new_env_core = _get_env_core(env)
                    new_env_core.success_count = old_success_count
                    new_env_core.success_deque = old_success_deque
                    new_env_core.level = old_level
                    new_env_core.game_config_handler = old_game_config_handler
                    next_restart += restart_interval

                # Reset
                try:
                    obs, _ = env.reset(seed=args.seed + episode_num)
                except Exception as e:
                    print(f"CRITICAL ERROR in env.reset: {e}")
                    print("Checking game status and attempting recovery...")
                    env_core = _get_env_core(env)
                    if env_core.check_ue4_status(force_restart=True, reason="env_reset_exception"):
                        obs, _ = env.reset(seed=args.seed + episode_num)
                    else:
                        raise
                state = obs['depth']
                base = obs['base']
                if use_base_sequence:
                    base_seq_deque = collections.deque(maxlen=n_frames)
                    reset_base = np.asarray(base, dtype=np.float32)
                    for _ in range(n_frames):
                        base_seq_deque.append(reset_base.copy())
                    base_seq = np.stack(list(base_seq_deque), axis=0).astype(np.float32)

                # Memory cleanup after episode end
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Training Update
        
        if agent.replay_buffer.size() >= args.batch_size and total_timesteps >= start_timesteps:
            _pause_env_simulation(env)

            # 计算实际的梯度更新次数：收集步数 * gradient_steps 倍数
            n_updates = int(steps_per_update * args.gradient_steps)
            n_updates = max(1, n_updates)  # 至少更新1次
            
            # Show progress bar for the update steps
            for _ in tqdm(range(n_updates), desc=f"Training ({total_timesteps})", leave=False):
                # Pass progress for schedulers if needed (conceptually)
                progress_ratio = total_timesteps / max_timesteps
                train_info = agent.train(progress_ratio=progress_ratio)
                if train_info:
                    update_step += 1
                    _log_train_metrics_per_update(
                        writer=writer,
                        train_info=train_info,
                        update_step=update_step,
                        algo_name=display_algo_name,
                        total_timesteps=total_timesteps,
                    )
            
            # Memory cleanup after training updates
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Checkpointing
        if total_timesteps % 10000 == 0:
            model_path = os.path.join(model_dir, f"async_{total_timesteps}.pth")
            agent.save(model_path)
            print(f"Model saved at timestep {total_timesteps}: {model_path}")
        
        # Periodic CUDA memory cleanup every 5000 steps
        if total_timesteps % 5000 == 0:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_summary') and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[Memory] Step {total_timesteps}: CUDA cache cleared. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            # Memory cleanup after checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    final_model_path = os.path.join(model_dir, "async_final.pth")
    agent.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")
    
    if args.render_window:
        cv2.destroyAllWindows()
        
    if hasattr(env, 'close'):
        env.close()
    writer.close()

    return env

if __name__ == "__main__":
    main()
