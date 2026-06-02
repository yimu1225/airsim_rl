#!/usr/bin/env python3
"""Independent training entry point for Mamba-RSAC and PL-Mamba-RSAC."""

import copy
import csv
import gc
import os
import random
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import gym_airsim  # noqa: F401
from algo_name_utils import to_internal_core_algorithm_name, to_internal_algorithm_name
from algorithm.config_loader import apply_algorithm_params
from algorithm.Mamba_RSAC.agent import MambaRSACAgent
from algorithm.PL_Mamba_RSAC.agent import PLMambaRSACAgent
from config import get_config
from gym_airsim.envs import AirSimEnv


AGENTS = {
    "Mamba_RSAC": MambaRSACAgent,
    "PL_Mamba_RSAC": PLMambaRSACAgent,
}


def _raise_if_non_finite(name, value, step_info=""):
    arr = np.asarray(value)
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        non_finite = arr.size - int(finite_mask.sum())
        message = f"[NaNMonitor][{name}] non-finite detected: {non_finite}/{arr.size} elements"
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


def _log_train_metrics(writer, train_info, total_timesteps):
    if not isinstance(train_info, dict):
        return
    for key, value in train_info.items():
        scalar_value = _to_scalar_float(value)
        if scalar_value is None:
            continue
        _raise_if_non_finite(f"train.{key}", scalar_value, f"total_timesteps={total_timesteps}")
        tag_group = "loss" if "loss" in str(key).lower() else "update"
        writer.add_scalar(f"{tag_group}/{key}", scalar_value, total_timesteps)


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
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _get_env_core(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def create_env_from_name(args):
    env_name = str(getattr(args, "env_name", "AirSimEnv-v42")).strip()
    env_aliases = {"AirSimEnv-v42": AirSimEnv}
    env_cls = env_aliases.get(env_name)
    if env_cls is not None:
        return env_cls(takeoff_height=args.takeoff_height, config=args, stack_frames=1)

    try:
        return gym.make(env_name, takeoff_height=args.takeoff_height, config=args, stack_frames=1)
    except Exception as exc:
        supported_aliases = ", ".join(sorted(env_aliases))
        raise ValueError(
            f"Unsupported --env_name '{env_name}'. Supported aliases: {supported_aliases}; "
            "or pass a valid gymnasium env id."
        ) from exc


def _set_env_curriculum_progress(env, total_timesteps, max_timesteps):
    env_core = _get_env_core(env)
    if not bool(getattr(env_core, "use_curriculum", False)):
        return None
    if getattr(env_core, "curriculum_mode", "progress") != "progress":
        return None
    set_progress = getattr(env_core, "set_curriculum_progress", None)
    if not callable(set_progress):
        return None
    progress_ratio = min(max(float(total_timesteps) / max(float(max_timesteps), 1.0), 0.0), 1.0)
    return set_progress(progress_ratio)


def _pause_env_simulation(env):
    env_core = _get_env_core(env)
    try:
        airgym = getattr(env_core, "airgym", None)
        client = getattr(airgym, "client", None)
        if client is not None:
            client.simPause(True)
    except Exception as exc:
        print(f"WARNING: failed to pause simulator before training update: {exc}")


def _render_depth_window(depth, scale):
    vis = np.asarray(depth, dtype=np.float32)
    if vis.ndim == 3:
        vis = vis[0]
    vis = np.clip(vis, 0.0, 255.0).astype(np.uint8)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.resizeWindow("Depth View", vis.shape[1], vis.shape[0])
    cv2.imshow("Depth View", vis)
    cv2.waitKey(1)


def _algorithm_from_args(args):
    algo_name = to_internal_algorithm_name(args.algorithm_name)
    core_name = to_internal_core_algorithm_name(algo_name)
    if core_name not in AGENTS:
        raise ValueError(
            "main_mamba_rsac.py only supports Mamba_RSAC and PL_Mamba_RSAC. "
            f"Got {args.algorithm_name!r}."
        )
    return algo_name, core_name


def _seed_list(seed):
    return seed if isinstance(seed, (list, tuple)) else [seed]


def train_one(seed_args, algo_name, core_name):
    _configure_reproducibility(int(seed_args.seed), seed_args)
    params_path, loaded_keys = apply_algorithm_params(seed_args, core_name)
    seed_args.algorithm_name = algo_name

    print(f"\n{'=' * 50}")
    print(f"Training algorithm: {algo_name} (seed={seed_args.seed})")
    print(f"{'=' * 50}")
    print(f"  [Algo Params] Loaded {len(loaded_keys)} params from {params_path}")

    env = create_env_from_name(seed_args)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(int(seed_args.seed))

    _set_env_curriculum_progress(env, 0, seed_args.max_timesteps)
    obs, _ = env.reset(seed=int(seed_args.seed))
    depth = obs["depth"]
    base = obs["base"]
    base_dim = int(base.shape[0])
    depth_shape = tuple(depth.shape)
    action_space = env.action_space
    device = torch.device("cuda" if seed_args.cuda and torch.cuda.is_available() else "cpu")

    AgentClass = AGENTS[core_name]
    agent = AgentClass(base_dim, depth_shape, action_space, seed_args, device=device, seed=int(seed_args.seed))
    if seed_args.load_model != "":
        print(f"Loading model: {seed_args.load_model}")
        agent.load(seed_args.load_model)

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    model_dir = os.path.join("./models", algo_name, f"seed{seed_args.seed}")
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join("./results", f"{algo_name}_seed{seed_args.seed}")
    if os.path.exists(log_dir):
        import shutil

        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    csv_filename = os.path.join(log_dir, f"{algo_name}_seed{seed_args.seed}_log.csv")
    with open(csv_filename, mode="w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_timesteps", "reward", "episode_length", "success_rate"])

    print(f"Observation shapes: Depth {depth_shape}, Base {base_dim}")
    print(f"Action space: {action_space}")
    print(f"Logging to {csv_filename}")
    print(f"Model checkpoints will be saved to {model_dir}")

    total_timesteps = 0
    episode_num = 0
    episode_reward = 0.0
    episode_timesteps = 0
    update_step = 0
    max_timesteps = int(seed_args.max_timesteps)
    steps_per_update = int(seed_args.steps_per_update)
    start_timesteps = int(seed_args.learning_starts)
    update_after = int(seed_args.update_after)
    render_scale = max(float(getattr(seed_args, "depth_view_scale", 2.5)), 1.0)
    is_pl = core_name == "PL_Mamba_RSAC"

    agent.reset_history()
    if seed_args.render_window:
        cv2.namedWindow("Depth View", cv2.WINDOW_NORMAL)

    while total_timesteps < max_timesteps:
        for _ in range(steps_per_update):
            if total_timesteps >= max_timesteps:
                break

            episode_timesteps += 1
            total_timesteps += 1
            prev_action = agent.current_prev_action()

            if total_timesteps < start_timesteps and seed_args.load_model == "":
                # Populate online Mamba history even during random exploration.
                agent.select_action(base, depth)
                action = env.action_space.sample()
            else:
                action = agent.select_action(base, depth)
                _raise_if_non_finite(
                    "actor.action",
                    action,
                    f"algo={algo_name}, total_timesteps={total_timesteps}, episode={episode_num}",
                )

            critic_priv = np.asarray(obs.get("clean_depth", depth), dtype=np.float32) if is_pl else None

            try:
                next_obs, reward, terminated, truncated, step_info = env.step(action)
            except Exception as exc:
                print(f"CRITICAL ERROR in env.step: {exc}")
                env_core = _get_env_core(env)
                if hasattr(env_core, "check_ue4_status") and env_core.check_ue4_status(force_restart=True, reason="env_step_exception"):
                    _set_env_curriculum_progress(env, total_timesteps, max_timesteps)
                    obs, _ = env.reset(seed=int(seed_args.seed) + episode_num)
                    agent.reset_history()
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

            done = bool(terminated or truncated)
            done_float = float(done)
            next_depth = next_obs["depth"]
            next_base = next_obs["base"]
            next_critic_priv = np.asarray(next_obs.get("clean_depth", next_depth), dtype=np.float32) if is_pl else None

            _raise_if_non_finite("env.reward", reward, f"total_timesteps={total_timesteps}")
            _raise_if_non_finite("env.next_depth", next_depth, f"total_timesteps={total_timesteps}")
            _raise_if_non_finite("env.next_base", next_base, f"total_timesteps={total_timesteps}")

            if is_pl:
                agent.replay_buffer.add(
                    base,
                    depth,
                    prev_action,
                    action,
                    reward,
                    next_base,
                    next_depth,
                    done_float,
                    critic_priv=critic_priv,
                    next_critic_priv=next_critic_priv,
                )
            else:
                agent.replay_buffer.add(base, depth, prev_action, action, reward, next_base, next_depth, done_float)

            agent.observe_action(action)
            episode_reward += float(reward)

            if seed_args.render_window:
                _render_depth_window(next_depth, render_scale)

            obs = next_obs
            depth = next_depth
            base = next_base

            if done:
                env_core = _get_env_core(env)
                success_deque = getattr(env_core, "success_deque", [])
                success_rate = float(sum(success_deque) / len(success_deque)) if len(success_deque) > 0 else 0.0
                curriculum_info = _set_env_curriculum_progress(env, total_timesteps, max_timesteps)
                writer.add_scalar("train/episode_reward", episode_reward, total_timesteps)
                writer.add_scalar("train/episode_length", episode_timesteps, total_timesteps)
                writer.add_scalar("train/success_rate", success_rate, total_timesteps)
                if curriculum_info is not None:
                    writer.add_scalar("curriculum/level", getattr(env_core, "level", 0), total_timesteps)

                with open(csv_filename, mode="a", newline="") as f:
                    csv.writer(f).writerow([episode_num, total_timesteps, episode_reward, episode_timesteps, success_rate])

                print(
                    f"[{algo_name}] Episode {episode_num}, Reward: {episode_reward:.2f}, "
                    f"Length: {episode_timesteps}, Success Rate: {success_rate:.3f}, "
                    f"Level: {getattr(env_core, 'level', 0)}, Total Timesteps: {total_timesteps}, "
                    f"Total Successes: {getattr(env_core, 'success_count', 0)}"
                )

                episode_num += 1
                episode_reward = 0.0
                episode_timesteps = 0
                try:
                    obs, _ = env.reset(seed=int(seed_args.seed) + episode_num)
                except Exception as exc:
                    print(f"CRITICAL ERROR in env.reset: {exc}")
                    env_core = _get_env_core(env)
                    if hasattr(env_core, "check_ue4_status") and env_core.check_ue4_status(force_restart=True, reason="env_reset_exception"):
                        obs, _ = env.reset(seed=int(seed_args.seed) + episode_num)
                    else:
                        raise
                depth = obs["depth"]
                base = obs["base"]
                agent.reset_history()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        if agent.replay_buffer.size() >= seed_args.batch_size and total_timesteps >= update_after:
            _pause_env_simulation(env)
            n_updates = max(1, int(steps_per_update * float(seed_args.gradient_steps)))
            for _ in tqdm(range(n_updates), desc=f"Training ({total_timesteps})", leave=False):
                train_info = agent.train(progress_ratio=total_timesteps / max(float(max_timesteps), 1.0))
                if train_info:
                    update_step += 1
                    _log_train_metrics(writer, train_info, total_timesteps)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if total_timesteps > 0 and total_timesteps % 10000 == 0:
            model_path = os.path.join(model_dir, f"mamba_rsac_{total_timesteps}.pth")
            agent.save(model_path)
            print(f"Model saved at timestep {total_timesteps}: {model_path}")

    final_model_path = os.path.join(model_dir, "mamba_rsac_final.pth")
    agent.save(final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

    if seed_args.render_window:
        cv2.destroyAllWindows()
    if hasattr(env, "close"):
        env.close()
    writer.close()
    return env


def main():
    base_args = get_config()
    algo_name, core_name = _algorithm_from_args(base_args)

    for seed in _seed_list(base_args.seed):
        args = copy.deepcopy(base_args)
        args.seed = int(seed)
        args.algorithm_name = algo_name
        env = train_one(args, algo_name, core_name)
        env_core = _get_env_core(env)
        game_handler = getattr(env_core, "game_handler", None)
        if game_handler is not None:
            print(f"Closing AirSim for {algo_name} (seed={seed})...")
            game_handler.kill_game_in_editor()
            time.sleep(2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
