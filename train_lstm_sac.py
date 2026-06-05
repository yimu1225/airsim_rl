#!/usr/bin/env python3
"""
Standalone LSTM-SAC training script with paper-style curriculum learning.

Implements the 4-stage curriculum from Liu Bokai et al.:
  Stage 0 (0 ~ 20K steps): easy      — random exploration → feature pretrain → policy train
  Stage 1 (20K ~ 40K steps): junior   — feature pretrain → policy train (previous model)
  Stage 2 (40K ~ 60K steps): medium   — feature pretrain → policy train (previous model)
  Stage 3 (60K+ steps):      advanced — feature pretrain → policy train (previous model)

Feature extractor is trained for the first 2500 env steps of each stage.
During feature-only phase the policy (actor/critic) is frozen exactly as in the paper.
"""
import os
import sys
import time
import copy
import collections
import gc
import csv
import shutil
import numpy as np
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from algorithm.LSTM_SAC.agent import LSTMSACAgent
from gym_airsim.envs import AirSimEnv


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def _to_scalar_float(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if torch.is_tensor(value):
        return float(value.detach().cpu().item()) if value.numel() == 1 else None
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.item())
    return None


def _log_train_metrics_per_update(writer, train_info, update_step, algo_name, total_timesteps):
    if not isinstance(train_info, dict):
        return
    for key, value in train_info.items():
        sv = _to_scalar_float(value)
        if sv is None:
            continue
        key_str = str(key)
        if "loss" in key_str.lower():
            writer.add_scalar(f"loss/{key_str}", sv, total_timesteps)
        else:
            writer.add_scalar(f"update/{key_str}", sv, total_timesteps)


def _set_env_curriculum_progress(env, total_timesteps, max_timesteps):
    """Set curriculum progress on env if supported (progress_ratio)."""
    env_core = env.unwrapped if hasattr(env, "unwrapped") else env
    if not bool(getattr(env_core, "use_curriculum", False)):
        return None
    if getattr(env_core, "curriculum_mode", "progress") != "progress":
        return None
    set_progress = getattr(env_core, "set_curriculum_progress", None)
    if not callable(set_progress):
        return None
    ratio = min(max(float(total_timesteps) / max(float(max_timesteps), 1.0), 0.0), 1.0)
    return set_progress(ratio)


def _log_curriculum_info(writer, info, total_timesteps):
    if not isinstance(info, dict):
        return
    tag_map = {
        "progress_ratio": "curriculum/progress_ratio",
        "difficulty": "curriculum/difficulty",
        "level": "curriculum/level",
        "number_of_objects_min": "curriculum/number_of_objects_min",
        "number_of_objects_max": "curriculum/number_of_objects_max",
    }
    for key, tag in tag_map.items():
        v = info.get(key)
        if v is not None:
            writer.add_scalar(tag, v, total_timesteps)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def _train_single_seed(args, seed: int):
    """Run LSTM-SAC training for a single seed."""
    # Reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create a per-seed copy of args so env gets a single int, not a list
    seed_args = copy.deepcopy(args)
    seed_args.seed = seed

    # Create env (单帧深度图，与论文一致)
    seed_args.n_frames = 1
    env = AirSimEnv(takeoff_height=seed_args.takeoff_height, config=seed_args, stack_frames=1)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    _set_env_curriculum_progress(env, 0, seed_args.max_timesteps)
    obs, _ = env.reset(seed=seed)
    depth_image = obs["depth"]
    base_state = obs["base"]

    base_dim = base_state.shape[0]
    depth_shape = depth_image.shape  # (C,H,W) for stacked frames
    model_depth_shape = (1, depth_shape[-2], depth_shape[-1])  # LSTM-SAC is recurrent
    action_space = env.action_space

    print(f"[LSTM-SAC] Obs shapes: depth={depth_shape}, base={base_dim}")
    print(f"[LSTM-SAC] Action space: {action_space}")
    print(f"[LSTM-SAC] Max timesteps: {seed_args.max_timesteps}")
    print(f"[LSTM-SAC] Learning starts (random steps): {seed_args.learning_starts}")

    # Agent
    device = torch.device("cuda" if seed_args.cuda and torch.cuda.is_available() else "cpu")
    agent = LSTMSACAgent(base_dim, model_depth_shape, action_space, seed_args, device=device, seed=seed)

    # Logging
    run_name = f"LSTM_SAC_seed{seed}"
    log_dir = f"./results/{run_name}"
    model_dir = os.path.join("./models", run_name)
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    csv_path = os.path.join(log_dir, f"{run_name}_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "total_timesteps", "reward", "episode_length",
                     "success_rate"])

    print(f"[LSTM-SAC] Logging to {log_dir}")
    print(f"[LSTM-SAC] Models saved to {model_dir}")

    # ── Training loop ──
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0.0
    episode_timesteps = 0
    update_step = 0

    max_timesteps = int(seed_args.max_timesteps)
    steps_per_update = int(seed_args.steps_per_update)
    start_timesteps = int(seed_args.learning_starts)
    update_after = int(getattr(seed_args, "update_after", seed_args.learning_starts))

    print("Start LSTM-SAC Training Loop...")

    while total_timesteps < max_timesteps:
        # ── Collect steps_per_update ──
        for _ in range(steps_per_update):
            episode_timesteps += 1
            total_timesteps += 1

            # Prepare sequence input
            depth_seq = obs["depth"]
            if depth_seq.ndim == 3:
                depth_seq = np.expand_dims(depth_seq, axis=1)  # (T,1,H,W)

            # Action selection
            if total_timesteps <= start_timesteps:
                # Random exploration (paper Stage 0 early phase)
                action = env.action_space.sample()
            else:
                action = agent.select_action(
                    obs["base"], depth_seq, deterministic=False,
                    progress_ratio=total_timesteps / max_timesteps,
                )

            # Step
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            # Store in replay buffer
            next_depth_seq = next_obs["depth"]
            if next_depth_seq.ndim == 3:
                next_depth_seq = np.expand_dims(next_depth_seq, axis=1)

            agent.replay_buffer.add(
                obs["base"], depth_seq, action, reward,
                next_obs["base"], next_depth_seq, float(done),
            )

            obs = next_obs

            # ── Episode end ──
            if done:
                env_core = env.unwrapped if hasattr(env, "unwrapped") else env
                success_rate = (
                    sum(env_core.success_deque) / len(env_core.success_deque)
                    if env_core.success_deque else 0.0
                )
                curriculum_info = _set_env_curriculum_progress(env, total_timesteps, max_timesteps)

                # Log
                writer.add_scalar("train/episode_reward", episode_reward, total_timesteps)
                writer.add_scalar("train/episode_length", episode_timesteps, total_timesteps)
                writer.add_scalar("train/success_rate", success_rate, total_timesteps)
                writer.add_scalar("train/success_count", env_core.success_count, total_timesteps)
                _log_curriculum_info(writer, curriculum_info, total_timesteps)

                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([episode_num, total_timesteps, episode_reward,
                                episode_timesteps, success_rate])

                curriculum_suffix = ""
                if curriculum_info is not None:
                    curriculum_suffix = (
                        f", Difficulty: {curriculum_info['difficulty']:.3f}, "
                        f"Objects: {curriculum_info['number_of_objects_min']}-{curriculum_info['number_of_objects_max']}"
                    )
                print(f"[LSTM-SAC] Episode {episode_num}, Reward: {episode_reward:.2f}, Length: {episode_timesteps}, Success Rate: {success_rate:.3f}, Level: {env_core.level}{curriculum_suffix}, Total Timesteps: {total_timesteps}, Total Successes: {env_core.success_count}")

                # Periodic CUDA memory cleanup
                if episode_num % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'memory_summary'):
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"[Memory] CUDA cache cleared. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

                episode_num += 1
                episode_reward = 0.0
                episode_timesteps = 0

                obs, _ = env.reset(seed=seed + episode_num)

            # ── End of episode handling ──

        # ── Training update ──
        if agent.replay_buffer.size() >= seed_args.batch_size and total_timesteps >= update_after:
            n_updates = max(1, int(steps_per_update * seed_args.gradient_steps))
            for _ in tqdm(range(n_updates), desc=f"Training ({total_timesteps})", leave=False):
                progress_ratio = total_timesteps / max_timesteps
                train_info = agent.train(progress_ratio=progress_ratio)
                if train_info:
                    update_step += 1
                    _log_train_metrics_per_update(writer, train_info, update_step, "LSTM_SAC", total_timesteps)

            # Log feature-only status
            if hasattr(agent, "_feature_only_mode") and agent._feature_only_mode:
                stage = getattr(agent, "_current_stage", -1)
                writer.add_scalar("curriculum/stage", stage, total_timesteps)
                writer.add_scalar("curriculum/feature_only", 1.0, total_timesteps)

        # ── Checkpoint ──
        if total_timesteps % 10000 == 0:
            path = os.path.join(model_dir, f"async_{total_timesteps}.pth")
            agent.save(path)
            print(f"Model saved at timestep {total_timesteps}: {path}")

        # ── Periodic CUDA cleanup ──
        if total_timesteps % 5000 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            if hasattr(torch.cuda, 'memory_summary'):
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[Memory] CUDA cache cleared. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # ── Final save ──
    agent.save(os.path.join(model_dir, "async_final.pth"))
    writer.close()
    env.close()
    print(f"[LSTM-SAC seed={seed}] Training complete. Final model → {model_dir}/async_final.pth")


def main():
    args = get_config()
    # Parse all seeds
    raw_seed = args.seed
    if isinstance(raw_seed, (list, tuple)):
        seeds = [int(s) for s in raw_seed]
    else:
        seeds = [int(raw_seed)]

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training LSTM-SAC with seed={seed}")
        print(f"{'='*60}")
        _train_single_seed(args, seed)


if __name__ == "__main__":
    main()
