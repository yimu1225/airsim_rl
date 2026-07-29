#!/usr/bin/env python3
"""Shared evaluation helpers for AirSim RL training entrypoints."""

import csv
import inspect
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch


@dataclass
class EpisodeResult:
    episode: int
    reward: float
    length: int
    path_length: float
    success: bool
    碰撞率: bool
    termination_reason: str

    @property
    def is_timeout(self) -> bool:
        return self.termination_reason == "timeout"


def classify_termination(
    *,
    success: bool,
    collided: bool,
    truncated: bool,
    episode_length: int,
    max_episode_length: Optional[int],
    info: Dict,
) -> str:
    """Return one mutually exclusive terminal status for an evaluation episode."""
    if success:
        return "success"
    if collided:
        return "collision"
    if bool(info.get("ue4_restarted", False)):
        return "environment_restart"
    if truncated:
        return "timeout"
    if max_episode_length is not None and episode_length >= max_episode_length:
        return "timeout"
    return "other_failure"


def seeds_from_args(args) -> List[int]:
    raw_seed = args.seed
    if isinstance(raw_seed, (list, tuple)):
        return [int(seed) for seed in raw_seed]
    return [int(raw_seed)]


def get_env_core(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def set_eval_curriculum_progress(env):
    env_core = get_env_core(env)
    if not bool(getattr(env_core, "use_curriculum", False)):
        return None
    if getattr(env_core, "curriculum_mode", "progress") != "progress":
        return None
    set_progress = getattr(env_core, "set_curriculum_progress", None)
    if not callable(set_progress):
        return None
    return set_progress(1.0)


def set_agent_eval_mode(agent) -> None:
    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            value.eval()


def resolve_checkpoint(load_model: str, default_path: str) -> str:
    checkpoint = load_model if load_model else default_path
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}. "
            "Use --load_model to specify a checkpoint explicitly."
        )
    return checkpoint


def select_eval_action(agent, base, depth, *, progress_ratio: float = 1.0, critic_priv=None):
    select_action = agent.select_action
    params = inspect.signature(select_action).parameters
    kwargs = {}
    if "deterministic" in params:
        kwargs["deterministic"] = True
    if "noise" in params:
        kwargs["noise"] = False
    if "progress_ratio" in params:
        kwargs["progress_ratio"] = progress_ratio
    if "critic_priv" in params:
        kwargs["critic_priv"] = critic_priv

    action = select_action(base, depth, **kwargs)
    if isinstance(action, tuple):
        action = action[0]
    return action


def render_depth_window(depth, *, is_recurrent: bool, depth_view_scale: float) -> None:
    vis_imgs = []
    depth_arr = np.asarray(depth)
    if is_recurrent:
        if depth_arr.ndim == 4 and depth_arr.shape[1] == 1:
            depth_arr = depth_arr[:, 0]
        if depth_arr.ndim == 3:
            vis_imgs.extend([depth_arr[i] for i in range(depth_arr.shape[0])])
        else:
            vis_imgs.append(depth_arr)
    else:
        if depth_arr.ndim == 3:
            vis_imgs.extend([depth_arr[i] for i in range(depth_arr.shape[0])])
        else:
            vis_imgs.append(depth_arr)

    processed = []
    for img in vis_imgs:
        img = np.asarray(img)
        if img.ndim == 3:
            img = img.squeeze(0)
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        processed.append(img)

    if not processed:
        return

    vis_concat = np.hstack(processed)
    scale = max(float(depth_view_scale), 1.0)
    if scale != 1.0:
        vis_concat = cv2.resize(
            vis_concat,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    height, width = vis_concat.shape[:2]
    cv2.resizeWindow("Depth View", width, height)
    cv2.imshow("Depth View", vis_concat)
    cv2.waitKey(1)


def run_eval_episodes(
    env,
    agent,
    args,
    *,
    seed: int,
    is_recurrent: bool,
    prepare_action_inputs: Callable[[dict], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    after_step: Optional[Callable[[dict, dict], None]] = None,
    on_episode_reset: Optional[Callable[[dict], None]] = None,
    label: str,
    csv_path: Optional[str] = None,
) -> List[EpisodeResult]:
    set_agent_eval_mode(agent)
    depth_view_scale = max(float(getattr(args, "depth_view_scale", 2.5)), 1.0)
    if args.render_window:
        cv2.namedWindow("Depth View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth View", int(256 * depth_view_scale), int(256 * depth_view_scale))

    results: List[EpisodeResult] = []
    eval_episodes = int(getattr(args, "eval_episodes", 10))

    for episode in range(eval_episodes):
        set_eval_curriculum_progress(env)
        obs, _ = env.reset(seed=seed + episode)
        if on_episode_reset is not None:
            on_episode_reset(obs)
        episode_reward = 0.0
        episode_length = 0
        done = False
        last_info: Dict = {}
        last_truncated = False

        while not done:
            base, depth, critic_priv = prepare_action_inputs(obs)
            with torch.no_grad():
                action = select_eval_action(
                    agent,
                    base,
                    depth,
                    progress_ratio=1.0,
                    critic_priv=critic_priv,
                )

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)
            last_truncated = bool(truncated)
            last_info = step_info if isinstance(step_info, dict) else {}
            episode_reward += float(reward)
            episode_length += 1

            if args.render_window:
                render_depth_window(depth, is_recurrent=is_recurrent, depth_view_scale=depth_view_scale)

            if after_step is not None:
                after_step(obs, next_obs)
            obs = next_obs

        success = bool(last_info.get("is_success", False))
        碰撞率 = bool(last_info.get("has_collided", False))
        if "path_length" not in last_info:
            raise RuntimeError(
                "Evaluation environment did not report path_length; "
                "real travelled distance cannot be reconstructed from episode steps."
            )
        path_length = float(last_info["path_length"])
        if not np.isfinite(path_length) or path_length < 0.0:
            raise ValueError(f"Invalid episode path_length: {path_length}")
        configured_max_length = getattr(args, "episode_length", None)
        max_episode_length = (
            int(configured_max_length)
            if configured_max_length is not None
            else None
        )
        termination_reason = classify_termination(
            success=success,
            collided=碰撞率,
            truncated=last_truncated,
            episode_length=episode_length,
            max_episode_length=max_episode_length,
            info=last_info,
        )
        results.append(
            EpisodeResult(
                episode=episode,
                reward=episode_reward,
                length=episode_length,
                path_length=path_length,
                success=success,
                碰撞率=碰撞率,
                termination_reason=termination_reason,
            )
        )
        # 实时累计统计
        cum_rewards = [r.reward for r in results]
        cum_successes = [r.success for r in results]
        cum_collisions = [r.碰撞率 for r in results]
        successful_path_lengths = [r.path_length for r in results if r.success]
        path_length_summary = (
            f", avg_success_path_length={np.mean(successful_path_lengths):.2f}m"
            if successful_path_lengths
            else ""
        )
        print(
            f"[Eval][{label}] Episode {episode}: "
            f"avg_reward={np.mean(cum_rewards):.2f}, "
            f"success_rate={np.mean(cum_successes):.3f}, "
            f"collision_rate={np.mean(cum_collisions):.3f}"
            f"{path_length_summary}, "
            f"termination={termination_reason}"
        )
        # 增量写入 CSV，防止中途崩溃丢失数据
        if csv_path is not None:
            write_eval_csv(results, csv_path)

    if args.render_window:
        cv2.destroyAllWindows()
    return results


def write_eval_csv(results: List[EpisodeResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "reward",
            "episode_length",
            "path_length",
            "is_success",
            "has_collided",
            "is_timeout",
            "termination_reason",
            "success_rate",
            "collision_rate",
        ])
        success_count = 0
        collision_count = 0
        for idx, row in enumerate(results, start=1):
            success_count += int(row.success)
            collision_count += int(row.碰撞率)
            writer.writerow([
                row.episode,
                row.reward,
                row.length,
                f"{row.path_length:.2f}",
                int(row.success),
                int(row.碰撞率),
                int(row.is_timeout),
                row.termination_reason,
                success_count / idx,
                collision_count / idx,
            ])


def print_eval_summary(results: List[EpisodeResult], *, label: str, csv_path: str) -> None:
    rewards = np.asarray([row.reward for row in results], dtype=np.float32)
    successes = np.asarray([row.success for row in results], dtype=np.float32)
    collisions = np.asarray([row.碰撞率 for row in results], dtype=np.float32)
    successful_path_lengths = np.asarray(
        [row.path_length for row in results if row.success],
        dtype=np.float32,
    )
    path_length_summary = (
        f", mean_success_path_length={float(successful_path_lengths.mean()):.2f}m"
        if successful_path_lengths.size
        else ""
    )
    print(
        f"[Eval][{label}] Summary: "
        f"episodes={len(results)}, "
        f"mean_reward={float(rewards.mean()):.2f}, "
        f"std_reward={float(rewards.std()):.2f}, "
        f"success_rate={float(successes.mean()):.3f}, "
        f"collision_rate={float(collisions.mean()):.3f}"
        f"{path_length_summary}"
    )
    print(f"[Eval][{label}] CSV saved to {csv_path}")


def close_env(env, *, label: str = "") -> None:
    try:
        if hasattr(env, "close"):
            env.close()
    finally:
        env_core = get_env_core(env)
        game_handler = getattr(env_core, "game_handler", None)
        if game_handler is not None:
            prefix = f" for {label}" if label else ""
            print(f"Closing AirSim{prefix}...")
            game_handler.kill_game_in_editor()
            time.sleep(2)
