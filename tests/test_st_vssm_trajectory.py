#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render and record one successful VSSM-SAC trajectory in the ST UE4 scene.

The ST AirSim plugin must first be patched and rebuilt::

    python scripts/patch_st_trajectory_renderer.py --root /mnt/d/Projects/ST

This script then launches the ST project, evaluates deterministic CL-VSSM-SAC
episodes, and streams each live vehicle Actor position into a persistent UE4
line. Failed paths are cleared before the next attempt. On the first successful
episode, the line is finalized, the viewport switches to the fixed top-down
camera, and the complete NED trajectory is saved as CSV and NPZ.

Example::

    python tests/test_st_vssm_trajectory.py --model-seed 25 --max-episodes 30

After success, take the screenshot directly from the UE4 window. Press Ctrl+C
when finished; UE4 is left running unless ``--close-ue4-on-exit`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.ue4_trajectory import (  # noqa: E402
    TrajectoryPoint,
    UE4TrajectoryRpcError,
    UE4TrajectoryRenderer,
    default_trajectory_output_dir,
    save_successful_trajectory,
)

if TYPE_CHECKING:
    from eval.eval_env import SceneEvalAirSimEnv


ALGORITHM_NAME = "CL-VSSM-SAC"
DEFAULT_MODEL_SEED = 28
DEFAULT_FIRST_EPISODE_SEED = 1


def _parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic VSSM-SAC until one successful ST episode, "
            "render its trajectory inside UE4, and save all path points."
        )
    )
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument(
        "--first-episode-seed", type=int, default=DEFAULT_FIRST_EPISODE_SEED
    )
    parser.add_argument("--max-episodes", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return immediately after success instead of waiting for a manual screenshot.",
    )
    parser.add_argument(
        "--close-ue4-on-exit",
        action="store_true",
        help="Close the ST UE4 process when this script exits.",
    )
    script_args, remaining = parser.parse_known_args(argv)
    if script_args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")

    # Import the model stack only after argparse has handled --help. The local
    # Mamba/Triton build initializes a CUDA driver while importing agents.
    from algo_name_utils import to_internal_algorithm_name
    from algorithm.config_loader import apply_algorithm_params
    from config import get_config

    common_args = get_config(remaining)
    internal_name = to_internal_algorithm_name(ALGORITHM_NAME)
    common_args.algorithm_name = internal_name
    apply_algorithm_params(common_args, internal_name)
    return script_args, common_args


def _default_checkpoint(model_seed: int) -> str:
    model_dir = REPO_ROOT / "models" / ALGORITHM_NAME / f"seed{int(model_seed)}"
    candidates = (model_dir / "test.pth", model_dir / "async_final.pth")
    return str(next((path for path in candidates if path.is_file()), candidates[0]))


def _prepare_action_inputs(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(observation["base"], dtype=np.float32)
    depth = np.asarray(observation["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = np.expand_dims(depth, axis=1)
    return base, depth


def _padded_action(action: Any) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    padded = np.full((3,), np.nan, dtype=np.float64)
    padded[: min(values.size, padded.size)] = values[: padded.size]
    return padded


def _capture_point(
    env: SceneEvalAirSimEnv,
    *,
    airsim_module: Any,
    step: int,
    started_at: float,
    action: Any,
    reward: float, 
    cumulative_reward: float,
    path_length: float,
    is_terminal: bool,
) -> TrajectoryPoint:
    vehicle = env.airgym.client
    pose = vehicle.simGetVehiclePose()
    position = pose.position
    pitch, roll, yaw = airsim_module.to_eularian_angles(pose.orientation)
    state = vehicle.getMultirotorState()
    velocity = state.kinematics_estimated.linear_velocity
    action_values = _padded_action(action)

    xyz = np.asarray(
        [position.x_val, position.y_val, position.z_val], dtype=np.float64
    )
    goal = np.asarray(env.goal, dtype=np.float64).reshape(3)
    return TrajectoryPoint(
        step=int(step),
        elapsed_seconds=float(time.monotonic() - started_at),
        x_ned_m=float(xyz[0]),
        y_ned_m=float(xyz[1]),
        z_ned_m=float(xyz[2]),
        vx_m_s=float(velocity.x_val),
        vy_m_s=float(velocity.y_val),
        vz_m_s=float(velocity.z_val),
        roll_rad=float(roll),
        pitch_rad=float(pitch),
        yaw_rad=float(yaw),
        action_forward_m_s=float(action_values[0]),
        action_yaw_rate_rad_s=float(action_values[1]),
        action_vertical_m_s=float(action_values[2]),
        reward=float(reward),
        cumulative_reward=float(cumulative_reward),
        distance_to_goal_m=float(np.linalg.norm(goal - xyz)),
        path_length_m=float(path_length),
        is_terminal=bool(is_terminal),
    )


def _build_agent(
    args: Any,
    observation: dict,
    action_space: Any,
    *,
    model_seed: int,
    checkpoint: str,
) -> tuple[Any, Any]:
    import torch

    from eval.eval_common import set_agent_eval_mode
    from main_async import get_agent_class

    depth_shape = np.asarray(observation["depth"]).shape
    model_depth_shape = (1, int(depth_shape[-2]), int(depth_shape[-1]))
    base_dim = int(np.asarray(observation["base"]).size)
    device = torch.device(
        "cuda" if bool(args.cuda) and torch.cuda.is_available() else "cpu"
    )
    AgentClass = get_agent_class(args.algorithm_name)
    agent = AgentClass(
        base_dim,
        model_depth_shape,
        action_space,
        args,
        device=device,
        seed=int(model_seed),
    )
    agent.load(checkpoint)
    set_agent_eval_mode(agent)
    return agent, device


def _termination(info: dict, *, truncated: bool, step: int, max_steps: int) -> str:
    if bool(info.get("ue4_restarted", False)):
        return "environment_restart"
    if bool(info.get("is_success", False)):
        return "success"
    if bool(info.get("has_collided", False)):
        return "collision"
    if truncated or step >= max_steps:
        return "timeout"
    return "other_failure"


def _append_episode_trajectory_point(
    renderer: UE4TrajectoryRenderer,
    *,
    episode_vehicle: Any,
    current_vehicle: Any,
    info: dict[str, Any],
) -> bool:
    """Append one point while the episode still belongs to the same UE4 process."""

    if bool(info.get("ue4_restarted", False)) or current_vehicle is not episode_vehicle:
        return False
    try:
        renderer.append_current_vehicle_position()
    except UE4TrajectoryRpcError as exc:
        print(f"[ST-Trajectory] Trajectory renderer unavailable: {exc}")
        return False
    return True


def _clear_failed_trajectory(renderer: UE4TrajectoryRenderer) -> None:
    try:
        renderer.clear()
    except UE4TrajectoryRpcError as exc:
        print(f"[ST-Trajectory] Could not clear failed UE4 path: {exc}")


def run(script_args: argparse.Namespace, common_args: Any) -> dict[str, Path]:
    import airsim
    import torch

    from eval.eval_common import close_env, resolve_checkpoint, select_eval_action
    from eval.eval_env import SceneEvalAirSimEnv
    from main_async import _configure_reproducibility

    model_seed = int(script_args.model_seed)
    common_args.seed = int(script_args.first_episode_seed)
    _configure_reproducibility(model_seed, common_args)
    checkpoint = resolve_checkpoint(
        script_args.checkpoint,
        _default_checkpoint(model_seed),
    )

    env = SceneEvalAirSimEnv(
        takeoff_height=common_args.takeoff_height,
        config=common_args,
        stack_frames=int(common_args.n_frames),
    )
    attempts: list[dict[str, Any]] = []
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(model_seed)
        initial_obs, _ = env.reset(seed=int(script_args.first_episode_seed))
        agent, device = _build_agent(
            common_args,
            initial_obs,
            env.action_space,
            model_seed=model_seed,
            checkpoint=checkpoint,
        )
        print(f"[ST-Trajectory] Loaded {ALGORITHM_NAME}: {checkpoint}")
        print(f"[ST-Trajectory] Inference device: {device}")

        max_steps = int(common_args.episode_length)
        for attempt in range(1, int(script_args.max_episodes) + 1):
            episode_seed = int(script_args.first_episode_seed) + attempt - 1
            print(
                f"[ST-Trajectory] Attempt {attempt}/{script_args.max_episodes}, "
                f"episode_seed={episode_seed}"
            )
            observation, _ = env.reset(seed=episode_seed)
            episode_vehicle = env.airgym.client
            renderer = UE4TrajectoryRenderer(episode_vehicle)
            try:
                renderer.clear()
                renderer.set_goal_ned(env.goal)
                renderer.append_current_vehicle_position()
            except UE4TrajectoryRpcError as exc:
                attempts.append(
                    {
                        "attempt": attempt,
                        "episode_seed": episode_seed,
                        "termination": "renderer_unavailable",
                        "episode_steps": 0,
                        "episode_reward": 0.0,
                        "path_length_m": 0.0,
                    }
                )
                print(
                    "[ST-Trajectory] termination=renderer_unavailable, "
                    f"episode will be retried: {exc}"
                )
                continue

            points: list[TrajectoryPoint] = []
            episode_reward = 0.0
            started_at = time.monotonic()
            points.append(
                _capture_point(
                    env,
                    airsim_module=airsim,
                    step=0,
                    started_at=started_at,
                    action=[math.nan, math.nan, math.nan],
                    reward=0.0,
                    cumulative_reward=0.0,
                    path_length=0.0,
                    is_terminal=False,
                )
            )

            termination = "timeout"
            last_info: dict[str, Any] = {}
            for step in range(1, max_steps + 1):
                base, depth = _prepare_action_inputs(observation)
                with torch.inference_mode():
                    action = select_eval_action(
                        agent,
                        base,
                        depth,
                        progress_ratio=1.0,
                    )
                next_observation, reward, terminated, truncated, info = env.step(action)
                last_info = dict(info) if isinstance(info, dict) else {}
                episode_reward += float(reward)
                done = bool(terminated or truncated)

                trajectory_appended = _append_episode_trajectory_point(
                    renderer,
                    episode_vehicle=episode_vehicle,
                    current_vehicle=env.airgym.client,
                    info=last_info,
                )
                if not trajectory_appended:
                    if bool(last_info.get("ue4_restarted", False)) or (
                        env.airgym.client is not episode_vehicle
                    ):
                        last_info["ue4_restarted"] = True
                        termination = "environment_restart"
                    else:
                        termination = "renderer_unavailable"
                    break
                points.append(
                    _capture_point(
                        env,
                        airsim_module=airsim,
                        step=step,
                        started_at=started_at,
                        action=action,
                        reward=float(reward),
                        cumulative_reward=episode_reward,
                        path_length=float(last_info.get("path_length", 0.0)),
                        is_terminal=done,
                    )
                )
                observation = next_observation
                if done:
                    termination = _termination(
                        last_info,
                        truncated=bool(truncated),
                        step=step,
                        max_steps=max_steps,
                    )
                    break

            attempt_summary = {
                "attempt": attempt,
                "episode_seed": episode_seed,
                "termination": termination,
                "episode_steps": len(points) - 1,
                "episode_reward": episode_reward,
                "path_length_m": float(last_info.get("path_length", 0.0)),
            }
            attempts.append(attempt_summary)
            print(
                "[ST-Trajectory] "
                f"termination={termination}, steps={attempt_summary['episode_steps']}, "
                f"reward={episode_reward:.3f}, "
                f"path_length={attempt_summary['path_length_m']:.3f}m"
            )

            if termination != "success":
                _clear_failed_trajectory(renderer)
                continue

            env.airgym.client.simPause(True)
            try:
                renderer.finalize_success()
                renderer.switch_to_topdown()
            except UE4TrajectoryRpcError as exc:
                print(
                    "[ST-Trajectory] Successful flight could not be prepared for "
                    f"the UE4 screenshot; retrying another episode: {exc}"
                )
                _clear_failed_trajectory(renderer)
                continue
            output_dir = (
                script_args.output_dir.resolve()
                if script_args.output_dir is not None
                else default_trajectory_output_dir(
                    REPO_ROOT,
                    model_seed=model_seed,
                    episode_seed=episode_seed,
                )
            )
            paths = save_successful_trajectory(
                output_dir,
                points,
                {
                    "algorithm": ALGORITHM_NAME,
                    "model_seed": model_seed,
                    "episode_seed": episode_seed,
                    "successful_attempt": attempt,
                    "termination": "success",
                    "episode_reward": episode_reward,
                    "episode_steps": len(points) - 1,
                    "episode_path_length_m": float(
                        last_info.get("path_length", 0.0)
                    ),
                    "goal_ned_m": np.asarray(env.goal, dtype=float).tolist(),
                    "checkpoint": str(Path(checkpoint).resolve()),
                    "coordinate_frame": "AirSim local NED metres",
                    "ue4_rendering": (
                        "Persistent UE4 debug line sampled from the live vehicle Actor; "
                        "green=start, blue=terminal position, red=configured goal"
                    ),
                    "episode_attempts": attempts,
                },
            )
            print(f"[ST-Trajectory] Successful CSV: {paths['csv']}")
            print(f"[ST-Trajectory] Successful NPZ: {paths['npz']}")
            print(
                "[ST-Trajectory] UE4 is paused in top-down view with the "
                "successful path visible. Take the screenshot now."
            )
            if not script_args.no_wait:
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print("\n[ST-Trajectory] Screenshot wait finished.")
            return paths

        raise RuntimeError(
            f"No successful VSSM-SAC episode was found within "
            f"{script_args.max_episodes} attempts"
        )
    finally:
        if script_args.close_ue4_on_exit:
            close_env(env, label="ST VSSM-SAC trajectory capture")
        else:
            print("[ST-Trajectory] UE4 is left running.")


def main(argv: Sequence[str] | None = None) -> int:
    script_args, common_args = _parse_args(argv)
    run(script_args, common_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
