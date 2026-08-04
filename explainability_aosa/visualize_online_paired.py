#!/usr/bin/env python3
"""Collect one successful training-environment rollout and explain it online.

The CL-VSSM-SAC actor controls the AirSim training environment with
deterministic actions.  Failed episodes are discarded.  Once a successful
episode with enough observations is obtained, twenty temporally separated
candidate steps are saved and explained for both CL-VSSM-SAC and CL-SAC.
The paper-style trajectory figure displays four evenly spaced candidates from
those twenty explanations.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from algorithm.SB_PER_VSSM_SAC.agent import (  # noqa: E402
    SB_PERVSSM_SACAgent,
)
from algorithm.config_loader import apply_algorithm_params  # noqa: E402
from config import get_config  # noqa: E402
from eval.eval_common import (  # noqa: E402
    close_env,
    resolve_checkpoint,
    select_eval_action,
    set_agent_eval_mode,
)
from gym_airsim.envs import AirSimEnv  # noqa: E402
from settings_folder import settings  # noqa: E402
from visualize_paired import (  # noqa: E402
    TrajectorySample,
    _configure_reproducibility,
    _load_actor,
    _prepare_depth,
    _resolve_device,
    run as run_paired_explanations,
)


DEFAULT_MODEL_SEED = 25
DEFAULT_ENVIRONMENT_SEED = 20260803
DEFAULT_CANDIDATE_STEPS = 20
DEFAULT_SUMMARY_STEPS = 4
FIXED_NUMBER_OF_OBJECTS = 180


@dataclass(frozen=True)
class CollectedStep:
    sample: TrajectorySample
    reward: float


@dataclass(frozen=True)
class SuccessfulEpisode:
    attempt: int
    episode_seed: int
    steps: tuple[CollectedStep, ...]
    total_reward: float
    path_length: float


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )


def _default_output_dir(model_seed: int, environment_seed: int) -> Path:
    return (
        REPO_ROOT
        / "results"
        / "explainability"
        / "aosa"
        / "online_training"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / f"environment_seed{int(environment_seed)}"
        / _utc_stamp()
    )


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _obstacle_proximity(depth: np.ndarray) -> float:
    latest = _prepare_depth(depth)[-1]
    return float(255.0 - np.percentile(latest, 10.0))


def _configure_fixed_obstacle_count(number_of_objects: int) -> None:
    """Pin every training difficulty range before AirSim is launched."""

    fixed_range = [int(number_of_objects)]
    for range_name in (
        "easy_range_dic",
        "medium_range_dic",
        "hard_range_dic",
        "dynamic_obstacles_dic",
    ):
        range_config = getattr(settings, range_name)
        range_config["NumberOfObjects"] = fixed_range.copy()


def _assert_obstacle_count(env, expected: int) -> int:
    actual = int(env.getItemCurGameConfig("NumberOfObjects"))
    if actual != int(expected):
        raise RuntimeError(
            "The online explanation environment must contain exactly "
            f"{expected} obstacles, but AirSim loaded {actual}"
        )
    return actual


def _select_spaced_top_indices(
    scores: Sequence[float],
    *,
    count: int,
    min_gap: int,
) -> list[int]:
    """Select high-proximity observations with deterministic spacing."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores must contain finite values")
    if count <= 0:
        raise ValueError("count must be positive")
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")

    target_count = min(int(count), int(values.size))
    ranking = np.argsort(-values, kind="stable").tolist()
    for effective_gap in range(int(min_gap), -1, -1):
        selected: list[int] = []
        for index in ranking:
            if all(
                abs(index - previous) >= effective_gap
                for previous in selected
            ):
                selected.append(index)
                if len(selected) == target_count:
                    return sorted(selected)
    raise AssertionError("gap-zero selection must satisfy the target count")


def _build_rollout_agent(
    common_args,
    initial_observation: dict,
    action_space,
    *,
    model_seed: int,
    checkpoint: str,
    device: torch.device,
):
    model_args = copy.deepcopy(common_args)
    model_args.algorithm_name = "CL-VSSM-SAC"
    model_args.seed = int(model_seed)
    model_args.n_frames = int(initial_observation["depth"].shape[0])
    apply_algorithm_params(model_args, model_args.algorithm_name)

    height, width = map(int, initial_observation["depth"].shape[-2:])
    agent = SB_PERVSSM_SACAgent(
        int(initial_observation["base"].size),
        (1, height, width),
        action_space,
        model_args,
        device=device,
        seed=int(model_seed),
    )
    _load_actor(agent, checkpoint)
    set_agent_eval_mode(agent)
    return agent


def _run_episode(
    env,
    agent,
    *,
    attempt: int,
    episode_seed: int,
    max_steps: int,
    number_of_objects: int,
) -> tuple[SuccessfulEpisode | None, dict]:
    obs, _ = env.reset(seed=int(episode_seed))
    loaded_number_of_objects = _assert_obstacle_count(
        env, number_of_objects
    )
    trajectory: list[CollectedStep] = []
    total_reward = 0.0
    termination = "max_steps"
    last_info: dict = {}

    for step in range(1, int(max_steps) + 1):
        base = np.asarray(obs["base"], dtype=np.float32).copy()
        depth = _prepare_depth(obs["depth"]).copy()
        with torch.inference_mode():
            action = np.asarray(
                select_eval_action(
                    agent,
                    base,
                    depth,
                    progress_ratio=1.0,
                ),
                dtype=np.float32,
            ).reshape(-1)

        next_obs, reward, terminated, truncated, info = env.step(action)
        last_info = dict(info) if isinstance(info, dict) else {}
        total_reward += float(reward)
        trajectory.append(
            CollectedStep(
                sample=TrajectorySample(
                    step=step,
                    base_state=base,
                    depth=depth,
                    reference_physical_action=action.copy(),
                    obstacle_proximity=_obstacle_proximity(depth),
                ),
                reward=float(reward),
            )
        )
        obs = next_obs

        if terminated or truncated:
            if bool(last_info.get("is_success", False)):
                termination = "success"
            elif bool(last_info.get("has_collided", False)):
                termination = "collision"
            elif bool(last_info.get("ue4_restarted", False)):
                termination = "environment_restart"
            elif truncated or step >= int(max_steps):
                termination = "timeout"
            else:
                termination = "other_failure"
            break

    summary = {
        "attempt": int(attempt),
        "episode_seed": int(episode_seed),
        "termination": termination,
        "length": len(trajectory),
        "total_reward": float(total_reward),
        "path_length": float(last_info.get("path_length", 0.0)),
        "number_of_objects": loaded_number_of_objects,
    }
    if termination != "success":
        return None, summary
    return (
        SuccessfulEpisode(
            attempt=int(attempt),
            episode_seed=int(episode_seed),
            steps=tuple(trajectory),
            total_reward=float(total_reward),
            path_length=float(last_info.get("path_length", 0.0)),
        ),
        summary,
    )


def _collect_successful_episode(
    env,
    agent,
    *,
    first_episode_seed: int,
    max_episodes: int,
    max_steps: int,
    candidate_steps: int,
    curriculum_progress: float,
    number_of_objects: int,
) -> tuple[SuccessfulEpisode, list[dict]]:
    attempts: list[dict] = []
    for attempt in range(1, int(max_episodes) + 1):
        set_progress = getattr(env, "set_curriculum_progress", None)
        if callable(set_progress):
            set_progress(float(curriculum_progress))
        episode_seed = int(first_episode_seed) + attempt - 1
        print(
            f"[Online-AOSA] Episode attempt {attempt}/{max_episodes}, "
            f"seed={episode_seed}"
        )
        episode, summary = _run_episode(
            env,
            agent,
            attempt=attempt,
            episode_seed=episode_seed,
            max_steps=max_steps,
            number_of_objects=number_of_objects,
        )
        if episode is not None and len(episode.steps) < int(candidate_steps):
            summary["termination"] = "success_but_too_short"
            summary["minimum_required_steps"] = int(candidate_steps)
            episode = None
        attempts.append(summary)
        print(
            "[Online-AOSA] "
            f"termination={summary['termination']}, "
            f"length={summary['length']}, "
            f"reward={summary['total_reward']:.3f}"
        )
        if episode is not None:
            return episode, attempts
    raise RuntimeError(
        f"No successful episode with at least {candidate_steps} steps was "
        f"found within {max_episodes} attempts"
    )


def _save_online_trajectory(
    episode: SuccessfulEpisode,
    trajectory_dir: Path,
    *,
    candidate_indices: Sequence[int],
    model_seed: int,
    environment_seed: int,
    curriculum_progress: float,
    number_of_objects: int,
    checkpoint: str,
    attempts: Sequence[dict],
) -> tuple[list[TrajectorySample], dict]:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    steps = list(episode.steps)
    selected = [steps[int(index)].sample for index in candidate_indices]

    np.savez_compressed(
        trajectory_dir / "successful_episode.npz",
        step=np.asarray(
            [item.sample.step for item in steps], dtype=np.int32
        ),
        base_state=np.stack(
            [item.sample.base_state for item in steps], axis=0
        ).astype(np.float32),
        depth=np.stack(
            [item.sample.depth for item in steps], axis=0
        ).astype(np.float32),
        physical_action=np.stack(
            [item.sample.reference_physical_action for item in steps], axis=0
        ).astype(np.float32),
        reward=np.asarray([item.reward for item in steps], dtype=np.float32),
        obstacle_proximity=np.asarray(
            [item.sample.obstacle_proximity for item in steps],
            dtype=np.float32,
        ),
    )

    for sample in selected:
        np.savez_compressed(
            trajectory_dir / f"step_{sample.step:04d}_online.npz",
            step=np.int32(sample.step),
            base_state=sample.base_state.astype(np.float32),
            depth=sample.depth.astype(np.float32),
            reference_physical_action=(
                sample.reference_physical_action.astype(np.float32)
            ),
            obstacle_proximity=np.float32(sample.obstacle_proximity),
        )

    metadata = {
        "source_type": "online_training_environment",
        "algorithm": "CL-VSSM-SAC",
        "termination": "success",
        "model_seed": int(model_seed),
        "environment_seed": int(environment_seed),
        "episode_seed": int(episode.episode_seed),
        "successful_attempt": int(episode.attempt),
        "episode_length": len(steps),
        "episode_reward": float(episode.total_reward),
        "episode_path_length": float(episode.path_length),
        "curriculum_progress": float(curriculum_progress),
        "number_of_objects": int(number_of_objects),
        "number_of_objects_mode": "fixed",
        "checkpoint": str(Path(checkpoint).resolve()),
        "full_trajectory_file": "successful_episode.npz",
        "sample_file_pattern": "step_{step:04d}_online.npz",
        "reference_action_key": "reference_physical_action",
        "selection_method": (
            "top_obstacle_proximity_with_temporal_spacing"
        ),
        "capture_steps_requested": len(selected),
        "capture_steps_completed": [sample.step for sample in selected],
        "episode_attempts": list(attempts),
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_json(trajectory_dir / "metadata.json", metadata)
    return selected, metadata


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Launch the AirSim training environment, collect one successful "
            "CL-VSSM-SAC episode, explain 20 candidate steps with both "
            "CL-VSSM-SAC and CL-SAC, and display 4 steps."
        )
    )
    parser.add_argument("--model_seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument(
        "--environment_seed", type=int, default=DEFAULT_ENVIRONMENT_SEED
    )
    parser.add_argument("--first_episode_seed", type=int, default=1)
    parser.add_argument("--max_episodes", type=int, default=30)
    parser.add_argument(
        "--candidate_steps", type=int, default=DEFAULT_CANDIDATE_STEPS
    )
    parser.add_argument(
        "--summary_steps", type=int, default=DEFAULT_SUMMARY_STEPS
    )
    parser.add_argument("--min_sample_gap", type=int, default=5)
    parser.add_argument(
        "--curriculum_progress", type=float, default=1.0
    )
    parser.add_argument("--vssm_checkpoint", type=str, default=None)
    parser.add_argument("--sac_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="cuda"
    )
    parser.add_argument("--window_height", type=int, default=16)
    parser.add_argument("--window_width", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--temporal_radius", type=int, default=1)
    parser.add_argument("--reference_depth", type=float, default=255.0)
    parser.add_argument("--occlusion_batch_size", type=int, default=32)
    parser.add_argument(
        "--motion", choices=("flow", "fixed"), default="flow"
    )
    parser.add_argument("--max_masks", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    script_args, remaining = parser.parse_known_args(argv)

    positive = {
        "max_episodes": script_args.max_episodes,
        "candidate_steps": script_args.candidate_steps,
        "summary_steps": script_args.summary_steps,
        "window_height": script_args.window_height,
        "window_width": script_args.window_width,
        "stride": script_args.stride,
        "occlusion_batch_size": script_args.occlusion_batch_size,
        "dpi": script_args.dpi,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            parser.error(f"--{name} must be positive")
    if script_args.summary_steps > script_args.candidate_steps:
        parser.error("--summary_steps cannot exceed --candidate_steps")
    if script_args.min_sample_gap < 0:
        parser.error("--min_sample_gap must be non-negative")
    if not 0.0 <= script_args.curriculum_progress <= 1.0:
        parser.error("--curriculum_progress must be in [0, 1]")
    if script_args.temporal_radius < 0:
        parser.error("--temporal_radius must be non-negative")
    if script_args.max_masks is not None and script_args.max_masks <= 0:
        parser.error("--max_masks must be positive")

    common_args = get_config(remaining)
    common_args.algorithm_name = "CL-VSSM-SAC"
    common_args.seed = int(script_args.environment_seed)
    apply_algorithm_params(common_args, common_args.algorithm_name)
    return script_args, common_args


def run(script_args, common_args) -> Path:
    model_seed = int(script_args.model_seed)
    environment_seed = int(script_args.environment_seed)
    output_dir = (
        Path(script_args.output_dir).resolve()
        if script_args.output_dir
        else _default_output_dir(model_seed, environment_seed)
    )
    trajectory_dir = output_dir / "online_trajectory"
    explanation_dir = output_dir / "paired_explanation"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(script_args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "Online CL-VSSM-SAC collection requires CUDA for the fused "
            "Mamba/Triton encoder"
        )
    _configure_reproducibility(model_seed)

    vssm_checkpoint = resolve_checkpoint(
        script_args.vssm_checkpoint,
        str(
            REPO_ROOT
            / "models"
            / "CL-VSSM-SAC"
            / f"seed{model_seed}"
            / "test.pth"
        ),
    )
    sac_checkpoint = resolve_checkpoint(
        script_args.sac_checkpoint,
        str(
            REPO_ROOT
            / "models"
            / "CL-SAC"
            / f"seed{model_seed}"
            / "async_final.pth"
        ),
    )

    env_args = copy.deepcopy(common_args)
    env_args.algorithm_name = "CL-VSSM-SAC"
    env_args.seed = environment_seed
    apply_algorithm_params(env_args, env_args.algorithm_name)
    _configure_fixed_obstacle_count(FIXED_NUMBER_OF_OBJECTS)

    env = None
    rollout_agent = None
    try:
        print("[Online-AOSA] Launching AirSim training environment")
        env = AirSimEnv(
            takeoff_height=env_args.takeoff_height,
            config=env_args,
            stack_frames=int(env_args.n_frames),
        )
        loaded_number_of_objects = _assert_obstacle_count(
            env, FIXED_NUMBER_OF_OBJECTS
        )
        print(
            "[Online-AOSA] Fixed training-environment obstacles: "
            f"{loaded_number_of_objects}"
        )
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(model_seed)
        set_progress = getattr(env, "set_curriculum_progress", None)
        if callable(set_progress):
            set_progress(float(script_args.curriculum_progress))

        # AirSimEnv has already acquired its initial observation during
        # construction.  Use it only to determine model input dimensions; the
        # first reset belongs to the first recorded rollout attempt below.
        initial_obs = env.get_obs()
        rollout_agent = _build_rollout_agent(
            common_args,
            initial_obs,
            env.action_space,
            model_seed=model_seed,
            checkpoint=vssm_checkpoint,
            device=device,
        )
        print(
            f"[Online-AOSA] Loaded CL-VSSM-SAC rollout actor: "
            f"{vssm_checkpoint}"
        )
        episode, attempts = _collect_successful_episode(
            env,
            rollout_agent,
            first_episode_seed=int(script_args.first_episode_seed),
            max_episodes=int(script_args.max_episodes),
            max_steps=int(getattr(env_args, "episode_length", 300)),
            candidate_steps=int(script_args.candidate_steps),
            curriculum_progress=float(script_args.curriculum_progress),
            number_of_objects=FIXED_NUMBER_OF_OBJECTS,
        )
    finally:
        if env is not None:
            close_env(env, label="online AOSA collection")

    scores = [item.sample.obstacle_proximity for item in episode.steps]
    candidate_indices = _select_spaced_top_indices(
        scores,
        count=int(script_args.candidate_steps),
        min_gap=int(script_args.min_sample_gap),
    )
    selected_samples, source_metadata = _save_online_trajectory(
        episode,
        trajectory_dir,
        candidate_indices=candidate_indices,
        model_seed=model_seed,
        environment_seed=environment_seed,
        curriculum_progress=float(script_args.curriculum_progress),
        number_of_objects=FIXED_NUMBER_OF_OBJECTS,
        checkpoint=vssm_checkpoint,
        attempts=attempts,
    )
    print(
        f"[Online-AOSA] Saved {len(selected_samples)} online candidates: "
        + ", ".join(str(sample.step) for sample in selected_samples)
    )

    del rollout_agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    paired_args = SimpleNamespace(
        model_seed=model_seed,
        reference_run=str(trajectory_dir),
        capture_steps=[sample.step for sample in selected_samples],
        vssm_checkpoint=vssm_checkpoint,
        sac_checkpoint=sac_checkpoint,
        output_dir=str(explanation_dir),
        device=str(device),
        window_height=int(script_args.window_height),
        window_width=int(script_args.window_width),
        stride=int(script_args.stride),
        temporal_radius=int(script_args.temporal_radius),
        reference_depth=float(script_args.reference_depth),
        occlusion_batch_size=int(script_args.occlusion_batch_size),
        motion=str(script_args.motion),
        max_masks=script_args.max_masks,
        overlay_alpha=0.82,
        dpi=int(script_args.dpi),
        skip_signed=True,
        summary_only=True,
        summary_max_steps=int(script_args.summary_steps),
        dry_run=False,
    )
    result_dir = run_paired_explanations(paired_args, common_args)
    if result_dir is None:
        raise RuntimeError("Paired explanation unexpectedly returned no output")

    paired_metadata_path = result_dir / "metadata.json"
    with paired_metadata_path.open("r", encoding="utf-8") as handle:
        paired_metadata = json.load(handle)
    source_metadata["paired_explanation_dir"] = str(result_dir)
    source_metadata["final_figure"] = str(
        result_dir / "trajectory_paired_all_steps_absolute_heatmap.png"
    )
    source_metadata["final_figure_steps"] = paired_metadata.get(
        "trajectory_summary", {}
    ).get("steps_shown", [])
    _write_json(trajectory_dir / "metadata.json", source_metadata)

    print(
        "[Online-AOSA] Final four-step figure: "
        f"{source_metadata['final_figure']}"
    )
    return result_dir


def main(argv=None) -> None:
    script_args, common_args = _parse_args(argv)
    run(script_args, common_args)


if __name__ == "__main__":
    main()
