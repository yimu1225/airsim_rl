#!/usr/bin/env python3
"""Evaluate trained policies in the training scene at fixed obstacle densities.

The three default test tiers contain exactly 160, 180, and 200 static
obstacles.  Environment layouts are generated from a dedicated layout seed,
not from a model's training seed.  Consequently, episode N has the same UE4
seed, obstacle count/types/locations, arena settings, and goal for every
algorithm.  A persistent manifest detects accidental layout drift between
separate evaluation runs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algo_name_utils import (
    expand_algorithm_spec,
    to_internal_algorithm_name,
    to_internal_core_algorithm_name,
    to_output_algorithm_name,
)
from algorithm.LSTM_SAC.agent import LSTMSACAgent
from algorithm.config_loader import apply_algorithm_params
from common.utils import airsimize_coordinates
from config import get_config
from eval.eval_async import RECURRENT_ALGOS, _build_action_input_preparer
from eval.eval_common import (
    close_env,
    print_eval_summary,
    resolve_checkpoint,
    run_eval_episodes,
    seeds_from_args,
)
from gym_airsim.envs.AirGym import AirSimEnv
from main_async import _configure_reproducibility, get_agent_class as get_async_agent_class
from main_ppo import get_agent_class as get_ppo_agent_class
from settings_folder import settings


DEFAULT_OBSTACLE_COUNTS = (160, 180, 200)
DEFAULT_LAYOUT_SEED = 20260722
DEFAULT_RESULTS_ROOT = os.path.join(
    REPO_ROOT, "results", "eval", "training_density_path_length"
)
LEGACY_RESULTS_ROOT = os.path.join(REPO_ROOT, "results", "eval", "training_density")
PPO_ALGORITHMS = {"PPO", "VSSM_PPO", "PL_VSSM_PPO"}

# Sample every layout-defining field on every episode.  UE4 uses Seed together
# with these values to deterministically choose obstacle types and positions.
LAYOUT_SAMPLE_VARS = (
    "Seed",
    "ArenaSize",
    "NumberOfObjects",
    "NumberOfDynamicObjects",
    "End",
    "Walls1",
    "MinimumDistance",
    "VelocityRange",
    "EnvType",
    "PlayerStart",
    "Name",
)

LAYOUT_MANIFEST_KEYS = LAYOUT_SAMPLE_VARS


@dataclass(frozen=True)
class AlgorithmProfile:
    algorithm_name: str
    core_name: str
    is_lstm: bool
    is_ppo: bool
    is_recurrent: bool


def _algorithm_profile(algorithm_name: str) -> AlgorithmProfile:
    core_name = to_internal_core_algorithm_name(algorithm_name)
    is_lstm = core_name == "LSTM_SAC"
    is_ppo = core_name in PPO_ALGORITHMS
    return AlgorithmProfile(
        algorithm_name=algorithm_name,
        core_name=core_name,
        is_lstm=is_lstm,
        is_ppo=is_ppo,
        is_recurrent=(
            is_lstm
            or core_name in RECURRENT_ALGOS
            or core_name in {"VSSM_PPO", "PL_VSSM_PPO"}
        ),
    )


def _layout_config(env: AirSimEnv) -> Dict:
    """Return the UE4 inputs that fully identify one generated layout."""
    return {
        key: copy.deepcopy(env.game_config_handler.get_cur_item(key))
        for key in LAYOUT_MANIFEST_KEYS
    }


def _layout_digest(config: Dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LayoutManifest:
    """Persist and validate the common episode layouts used by all policies."""

    def __init__(self, path: str, *, obstacle_count: int, layout_seed: int) -> None:
        self.path = path
        self.obstacle_count = int(obstacle_count)
        self.layout_seed = int(layout_seed)
        self.data = {
            "format_version": 1,
            "layout_seed": self.layout_seed,
            "obstacle_count": self.obstacle_count,
            "episodes": {},
        }
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if int(loaded.get("layout_seed", -1)) != self.layout_seed:
                raise ValueError(f"Layout manifest seed mismatch: {path}")
            if int(loaded.get("obstacle_count", -1)) != self.obstacle_count:
                raise ValueError(f"Layout manifest obstacle-count mismatch: {path}")
            self.data = loaded

    def record_or_validate(self, episode: int, config: Dict) -> None:
        episode_key = str(int(episode))
        record = {"digest": _layout_digest(config), "config": config}
        previous = self.data.setdefault("episodes", {}).get(episode_key)
        if previous is not None and previous != record:
            raise RuntimeError(
                "Environment layout drift detected for "
                f"obstacles={self.obstacle_count}, episode={episode}. "
                f"Expected {previous.get('digest')}, got {record['digest']}. "
                "Use the same code/settings or choose a new --layout_seed."
            )
        if previous is None:
            self.data["episodes"][episode_key] = record
            self._write()

    def _write(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary_path = f"{self.path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, self.path)


class FixedDensityTrainingAirSimEnv(AirSimEnv):
    """Training-scene environment with one deterministic layout per episode."""

    def __init__(
        self,
        *,
        takeoff_height: float,
        config,
        stack_frames: int,
        obstacle_count: int,
        layout_seed: int,
        manifest: LayoutManifest,
    ) -> None:
        self.fixed_obstacle_count = int(obstacle_count)
        if self.fixed_obstacle_count <= 0:
            raise ValueError("obstacle_count must be positive")
        self.layout_seed = int(layout_seed)
        self.layout_manifest = manifest
        self._next_layout_episode = 0
        self._expected_layout = None
        self._active_layout_episode = None

        # AirSimEnv selects its range from settings by level before UE4 starts.
        # Temporarily replace the hard-level range so its initial JSON already
        # contains the requested fixed density, then restore global settings.
        original_hard_range = settings.hard_range_dic
        density_range = copy.deepcopy(original_hard_range)
        density_range["NumberOfObjects"] = [self.fixed_obstacle_count]
        density_range["NumberOfDynamicObjects"] = [0]

        env_config = copy.deepcopy(config)
        env_config.seed = self.layout_seed
        env_config.non_curriculum_level = 2
        env_config.algorithm_name = to_internal_core_algorithm_name(config.algorithm_name)
        env_config.enable_takeoff_obstacle_check = False

        settings.hard_range_dic = density_range
        try:
            super().__init__(
                takeoff_height=takeoff_height,
                config=env_config,
                stack_frames=stack_frames,
            )
        finally:
            settings.hard_range_dic = original_hard_range

        # Keep the copied handler fixed even after the temporary global override
        # has been restored.
        self.game_config_handler.set_range(("NumberOfObjects", [self.fixed_obstacle_count]))
        self.game_config_handler.set_range(("NumberOfDynamicObjects", [0]))
        self.base_seed = self.layout_seed
        self.use_curriculum = False

    def randomize_env(self) -> bool:
        episode = self._next_layout_episode
        self.game_config_handler.sample(
            *LAYOUT_SAMPLE_VARS,
            change_counter=episode,
            base_seed=self.layout_seed,
        )
        self.goal = airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
        self.change_counter = episode
        self._active_layout_episode = episode
        self._expected_layout = _layout_config(self)
        self._next_layout_episode += 1
        print(
            f"[DensityEval] layout episode={episode}, "
            f"obstacles={self.fixed_obstacle_count}, "
            f"digest={_layout_digest(self._expected_layout)[:12]}"
        )
        return True

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        actual_layout = _layout_config(self)
        if actual_layout != self._expected_layout:
            raise RuntimeError(
                "The UE4 layout configuration changed during reset (usually after a "
                "takeoff-recovery retry), so algorithm fairness cannot be guaranteed."
            )
        if int(actual_layout["NumberOfObjects"]) != self.fixed_obstacle_count:
            raise RuntimeError(
                f"Expected {self.fixed_obstacle_count} obstacles, got "
                f"{actual_layout['NumberOfObjects']}."
            )
        self.layout_manifest.record_or_validate(self._active_layout_episode, actual_layout)
        info = dict(info or {})
        info.update(
            {
                "layout_episode": self._active_layout_episode,
                "layout_digest": _layout_digest(actual_layout),
                "obstacle_count": self.fixed_obstacle_count,
            }
        )
        return observation, info


@dataclass
class DensitySummary:
    obstacle_count: int
    episodes: int
    mean_reward: float
    mean_path_length: float
    success_rate: float
    collision_rate: float
    episode_csv: str


def _default_checkpoint(profile: AlgorithmProfile, seed: int) -> str:
    if profile.is_lstm:
        return os.path.join(REPO_ROOT, "models", f"LSTM_SAC_seed{seed}", "async_final.pth")
    return os.path.join(
        REPO_ROOT, "models",
        to_output_algorithm_name(profile.algorithm_name),
        f"seed{seed}",
        "async_final.pth",
    )


def _manifest_path(manifest_root: str, layout_seed: int, obstacle_count: int) -> str:
    return os.path.join(
        manifest_root,
        f"layout_seed_{layout_seed}",
        f"obstacles_{obstacle_count}.json",
    )


def _summary_from_results(obstacle_count: int, results, episode_csv: str) -> DensitySummary:
    successful_path_lengths = [row.path_length for row in results if row.success]
    return DensitySummary(
        obstacle_count=int(obstacle_count),
        episodes=len(results),
        mean_reward=float(np.mean([row.reward for row in results])),
        mean_path_length=(
            float(np.mean(successful_path_lengths))
            if successful_path_lengths
            else float("nan")
        ),
        success_rate=float(np.mean([row.success for row in results])),
        collision_rate=float(np.mean([row.碰撞率 for row in results])),
        episode_csv=episode_csv,
    )


def _write_summary(path: str, summaries: Sequence[DensitySummary]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "obstacle_count",
                "episodes",
                "mean_reward",
                "mean_path_length",
                "success_rate",
                "collision_rate",
                "episode_csv",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.obstacle_count,
                    row.episodes,
                    row.mean_reward,
                    (
                        f"{row.mean_path_length:.2f}"
                        if np.isfinite(row.mean_path_length)
                        else "N/A"
                    ),
                    row.success_rate,
                    row.collision_rate,
                    row.episode_csv,
                ]
            )


def _build_agent_and_preparers(args, env, initial_obs, profile: AlgorithmProfile, seed: int):
    n_frames = int(args.n_frames)

    depth_shape = tuple(int(value) for value in initial_obs["depth"].shape)
    model_depth_shape = (
        (1, depth_shape[-2], depth_shape[-1])
        if profile.is_recurrent
        else depth_shape
    )
    if profile.core_name == "SAC_FAE" and depth_shape != (4, 128, 128):
        raise ValueError(f"SAC_FAE evaluation requires depth shape (4,128,128), got {depth_shape}.")
    if profile.is_ppo and profile.is_recurrent:
        args.depth_shape = model_depth_shape

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if profile.is_lstm:
        agent_class = LSTMSACAgent

        def prepare(current_obs):
            depth = current_obs["depth"]
            if depth.ndim == 3:
                depth = np.expand_dims(depth, axis=1)
            return current_obs["base"], depth, None

        after_step = None
        on_episode_reset = None
    elif profile.is_ppo:
        agent_class = get_ppo_agent_class(profile.algorithm_name)

        def prepare(current_obs):
            return current_obs["base"], current_obs["depth"], current_obs.get("distance_sensor")

        after_step = None
        on_episode_reset = None
    else:
        agent_class = get_async_agent_class(profile.algorithm_name)
        prepare, after_step, on_episode_reset = _build_action_input_preparer(
            initial_obs["base"],
            is_recurrent=profile.is_recurrent,
            core_algo_name=profile.core_name,
            n_frames=n_frames,
        )
    agent = agent_class(
        initial_obs["base"].shape[0],
        model_depth_shape,
        env.action_space,
        args,
        device=device,
        seed=seed,
    )
    return agent, profile.is_recurrent, prepare, after_step, on_episode_reset


def evaluate_density(
    base_args,
    algorithm_name: str,
    model_seed: int,
    obstacle_count: int,
    *,
    layout_seed: int,
    manifest_root: str,
    results_root: str,
) -> DensitySummary:
    profile = _algorithm_profile(algorithm_name)
    args = copy.deepcopy(base_args)
    args.algorithm_name = algorithm_name
    args.seed = int(model_seed)
    apply_algorithm_params(args, algorithm_name)
    if profile.is_lstm:
        args.n_frames = 1
    _configure_reproducibility(model_seed, args)

    display_name = to_output_algorithm_name(algorithm_name)
    label = f"{display_name}_seed{model_seed}_obstacles{obstacle_count}"
    manifest = LayoutManifest(
        _manifest_path(manifest_root, layout_seed, obstacle_count),
        obstacle_count=obstacle_count,
        layout_seed=layout_seed,
    )
    env = FixedDensityTrainingAirSimEnv(
        takeoff_height=args.takeoff_height,
        config=args,
        stack_frames=int(args.n_frames),
        obstacle_count=obstacle_count,
        layout_seed=layout_seed,
        manifest=manifest,
    )
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(layout_seed)

        # AirSimEnv already acquired an initial observation in __init__.  Using
        # it for shape discovery avoids consuming episode layout 0 before the
        # actual 100-episode evaluation starts.
        initial_obs = env.init_state
        agent, is_recurrent, prepare, after_step, on_episode_reset = _build_agent_and_preparers(
            args,
            env,
            initial_obs,
            profile,
            model_seed,
        )
        checkpoint = resolve_checkpoint(
            args.load_model,
            _default_checkpoint(profile, model_seed),
        )
        print(f"[DensityEval][{label}] Loading model: {checkpoint}")
        agent.load(checkpoint)

        output_dir = os.path.join(
            results_root,
            display_name,
            f"seed{model_seed}",
        )
        episode_csv = os.path.join(output_dir, f"obstacles_{obstacle_count}_episodes.csv")
        results = run_eval_episodes(
            env,
            agent,
            args,
            seed=layout_seed,
            is_recurrent=is_recurrent,
            prepare_action_inputs=prepare,
            after_step=after_step,
            on_episode_reset=on_episode_reset,
            label=label,
            csv_path=episode_csv,
        )
        print_eval_summary(results, label=label, csv_path=episode_csv)
        return _summary_from_results(obstacle_count, results, episode_csv)
    finally:
        close_env(env, label=label)


def _parse_args(argv=None):
    density_parser = argparse.ArgumentParser(add_help=False)
    density_parser.add_argument(
        "--obstacle_counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_OBSTACLE_COUNTS),
        help="固定静态障碍物数量档位（默认: 160 180 200）",
    )
    density_parser.add_argument(
        "--layout_seed",
        type=int,
        default=DEFAULT_LAYOUT_SEED,
        help="所有算法共享的环境布局种子",
    )
    density_parser.add_argument(
        "--results_root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help="新评估结果根目录；禁止指向旧 training_density 目录",
    )
    density_parser.add_argument(
        "--layout_manifest_dir",
        type=str,
        default=None,
        help="跨算法复用并校验环境布局的 manifest 目录",
    )
    density_args, remaining = density_parser.parse_known_args(argv)
    base_args = get_config(remaining)
    counts = [int(value) for value in density_args.obstacle_counts]
    if not counts or any(value <= 0 for value in counts):
        density_parser.error("--obstacle_counts must contain positive integers")
    if len(set(counts)) != len(counts):
        density_parser.error("--obstacle_counts must not contain duplicates")
    results_root = os.path.abspath(density_args.results_root)
    legacy_results_root = os.path.abspath(LEGACY_RESULTS_ROOT)
    if os.path.normcase(results_root) == os.path.normcase(legacy_results_root):
        density_parser.error(
            "--results_root points to the legacy training_density directory; "
            "choose a new directory to avoid overwriting existing evaluation data"
        )
    manifest_root = (
        os.path.abspath(density_args.layout_manifest_dir)
        if density_args.layout_manifest_dir
        else os.path.join(results_root, "manifests")
    )
    return (
        base_args,
        counts,
        int(density_args.layout_seed),
        manifest_root,
        results_root,
    )


def main(argv=None) -> None:
    base_args, obstacle_counts, layout_seed, manifest_root, results_root = _parse_args(argv)
    algorithms = [
        to_internal_algorithm_name(name)
        for name in expand_algorithm_spec(base_args.algorithm_name)
    ]
    model_seeds = seeds_from_args(base_args)
    if base_args.load_model and len(algorithms) * len(model_seeds) > 1:
        raise ValueError(
            "--load_model can only be used with one algorithm and one model seed. "
            "For comparisons, omit it so each default checkpoint is selected automatically."
        )

    print(
        "[DensityEval] Shared test protocol: "
        f"obstacles={obstacle_counts}, episodes={base_args.eval_episodes}, "
        f"layout_seed={layout_seed}, results_root={results_root}"
    )
    for model_seed in model_seeds:
        for algorithm_name in algorithms:
            summaries: List[DensitySummary] = []
            for obstacle_count in obstacle_counts:
                summaries.append(
                    evaluate_density(
                        base_args,
                        algorithm_name,
                        int(model_seed),
                        obstacle_count,
                        layout_seed=layout_seed,
                        manifest_root=manifest_root,
                        results_root=results_root,
                    )
                )
                display_name = to_output_algorithm_name(algorithm_name)
                summary_path = os.path.join(
                    results_root,
                    display_name,
                    f"seed{model_seed}",
                    "summary.csv",
                )
                _write_summary(summary_path, summaries)
            print(f"[DensityEval] Four-metric summary saved to {summary_path}")


if __name__ == "__main__":
    main()
