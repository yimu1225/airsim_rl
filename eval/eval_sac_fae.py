#!/usr/bin/env python3
"""Evaluate SAC_FAE checkpoints produced by main_async.py."""

from __future__ import annotations

import copy
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithm.SAC_FAE.agent import SACFAEAgent
from algorithm.config_loader import apply_algorithm_params
from algo_name_utils import to_internal_core_algorithm_name, to_output_algorithm_name
from config import get_config
from eval.eval_common import (
    close_env,
    print_eval_summary,
    resolve_checkpoint,
    run_eval_episodes,
    seeds_from_args,
)
from eval.eval_env import SceneEvalAirSimEnv


def _configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _default_checkpoint(algorithm_name: str, seed: int) -> str:
    return os.path.join(
        "./models",
        to_output_algorithm_name(algorithm_name),
        f"seed{seed}",
        "async_final.pth",
    )


def evaluate_seed(base_args, seed: int) -> None:
    args = copy.deepcopy(base_args)
    apply_algorithm_params(args, args.algorithm_name)
    args.seed = int(seed)
    _configure_reproducibility(args.seed, bool(args.cuda_deterministic))

    env = SceneEvalAirSimEnv(takeoff_height=args.takeoff_height, config=args, stack_frames=4)
    output_name = to_output_algorithm_name(args.algorithm_name)
    label = f"{output_name}_seed{args.seed}"
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(args.seed)
        obs, _ = env.reset(seed=args.seed)
        depth_shape = tuple(int(value) for value in obs["depth"].shape)
        if depth_shape != (4, 128, 128):
            raise ValueError(
                f"SAC_FAE evaluation requires depth shape (4,128,128), got {depth_shape}."
            )

        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        agent = SACFAEAgent(
            obs["base"].shape[0],
            depth_shape,
            env.action_space,
            args,
            device=device,
            seed=args.seed,
        )
        checkpoint = resolve_checkpoint(
            args.load_model,
            _default_checkpoint(args.algorithm_name, args.seed),
        )
        print(f"[Eval][{label}] Loading model: {checkpoint}")
        agent.load(checkpoint)

        def prepare(current_obs):
            return current_obs["base"], current_obs["depth"], None

        csv_path = os.path.join("./results", "eval", f"{label}_eval.csv")
        results = run_eval_episodes(
            env,
            agent,
            args,
            seed=args.seed,
            is_recurrent=False,
            prepare_action_inputs=prepare,
            label=label,
            csv_path=csv_path,
        )
        print_eval_summary(results, label=label, csv_path=csv_path)
    finally:
        close_env(env, label=label)


def main(argv=None) -> None:
    args = get_config(argv)
    try:
        core_name = to_internal_core_algorithm_name(args.algorithm_name)
    except ValueError:
        core_name = ""
    if core_name != "SAC_FAE":
        args.algorithm_name = "SAC_FAE"
    for seed in seeds_from_args(args):
        evaluate_seed(args, int(seed))


if __name__ == "__main__":
    main()
