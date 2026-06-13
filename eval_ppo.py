#!/usr/bin/env python3
"""Evaluate agents trained by main_ppo.py."""

import copy
import os

import torch

from algo_name_utils import expand_algorithm_spec, to_internal_algorithm_name, to_internal_core_algorithm_name
from algorithm.config_loader import apply_algorithm_params
from config import get_config
from eval_common import (
    close_env,
    print_eval_summary,
    resolve_checkpoint,
    run_eval_episodes,
    seeds_from_args,
    set_eval_curriculum_progress,
    write_eval_csv,
)
from main_ppo import create_env_from_name, get_agent_class, _configure_reproducibility


SUPPORTED_PPO_ALGOS = {"PPO", "VMPPO", "PL_VMPPO"}


def _default_checkpoint(algo_name: str) -> str:
    return os.path.join("./models", f"{algo_name}_async_final.pth")


def evaluate_algorithm(base_args, algo_name: str, seed: int) -> None:
    core_algo_name = to_internal_core_algorithm_name(algo_name)
    if core_algo_name not in SUPPORTED_PPO_ALGOS:
        print(f"Skipping {algo_name}: eval_ppo.py only supports PPO, VMPPO, PL_VMPPO.")
        return

    args = copy.deepcopy(base_args)
    args.seed = seed
    args.algorithm_name = algo_name
    apply_algorithm_params(args, algo_name)
    _configure_reproducibility(seed, args)

    is_recurrent = core_algo_name in {"VMPPO", "PL_VMPPO"}
    n_frames = int(args.n_frames)
    env = create_env_from_name(args, n_frames)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        set_eval_curriculum_progress(env)
        obs, _ = env.reset(seed=seed)

        depth_shape = obs["depth"].shape
        model_depth_shape = (1, depth_shape[-2], depth_shape[-1]) if is_recurrent else depth_shape
        if is_recurrent:
            args.depth_shape = model_depth_shape
        base_dim = obs["base"].shape[0]
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        AgentClass = get_agent_class(algo_name)
        agent = AgentClass(base_dim, model_depth_shape, env.action_space, args, device=device, seed=seed)
        checkpoint = resolve_checkpoint(args.load_model, _default_checkpoint(algo_name))
        print(f"[Eval][{algo_name}_seed{seed}] Loading model: {checkpoint}")
        agent.load(checkpoint)

        def prepare(current_obs):
            return current_obs["base"], current_obs["depth"], current_obs.get("distance_sensor")

        label = f"{algo_name}_seed{seed}"
        results = run_eval_episodes(
            env,
            agent,
            args,
            seed=seed,
            is_recurrent=is_recurrent,
            prepare_action_inputs=prepare,
            label=label,
        )
        csv_path = os.path.join("./results", "eval", f"{label}_eval.csv")
        write_eval_csv(results, csv_path)
        print_eval_summary(results, label=label, csv_path=csv_path)
    finally:
        close_env(env, label=f"{algo_name} seed={seed}")


def main():
    base_args = get_config()
    algorithms = [to_internal_algorithm_name(name) for name in expand_algorithm_spec(base_args.algorithm_name)]
    for seed in seeds_from_args(base_args):
        for algo_name in algorithms:
            evaluate_algorithm(base_args, algo_name, seed)


if __name__ == "__main__":
    main()
