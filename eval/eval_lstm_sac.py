#!/usr/bin/env python3
"""Evaluate agents trained by train_lstm_sac.py."""

import copy
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithm.config_loader import apply_algorithm_params
from algorithm.LSTM_SAC.agent import LSTMSACAgent
from config import get_config
from eval.eval_common import (
    close_env,
    print_eval_summary,
    resolve_checkpoint,
    run_eval_episodes,
    seeds_from_args,
)
from eval.eval_env import SceneEvalAirSimEnv
from main_async import _configure_reproducibility


def _default_checkpoint(seed: int) -> str:
    return os.path.join("./models", f"LSTM_SAC_seed{seed}", "async_final.pth")


def evaluate_seed(base_args, seed: int) -> None:
    args = copy.deepcopy(base_args)
    args.seed = seed
    args.algorithm_name = "LSTM_SAC"
    apply_algorithm_params(args, "LSTM_SAC")
    args.n_frames = 1
    _configure_reproducibility(seed, args)

    env = SceneEvalAirSimEnv(takeoff_height=args.takeoff_height, config=args, stack_frames=1)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)

        depth_shape = obs["depth"].shape
        model_depth_shape = (1, depth_shape[-2], depth_shape[-1])
        base_dim = obs["base"].shape[0]
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        agent = LSTMSACAgent(base_dim, model_depth_shape, env.action_space, args, device=device, seed=seed)
        checkpoint = resolve_checkpoint(args.load_model, _default_checkpoint(seed))
        print(f"[Eval][LSTM_SAC_seed{seed}] Loading model: {checkpoint}")
        agent.load(checkpoint)

        def prepare(current_obs):
            depth_seq = current_obs["depth"]
            if depth_seq.ndim == 3:
                depth_seq = np.expand_dims(depth_seq, axis=1)
            return current_obs["base"], depth_seq, None

        label = f"LSTM_SAC_seed{seed}"
        csv_path = os.path.join("./results", "eval", f"{label}_eval.csv")
        results = run_eval_episodes(
            env,
            agent,
            args,
            seed=seed,
            is_recurrent=True,
            prepare_action_inputs=prepare,
            label=label,
            csv_path=csv_path,
        )
        print_eval_summary(results, label=label, csv_path=csv_path)
    finally:
        close_env(env, label=f"LSTM_SAC seed={seed}")


def main():
    base_args = get_config()
    for seed in seeds_from_args(base_args):
        evaluate_seed(base_args, seed)


if __name__ == "__main__":
    main()
