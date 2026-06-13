#!/usr/bin/env python3
"""Evaluate agents trained by main_async.py."""

import collections
import copy
import os

import numpy as np
import torch

from algo_name_utils import (
    expand_algorithm_spec,
    is_curriculum_algorithm,
    split_curriculum_prefix,
    to_internal_algorithm_name,
    to_internal_core_algorithm_name,
)
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
from main_async import create_env_from_name, get_agent_class, _configure_reproducibility


RECURRENT_ALGOS = {
    "Mamba_TD3",
    "VMTD3",
    "STV_Patch_TD3",
    "Vim_TD3",
    "ST_Seq_Vim_TD3",
    "STV_Seq_Vim_TD3",
    "DPER_VMTD3",
    "PL_VMTD3",
    "PL_VMSAC",
    "PL_PER_VMSAC",
    "PL_DPER_VMSAC",
    "PL_DPER_VMSAC_Beta",
    "PL_DPER_VMTD3",
    "SAFE_VMTD3",
    "ST_DualVim_TD3",
    "VMSAC",
    "MM_VMSAC",
    "PER_VMSAC",
    "SVMSAC",
    "DPER_SVMSAC",
    "SAFE_VMSAC",
    "VMSAC_Beta",
    "DPER_VMSAC",
    "DPER_VMSAC_Beta",
}

BASE_SEQUENCE_ALGOS = {"ST_Seq_Vim_TD3", "STV_Seq_Vim_TD3", "MM_VMSAC"}
UNSUPPORTED_IN_ASYNC_EVAL = {"PPO", "VMPPO", "PL_VMPPO", "LSTM_SAC"}


def _build_action_input_preparer(initial_base, *, is_recurrent: bool, core_algo_name: str, n_frames: int):
    use_base_sequence = bool(is_recurrent and core_algo_name in BASE_SEQUENCE_ALGOS)
    base_seq_deque = None
    base_seq = None
    def reset_base_sequence(base):
        nonlocal base_seq_deque, base_seq
        if not use_base_sequence:
            return
        base_seq_deque = collections.deque(maxlen=n_frames)
        init_base = np.asarray(base, dtype=np.float32)
        for _ in range(n_frames):
            base_seq_deque.append(init_base.copy())
        base_seq = np.stack(list(base_seq_deque), axis=0).astype(np.float32)

    reset_base_sequence(initial_base)

    def prepare(obs):
        nonlocal base_seq
        depth = obs["depth"]
        base = obs["base"]
        if is_recurrent:
            if depth.ndim == 3:
                depth = np.expand_dims(depth, axis=1)
            actor_base = base_seq if use_base_sequence else base
            return actor_base, depth, None
        return base, depth, None

    def after_step(_obs, next_obs):
        nonlocal base_seq, base_seq_deque
        if not use_base_sequence:
            return
        base_seq_deque.append(np.asarray(next_obs["base"], dtype=np.float32))
        base_seq = np.stack(list(base_seq_deque), axis=0).astype(np.float32)

    def on_episode_reset(obs):
        reset_base_sequence(obs["base"])

    return prepare, after_step, on_episode_reset


def _default_checkpoint(algo_name: str, seed: int) -> str:
    return os.path.join("./models", algo_name, f"seed{seed}", "async_final.pth")


def evaluate_algorithm(base_args, algo_name: str, seed: int) -> None:
    args = copy.deepcopy(base_args)
    args.seed = seed
    args.algorithm_name = algo_name
    apply_algorithm_params(args, algo_name)
    _configure_reproducibility(seed, args)

    core_algo_name = to_internal_core_algorithm_name(algo_name)
    if core_algo_name in UNSUPPORTED_IN_ASYNC_EVAL:
        print(f"Skipping {algo_name}: use eval_ppo.py or eval_lstm_sac.py for this algorithm.")
        return

    if is_curriculum_algorithm(algo_name):
        print(f"[Eval] Curriculum enabled for {split_curriculum_prefix(algo_name)[1]}; using final progress.")
    else:
        print(f"[Eval] Curriculum disabled for {algo_name}.")

    is_recurrent = core_algo_name in RECURRENT_ALGOS
    n_frames = int(args.n_frames)
    env = create_env_from_name(args, n_frames)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        set_eval_curriculum_progress(env)
        obs, _ = env.reset(seed=seed)

        depth_shape = obs["depth"].shape
        model_depth_shape = (1, depth_shape[-2], depth_shape[-1]) if is_recurrent else depth_shape
        base_dim = obs["base"].shape[0]
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        AgentClass = get_agent_class(algo_name)
        agent = AgentClass(base_dim, model_depth_shape, env.action_space, args, device=device, seed=seed)
        checkpoint = resolve_checkpoint(args.load_model, _default_checkpoint(algo_name, seed))
        print(f"[Eval][{algo_name}_seed{seed}] Loading model: {checkpoint}")
        agent.load(checkpoint)

        prepare, after_step, on_episode_reset = _build_action_input_preparer(
            obs["base"],
            is_recurrent=is_recurrent,
            core_algo_name=core_algo_name,
            n_frames=n_frames,
        )
        label = f"{algo_name}_seed{seed}"
        results = run_eval_episodes(
            env,
            agent,
            args,
            seed=seed,
            is_recurrent=is_recurrent,
            prepare_action_inputs=prepare,
            after_step=after_step,
            on_episode_reset=on_episode_reset,
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
