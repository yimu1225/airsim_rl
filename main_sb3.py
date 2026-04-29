#!/usr/bin/env python3
"""SB3-based training entry point for migrated AirSim algorithms."""

from __future__ import annotations

import random
import sys
from pathlib import Path
import copy

import numpy as np
import torch as th
from torch import nn

import gymnasium as gym
import gym_airsim  # noqa: F401 - register AirSim env ids
from gym_airsim.envs import AirSimEnv
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from algo_name_utils import expand_algorithm_spec, split_curriculum_prefix, to_internal_core_algorithm_name
from config import get_config
from sb3_algorithms import (
    AsymTD3,
    DualVimTD3,
    LSTMSAC,
    MambaTD3,
    PERAsymTD3,
    PERSAC,
    PERSTVimSAC,
    PERSTVimTD3,
    PERTD3,
    STSeqVimTD3,
    STSVimTD3,
    STVSeqVimTD3,
    STVimAsymTD3,
    STVimPPO,
    STVimSAC,
    STVimTD3,
    VimPatchTD3,
    VimTD3,
)
from sb3_algorithms.config_loader import apply_algorithm_params
from sb3_extensions.callbacks import (
    AirSimHealthCallback,
    CSVLoggerCallback,
    DepthRenderCallback,
)
from sb3_extensions.feature_extractors import (
    AirSimCNNExtractor,
    DualVimFeatureExtractor,
    LSTMExtractor,
    MambaExtractor,
    STSeqVimFeatureExtractor,
    STSVimFeatureExtractor,
    STVSeqVimFeatureExtractor,
    STVimFeatureExtractor,
    VimFeatureExtractor,
    VimPatchFeatureExtractor,
)


SUPPORTED_ALGORITHMS = {
    "TD3": TD3,
    "DDPG": DDPG,
    "SAC": SAC,
    "PPO": PPO,
    "PER_TD3": PERTD3,
    "PER_TD3_asym": PERAsymTD3,
    "TD3_asym": AsymTD3,
    "ST_Vim_TD3": STVimTD3,
    "STV_Patch_TD3": VimPatchTD3,
    "Vim_TD3": VimTD3,
    "ST_Seq_Vim_TD3": STSeqVimTD3,
    "STV_Seq_Vim_TD3": STVSeqVimTD3,
    "PER_ST_Vim_TD3": PERSTVimTD3,
    "ST_SVim_TD3": STSVimTD3,
    "Mamba_TD3": MambaTD3,
    "ST_DualVim_TD3": DualVimTD3,
    "LSTM_SAC": LSTMSAC,
    "ST_Vim_SAC": STVimSAC,
    "PER_ST_Vim_SAC": PERSTVimSAC,
    "ST_Vim_PPO": STVimPPO,
    "ST_Vim_TD3_asym": STVimAsymTD3,
}

NORMALIZED_TO_CANONICAL_ALGORITHM = {name.lower(): name for name in SUPPORTED_ALGORITHMS}


FEATURE_EXTRACTORS = {
    "TD3": AirSimCNNExtractor,
    "DDPG": AirSimCNNExtractor,
    "SAC": AirSimCNNExtractor,
    "PPO": AirSimCNNExtractor,
    "PER_TD3": AirSimCNNExtractor,
    "PER_TD3_asym": AirSimCNNExtractor,
    "TD3_asym": AirSimCNNExtractor,
    "Vim_TD3": VimFeatureExtractor,
    "ST_Vim_TD3": STVimFeatureExtractor,
    "PER_ST_Vim_TD3": STVimFeatureExtractor,
    "ST_Vim_TD3_asym": STVimFeatureExtractor,
    "ST_Vim_SAC": STVimFeatureExtractor,
    "PER_ST_Vim_SAC": STVimFeatureExtractor,
    "ST_Vim_PPO": STVimFeatureExtractor,
    "ST_Seq_Vim_TD3": STSeqVimFeatureExtractor,
    "STV_Seq_Vim_TD3": STVSeqVimFeatureExtractor,
    "STV_Patch_TD3": VimPatchFeatureExtractor,
    "ST_SVim_TD3": STSVimFeatureExtractor,
    "Mamba_TD3": MambaExtractor,
    "ST_DualVim_TD3": DualVimFeatureExtractor,
    "LSTM_SAC": LSTMExtractor,
}

PER_ALGORITHMS = {"PER_TD3", "PER_TD3_asym", "PER_ST_Vim_TD3", "PER_ST_Vim_SAC"}
SAC_ALGORITHMS = {"SAC", "LSTM_SAC", "ST_Vim_SAC", "PER_ST_Vim_SAC"}
PPO_ALGORITHMS = {"PPO", "ST_Vim_PPO"}


class _EpisodePrintCallback(BaseCallback):
    """Minimal main_async-style episode print with the active algorithm name."""

    def __init__(self, algorithm_name: str) -> None:
        super().__init__()
        self.algorithm_name = str(algorithm_name)
        self.episode_num = 0
        self.episode_reward = 0.0
        self.episode_length = 0

    def _read_env_attr(self, name: str, default):
        try:
            values = self.training_env.get_attr(name)
            return values[0] if values else default
        except Exception:
            return default

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        if rewards.size == 0 or dones.size == 0:
            return True

        self.episode_reward += float(rewards[0])
        self.episode_length += 1
        if not bool(dones[0]):
            return True

        success_deque = self._read_env_attr("success_deque", [])
        success_rate = float(sum(success_deque) / len(success_deque)) if len(success_deque) > 0 else 0.0
        level = int(self._read_env_attr("level", 0))
        success_count = int(self._read_env_attr("success_count", 0))
        print(
            f"[{self.algorithm_name}] Episode {self.episode_num}, "
            f"Reward: {self.episode_reward:.2f}, Length: {self.episode_length}, "
            f"Success Rate: {success_rate:.3f}, Level: {level}, "
            f"Total Timesteps: {self.num_timesteps}, Total Successes: {success_count}"
        )
        self.episode_num += 1
        self.episode_reward = 0.0
        self.episode_length = 0
        return True


def configure_reproducibility(seed: int, cuda_deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    th.backends.cudnn.benchmark = not cuda_deterministic
    th.backends.cudnn.deterministic = cuda_deterministic
    th.backends.cuda.matmul.allow_tf32 = not cuda_deterministic
    th.backends.cudnn.allow_tf32 = not cuda_deterministic
    th.use_deterministic_algorithms(cuda_deterministic, warn_only=True)


def create_env_from_name(args, n_frames: int):
    env_name = str(getattr(args, "env_name", "AirSimEnv-v42")).strip()
    env_aliases = {
        "AirSimEnv-v42": AirSimEnv,
    }
    env_cls = env_aliases.get(env_name)
    if env_cls is not None:
        return env_cls(takeoff_height=args.takeoff_height, config=args, stack_frames=n_frames)

    try:
        return gym.make(
            env_name,
            takeoff_height=args.takeoff_height,
            config=args,
            stack_frames=n_frames,
        )
    except Exception as exc:
        supported_aliases = ", ".join(sorted(env_aliases))
        raise ValueError(
            f"Unsupported --env_name '{env_name}'. Supported aliases: {supported_aliases}; "
            "or pass a registered gymnasium env id."
        ) from exc


def _canonical_core_algorithm_name(algorithm_name: str) -> str:
    internal_name = to_internal_core_algorithm_name(algorithm_name)
    return NORMALIZED_TO_CANONICAL_ALGORITHM[internal_name.lower()]


def _canonical_algorithm_name(algorithm_name: str) -> str:
    use_curriculum, _ = split_curriculum_prefix(algorithm_name)
    core_name = _canonical_core_algorithm_name(algorithm_name)
    return f"CL-{core_name}" if use_curriculum else core_name


def _resolve_algorithms(args) -> list[str]:
    algorithm_names = [_canonical_algorithm_name(name) for name in expand_algorithm_spec(args.algorithm_name)]
    supported = set(SUPPORTED_ALGORITHMS)
    unsupported = [name for name in algorithm_names if _canonical_core_algorithm_name(name) not in supported]
    if unsupported:
        raise ValueError(
            f"Unsupported SB3 algorithm(s): {unsupported}. "
            f"Supported migrated algorithms: {', '.join(sorted(supported))}."
        )
    return algorithm_names


def _resolve_seed(seed_value) -> int:
    if isinstance(seed_value, list):
        if len(seed_value) != 1:
            raise ValueError("main_sb3.py currently runs one seed at a time. Pass a single --seed value.")
        return int(seed_value[0])
    return int(seed_value)


def _resolve_gradient_steps(args) -> int:
    gradient_steps = float(getattr(args, "gradient_steps", 1.0))
    if gradient_steps < 0:
        return int(gradient_steps)
    steps_per_update = max(1, int(getattr(args, "steps_per_update", 1)))
    return max(1, int(round(gradient_steps * steps_per_update)))


def _device(args) -> str:
    return "cuda" if bool(args.cuda) and th.cuda.is_available() else "cpu"


def _policy_kwargs(algo_name: str, args) -> dict:
    extractor_cls = FEATURE_EXTRACTORS[algo_name]
    extractor_kwargs = dict(
        features_dim=int(args.hidden_dim),
        base_feature_dim=int(args.base_feature_dim),
        algorithm_params=dict(getattr(args, "algorithm_params", {}) or {}),
    )
    if extractor_cls is AirSimCNNExtractor:
        extractor_kwargs["cnn_type"] = "nature"

    return dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=[int(args.hidden_dim), int(args.hidden_dim)],
        activation_fn=nn.ReLU,
        normalize_images=False,
    )


def _action_noise(env, args):
    action_dim = int(np.prod(env.action_space.shape))
    return NormalActionNoise(
        mean=np.zeros(action_dim, dtype=np.float32),
        sigma=np.full(action_dim, float(args.exploration_noise), dtype=np.float32),
    )


def _per_kwargs(args) -> dict:
    return dict(
        per_alpha=float(getattr(args, "per_td3_alpha", getattr(args, "per_alpha", 0.6))),
        per_beta=float(getattr(args, "per_td3_beta_start", getattr(args, "per_beta0", 0.4))),
        per_eps=float(getattr(args, "per_td3_priority_eps", getattr(args, "per_eps", 1e-6))),
    )


def _off_policy_common_kwargs(algo_name: str, env, args) -> dict:
    kwargs = dict(
        policy_kwargs=_policy_kwargs(algo_name, args),
        learning_rate=float(getattr(args, "lr", args.actor_lr)),
        buffer_size=int(args.buffer_size),
        learning_starts=int(args.learning_starts),
        batch_size=int(args.batch_size),
        tau=float(args.tau),
        gamma=float(args.gamma),
        train_freq=(int(args.steps_per_update), "step"),
        gradient_steps=_resolve_gradient_steps(args),
        tensorboard_log="./results/",
        verbose=1,
        seed=int(args.seed),
        device=_device(args),
    )
    if algo_name in PER_ALGORITHMS:
        kwargs.update(_per_kwargs(args))
    if algo_name in SAC_ALGORITHMS:
        kwargs.update(
            ent_coef=getattr(args, "ent_coef", "auto"),
            target_entropy=getattr(args, "target_entropy", "auto"),
            target_update_interval=int(getattr(args, "target_update_interval", 1)),
        )
    else:
        kwargs["action_noise"] = _action_noise(env, args)
    return kwargs


def build_off_policy_model(algo_name: str, env, args):
    algo_class = SUPPORTED_ALGORITHMS[algo_name]
    kwargs = _off_policy_common_kwargs(algo_name, env, args)
    if issubclass(algo_class, TD3) and algo_class is not DDPG:
        kwargs.update(
            policy_delay=int(args.policy_freq),
            target_policy_noise=float(args.policy_noise),
            target_noise_clip=float(args.noise_clip),
        )
    return algo_class(
        "MultiInputPolicy",
        env,
        **kwargs,
    )


def build_ppo_model(algo_name: str, env, args):
    algo_class = SUPPORTED_ALGORITHMS[algo_name]
    return algo_class(
        "MultiInputPolicy",
        env,
        policy_kwargs=_policy_kwargs(algo_name, args),
        learning_rate=float(getattr(args, "lr", args.actor_lr)),
        n_steps=int(getattr(args, "rollout_buffer_size", 2048)),
        batch_size=int(getattr(args, "ppo_batch_size", args.batch_size)),
        n_epochs=int(getattr(args, "ppo_epochs", 10)),
        gamma=float(args.gamma),
        gae_lambda=float(getattr(args, "gae_lambda", 0.95)),
        clip_range=float(getattr(args, "clip_range", 0.2)),
        clip_range_vf=getattr(args, "clip_range_vf", None),
        normalize_advantage=bool(getattr(args, "normalize_advantage", True)),
        ent_coef=float(getattr(args, "ent_coef", 0.0)),
        vf_coef=float(getattr(args, "vf_coef", 0.5)),
        max_grad_norm=float(getattr(args, "max_grad_norm", getattr(args, "grad_clip", 0.5))),
        target_kl=getattr(args, "target_kl", None),
        tensorboard_log="./results/",
        verbose=1,
        seed=int(args.seed),
        device=_device(args),
    )


def get_model(algo_name: str, env, args):
    if algo_name in PPO_ALGORITHMS:
        return build_ppo_model(algo_name, env, args)
    return build_off_policy_model(algo_name, env, args)


def _default_td3_when_algorithm_omitted(argv: list[str]) -> list[str]:
    if any(arg == "--algorithm_name" or arg.startswith("--algorithm_name=") for arg in argv):
        return argv
    return [*argv, "--algorithm_name", "TD3"]


def _build_callbacks(args, run_name: str, checkpoint_dir: Path):
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, int(args.eval_freq)),
            save_path=str(checkpoint_dir),
            name_prefix=run_name,
        ),
        _EpisodePrintCallback(algorithm_name=str(args.algorithm_name)),
    ]
    callbacks.extend(
        [
            AirSimHealthCallback(check_freq=max(1, int(getattr(args, "steps_per_update", 100)))),
            CSVLoggerCallback(Path("results") / run_name / "progress.csv", log_freq=max(1, int(args.log_interval))),
        ]
    )
    if bool(getattr(args, "render_window", False)):
        callbacks.append(
            DepthRenderCallback(
                render_freq=max(1, int(args.log_interval)),
                scale=float(getattr(args, "depth_view_scale", 2.5)),
            )
        )
    return callbacks


def run_one(base_args, algorithm_name: str, seed: int) -> None:
    args = copy.deepcopy(base_args)
    args.algorithm_name = _canonical_algorithm_name(algorithm_name)
    use_curriculum, core_without_cl = split_curriculum_prefix(args.algorithm_name)
    params_path, param_keys = apply_algorithm_params(args, core_without_cl)
    args.seed = int(seed)
    algo_name = _canonical_core_algorithm_name(args.algorithm_name)
    args.use_curriculum = bool(use_curriculum)
    args.core_algorithm_name = algo_name

    configure_reproducibility(int(args.seed), bool(args.cuda_deterministic))

    run_name = f"{args.algorithm_name}_seed{args.seed}"
    args.run_name = run_name
    extractor_name = FEATURE_EXTRACTORS[algo_name].__name__
    if args.use_curriculum:
        curriculum_desc = f"ON(start_level={args.curriculum_start_level})"
    else:
        curriculum_desc = f"OFF(fixed_level={args.non_curriculum_level})"
    print(
        "\n"
        f"[SB3] Starting run | algorithm={args.algorithm_name} | core={algo_name} | seed={args.seed} | "
        f"curriculum={curriculum_desc} | extractor={extractor_name} | params={params_path}"
    )
    if param_keys:
        print(f"[SB3] Loaded YAML params ({len(param_keys)}): {', '.join(param_keys)}")

    env = create_env_from_name(args, n_frames=int(args.n_frames))
    model = get_model(algo_name, env, args)

    checkpoint_dir = Path("models") / args.algorithm_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = _build_callbacks(args, run_name, checkpoint_dir)

    try:
        if str(getattr(args, "load_model", "")).strip():
            model = model.__class__.load(args.load_model, env=env, device=model.device)

        model.learn(
            total_timesteps=int(args.max_timesteps),
            callback=callbacks,
            log_interval=int(args.log_interval),
            tb_log_name=run_name,
        )
        model.save(str(checkpoint_dir / "final"))
    finally:
        env.close()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = get_config(argv)
    algorithm_names = _resolve_algorithms(args)
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for algorithm_name in algorithm_names:
        for seed in seeds:
            run_one(args, algorithm_name, _resolve_seed(seed))


if __name__ == "__main__":
    main()
