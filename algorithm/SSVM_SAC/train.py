#!/usr/bin/env python3
"""Independent staged trainer for SSVM-SAC.

Run this file directly to avoid importing the project's monolithic training
entry point::

    python algorithm/SSVM_SAC/train.py --help
"""

from __future__ import annotations

# Direct execution needs a package context for relative imports, but deliberately
# must not execute algorithm/__init__.py (which imports every existing agent).
if __package__ in (None, ""):
    import sys
    import types
    from pathlib import Path

    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parents[1]
    sys.path.insert(0, str(_ROOT))
    algorithm_package = types.ModuleType("algorithm")
    algorithm_package.__path__ = [str(_HERE.parent)]
    sys.modules.setdefault("algorithm", algorithm_package)
    ssvm_package = types.ModuleType("algorithm.SSVM_SAC")
    ssvm_package.__path__ = [str(_HERE)]
    sys.modules.setdefault("algorithm.SSVM_SAC", ssvm_package)
    __package__ = "algorithm.SSVM_SAC"

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .agent import SSVMSACAgent
from .checkpoints import (
    MODEL_VERSION,
    atomic_torch_save,
    build_perception,
    load_checkpoint,
    perception_from_checkpoint,
)
from .config import SSVMConfig
from .dataset import (
    DepthFrameDataset,
    EpisodeArchive,
    EpisodeSequenceDataset,
    build_reconstruction_targets,
    pad_episode_batch,
)
from .metrics import FinalStageCurveLogger
from .networks import (
    MultiFrameMambaReconstructor,
    TemporalMambaMemory,
    VisionMambaDecoder,
    VisionMambaEncoder,
)


ALGORITHM_NAME = "SSVM-SAC"
CURRICULUM_ALGORITHM_NAME = f"CL-{ALGORITHM_NAME}"


def _result_algorithm_name(curriculum: bool) -> str:
    """Return the canonical name used by logs, result paths, and plots."""
    return CURRICULUM_ALGORITHM_NAME if curriculum else ALGORITHM_NAME


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _writer(log_dir: Path):
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir)


def _event_log_dir(
    output: Path, curve_logger: FinalStageCurveLogger | None
) -> Path:
    if curve_logger is not None:
        return curve_logger.run_dir
    return output / "tensorboard"


def _offline_loader_options(workers: int, device: torch.device) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return options


def _depth_to_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    return values.to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    ).div_(255.0)


def _make_env(args: argparse.Namespace, unknown_args: list[str]):
    # Root config remains the authoritative environment configuration. Unknown
    # command-line options are forwarded to it, so all existing AirSim flags are
    # available without coupling this trainer to main_async.py.
    from config import get_config
    from gym_airsim.envs.AirGymSSVM import AirSimEnvSSVM

    environment_config = get_config(unknown_args)
    environment_config.seed = args.seed
    environment_config.n_frames = 1
    environment_config.max_timesteps = getattr(args, "max_steps", 0)
    environment_config.episode_length = getattr(
        args, "episode_length", environment_config.episode_length
    )
    environment_config.non_curriculum_level = getattr(args, "level", 3)
    environment_config.curriculum_final_level = getattr(args, "level", 3)
    if getattr(args, "curriculum", False) and environment_config.curriculum_mode != "progress":
        raise ValueError("SSVM curriculum requires --curriculum_mode progress")
    environment_config.include_clean_depth = bool(getattr(args, "clean_targets", False))
    environment_config.algorithm_name = _result_algorithm_name(
        getattr(args, "curriculum", False)
    )
    env = AirSimEnvSSVM(
        takeoff_height=environment_config.takeoff_height,
        config=environment_config,
        stack_frames=1,
    )
    env.action_space.seed(args.seed)
    return env


def _validate_observation(config: SSVMConfig, observation: dict[str, np.ndarray]) -> None:
    expected = (config.channels, config.image_height, config.image_width)
    if observation["depth"].shape != expected:
        raise ValueError(
            f"AirSim depth shape is {observation['depth'].shape}, expected {expected}; "
            "update the SSVM params.yaml image dimensions"
        )


def _close_env(env: Any) -> None:
    env.close()
    game_handler = getattr(env, "game_handler", None)
    if game_handler is not None:
        game_handler.kill_game_in_editor()


def _save_agent(agent: SSVMSACAgent, output: Path, stage: str, step: int) -> None:
    agent.save(output / "checkpoints" / f"{stage}_{step:09d}.pt", stage=stage, step=step)
    agent.save(output / f"{stage}_latest.pt", stage=stage, step=step)


def _format_episode_summary(
    *,
    display_name: str,
    episode: int,
    reward: float,
    episode_length: int,
    success_rate: float,
    level: int,
    total_timesteps: int,
    total_successes: int,
    difficulty: float | None = None,
    object_count_min: int | None = None,
    object_count_max: int | None = None,
) -> str:
    summary = (
        f"[{display_name}] Episode {episode}, Reward: {reward:.2f}, "
        f"Length: {episode_length}, Success Rate: {success_rate:.3f}, "
        f"Level: {level}"
    )
    if (
        difficulty is not None
        and object_count_min is not None
        and object_count_max is not None
    ):
        summary += (
            f", Difficulty: {difficulty:.3f}, "
            f"Objects: {object_count_min}-{object_count_max}"
        )
    return (
        f"{summary}, Total Timesteps: {total_timesteps}, "
        f"Total Successes: {total_successes}"
    )


def _environment_train(
    args: argparse.Namespace,
    unknown_args: list[str],
    *,
    stage: str,
    checkpoint: dict[str, Any] | None,
) -> None:
    device = _device(args.device)
    env = _make_env(args, unknown_args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    curve_logger = None
    display_name = "bootstrap"
    if stage == "sac":
        algorithm_name = _result_algorithm_name(args.curriculum)
        display_name = algorithm_name
        curve_logger = FinalStageCurveLogger(
            algorithm_name=algorithm_name,
            seed=args.seed,
            results_root=args.results_root,
            overwrite=args.overwrite_results,
        )
        print(f"Final-stage curve log: {curve_logger.csv_path}", flush=True)
    writer = _writer(_event_log_dir(output, curve_logger))
    try:
        observation, _ = env.reset(seed=args.seed)
        if checkpoint is None:
            config = SSVMConfig.load(args.config)
            config.image_height, config.image_width = observation["depth"].shape[-2:]
            perception = build_perception(config).to(device).freeze()
        else:
            config, perception = perception_from_checkpoint(checkpoint, device, freeze=True)
        _validate_observation(config, observation)
        agent = SSVMSACAgent(
            perception,
            observation["base"].size,
            env.action_space.low,
            env.action_space.high,
            config,
            device=device,
            seed=args.seed,
        )
        if checkpoint is not None and "actor" in checkpoint and args.warm_start_sac:
            agent.load_sac_state(checkpoint)

        total_steps = 0
        episode = 0
        recent_successes: collections.deque[float] = collections.deque(maxlen=100)
        if args.steps_per_update <= 0:
            raise ValueError("steps_per_update must be greater than zero")
        if args.gradient_steps <= 0:
            raise ValueError("gradient_steps must be greater than zero")
        while total_steps < args.max_steps:
            if hasattr(env, "set_curriculum_progress") and args.curriculum:
                env.set_curriculum_progress(total_steps / max(args.max_steps, 1))
            if episode:
                observation, _ = env.reset(seed=args.seed + episode)
            agent.reset()
            memory = agent.encode_frame(observation["depth"])
            episode_reward = 0.0
            success = False
            for episode_step in range(args.episode_length):
                if total_steps < args.learning_starts:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(observation["base"], memory)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_memory = agent.encode_frame(next_observation["depth"])
                trainer_truncated = episode_step + 1 >= args.episode_length
                budget_truncated = total_steps + 1 >= args.max_steps
                done = bool(
                    terminated or truncated or trainer_truncated or budget_truncated
                )
                step_success = bool(info.get("is_success", False))
                agent.add_transition(
                    observation["base"], memory, action, reward,
                    next_observation["base"], next_memory, done,
                    is_success=step_success,
                )
                observation, memory = next_observation, next_memory
                episode_reward += float(reward)
                total_steps += 1
                success = success or step_success
                if (
                    total_steps >= args.learning_starts
                    and total_steps % args.steps_per_update == 0
                ):
                    n_updates = max(
                        1, int(args.steps_per_update * args.gradient_steps)
                    )
                    for _ in tqdm(
                        range(n_updates),
                        desc=f"Training ({total_steps})",
                        leave=False,
                    ):
                        metrics = agent.update(
                            progress_ratio=total_steps / max(args.max_steps, 1)
                        )
                        for name, value in metrics.items():
                            writer.add_scalar(f"train/{name}", value, total_steps)
                if total_steps % args.checkpoint_interval == 0:
                    _save_agent(agent, output, stage, total_steps)
                if done or total_steps >= args.max_steps:
                    break
            writer.add_scalar("episode/reward", episode_reward, total_steps)
            writer.add_scalar("episode/success", float(success), total_steps)
            recent_successes.append(float(success))
            writer.add_scalar(
                "episode/success_rate_100",
                float(np.mean(recent_successes)),
                total_steps,
            )
            env_core = env.unwrapped if hasattr(env, "unwrapped") else env
            success_history = getattr(env_core, "success_deque", ())
            success_rate = (
                float(sum(success_history) / len(success_history))
                if success_history
                else float(success)
            )
            completed_episode_length = episode_step + 1
            total_successes = int(
                getattr(env_core, "success_count", sum(success_history))
            )
            writer.add_scalar("train/episode_reward", episode_reward, total_steps)
            writer.add_scalar(
                "train/episode_length", completed_episode_length, total_steps
            )
            writer.add_scalar("train/success_rate", success_rate, total_steps)
            writer.add_scalar("train/success_count", total_successes, total_steps)
            if curve_logger is not None:
                curve_logger.record(
                    episode=episode,
                    total_timesteps=total_steps,
                    reward=episode_reward,
                    episode_length=completed_episode_length,
                    success_rate=success_rate,
                )
            curriculum_info = None
            get_curriculum_info = getattr(env_core, "get_curriculum_info", None)
            if args.curriculum and callable(get_curriculum_info):
                curriculum_info = get_curriculum_info()
            print(
                _format_episode_summary(
                    display_name=display_name,
                    episode=episode,
                    reward=episode_reward,
                    episode_length=completed_episode_length,
                    success_rate=success_rate,
                    level=int(getattr(env_core, "level", args.level)),
                    total_timesteps=total_steps,
                    total_successes=total_successes,
                    difficulty=(
                        curriculum_info.get("difficulty")
                        if curriculum_info is not None
                        else None
                    ),
                    object_count_min=(
                        curriculum_info.get("number_of_objects_min")
                        if curriculum_info is not None
                        else None
                    ),
                    object_count_max=(
                        curriculum_info.get("number_of_objects_max")
                        if curriculum_info is not None
                        else None
                    ),
                ),
                flush=True,
            )
            episode += 1
        _save_agent(agent, output, stage, total_steps)
    finally:
        writer.close()
        _close_env(env)


def train_bootstrap(args: argparse.Namespace, unknown_args: list[str]) -> None:
    _environment_train(args, unknown_args, stage="bootstrap", checkpoint=None)


def train_final_sac(args: argparse.Namespace, unknown_args: list[str]) -> None:
    checkpoint = load_checkpoint(args.perception_checkpoint, _device(args.device))
    if checkpoint["stage"] not in {"memory", "sac"}:
        raise ValueError("final SAC requires a memory-stage checkpoint")
    _environment_train(args, unknown_args, stage="sac", checkpoint=checkpoint)


@torch.no_grad()
def collect_sequences(args: argparse.Namespace, unknown_args: list[str]) -> None:
    device = _device(args.device)
    checkpoint = load_checkpoint(args.policy_checkpoint, device)
    if checkpoint["stage"] not in {"bootstrap", "sac"} or "actor" not in checkpoint:
        raise ValueError("collection requires a bootstrap or final SAC policy checkpoint")
    env = _make_env(args, unknown_args)
    dataset_root = Path(args.dataset)
    collect_config = SSVMConfig.load(args.config)
    target_frames = (
        args.target_frames if args.target_frames is not None else collect_config.collect_frames
    )
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    if args.overwrite:
        dataset_root.mkdir(parents=True, exist_ok=True)
        old_files = list(dataset_root.glob("episode_*.npz"))
        for old_file in old_files:
            old_file.unlink()
        print(f"[collect] overwrite=True removed={len(old_files)} old episodes")
    try:
        observation, _ = env.reset(seed=args.seed)
        config, perception = perception_from_checkpoint(checkpoint, device, freeze=True)
        _validate_observation(config, observation)
        agent = SSVMSACAgent(
            perception, observation["base"].size,
            env.action_space.low, env.action_space.high, config,
            device=device, seed=args.seed,
        )
        agent.load_sac_state(checkpoint)
        first_episode_id = 0 if args.overwrite else EpisodeArchive.next_episode_id(dataset_root)
        collected_frames = 0
        episode = 0
        while collected_frames < target_frames or episode == 0:
            if episode:
                observation, _ = env.reset(seed=args.seed + episode)
            agent.reset()
            memory = agent.encode_frame(observation["depth"])
            fields: dict[str, list[Any]] = {
                name: [] for name in (
                    "depth", "base", "action", "reward", "terminated", "truncated"
                )
            }
            clean_depth: list[np.ndarray] = []
            success = False
            for episode_step in range(args.episode_length):
                action = agent.select_action(
                    observation["base"], memory, deterministic=args.deterministic
                )
                next_observation, reward, terminated, truncated, info = env.step(action)
                fields["depth"].append(
                    np.clip(observation["depth"], 0, 255).astype(np.uint8)
                )
                fields["base"].append(np.asarray(observation["base"], dtype=np.float32))
                fields["action"].append(np.asarray(action, dtype=np.float32))
                fields["reward"].append(float(reward))
                fields["terminated"].append(bool(terminated))
                trainer_truncated = episode_step + 1 >= args.episode_length
                fields["truncated"].append(bool(truncated or trainer_truncated))
                if "clean_depth" in observation:
                    clean_depth.append(
                        np.clip(observation["clean_depth"], 0, 255).astype(np.uint8)
                    )
                success = success or bool(info.get("is_success", False))
                observation = next_observation
                memory = agent.encode_frame(observation["depth"])
                if terminated or truncated or trainer_truncated:
                    break
            EpisodeArchive.save(
                dataset_root,
                episode_id=first_episode_id + episode,
                **{name: np.asarray(values) for name, values in fields.items()},
                clean_depth=np.asarray(clean_depth) if clean_depth else None,
                metadata={"success": success, "seed": args.seed + episode},
            )
            collected_frames += len(fields["depth"])
            print(
                f"[collect] episode={episode} frames={len(fields['depth'])} "
                f"collected={collected_frames}/{target_frames} success={int(success)}"
            )
            episode += 1
    finally:
        _close_env(env)


def _reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    *,
    charbonnier_epsilon: float = 1e-3,
) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(prediction, target)
    if kind == "mse":
        return F.mse_loss(prediction, target, reduction="sum")
    if kind == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    if kind == "charbonnier":
        epsilon = float(charbonnier_epsilon)
        if epsilon <= 0.0:
            raise ValueError("charbonnier_epsilon must be positive")
        error = prediction - target
        return (torch.sqrt(error.square() + epsilon**2) - epsilon).mean()
    raise ValueError(f"unsupported reconstruction loss: {kind}")


def _sum_memory_offset_losses(losses: list[torch.Tensor]) -> torch.Tensor:
    """Sum the enabled temporal reconstruction objectives."""
    if not losses:
        raise ValueError("at least one temporal reconstruction loss is required")
    return torch.stack(losses).sum()


def _memory_stage_config(
    vision_checkpoint_config: dict[str, Any], requested_config: SSVMConfig
) -> SSVMConfig:
    config = SSVMConfig.from_mapping(vision_checkpoint_config)
    # Reconstruction offsets and objective belong to temporal training, not
    # the spatial encoder architecture. This lets an existing vision
    # checkpoint be reused when either setting changes.
    config.reconstruction_offsets = requested_config.reconstruction_offsets
    config.memory_dim = requested_config.memory_dim
    config.memory_depth = requested_config.memory_depth
    config.memory_d_state = requested_config.memory_d_state
    config.reconstruction_latent_dim = requested_config.reconstruction_latent_dim
    config.reconstruction_loss = requested_config.reconstruction_loss
    config.memory_latent_loss_weight = requested_config.memory_latent_loss_weight
    config.charbonnier_epsilon = requested_config.charbonnier_epsilon
    # Stage 4 optimization settings come from the current YAML, not the
    # configuration snapshot stored during stage 3.
    config.memory_lr = requested_config.memory_lr
    config.memory_aux_lr = requested_config.memory_aux_lr
    config.memory_batch_size = requested_config.memory_batch_size
    config.memory_validation_interval = requested_config.memory_validation_interval
    config.memory_lrf = requested_config.memory_lrf
    config.offline_lr_decay = requested_config.offline_lr_decay
    return config


def _cosine_lr_scheduler(optimizer, final_ratio: float, epochs: int):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.0 <= final_ratio <= 1.0:
        raise ValueError("final_ratio must be in [0, 1]")
    def cosine_ratio(epoch: int) -> float:
        return final_ratio + (1.0 - final_ratio) * (
            1.0 + math.cos(math.pi * epoch / epochs)
        ) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[cosine_ratio] * len(optimizer.param_groups)
    )


class ReconstructionStats:
    """Pixel-weighted epoch metrics, independent of optimization reduction."""

    def __init__(self):
        self.sse = 0.0
        self.sae = 0.0
        self.pixels = 0

    def add(self, prediction, target):
        error = prediction.detach() - target.detach()
        self.sse += float(error.square().sum())
        self.sae += float(error.abs().sum())
        self.pixels += target.numel()

    @property
    def mse(self):
        return self.sse / self.pixels if self.pixels else float("nan")

    @property
    def mae(self):
        return self.sae / self.pixels if self.pixels else float("nan")


@torch.no_grad()
def _validate_reconstruction(loader, encoder, decoder, device, config, memory=None):
    modules = [encoder, decoder] + ([memory] if memory is not None else [])
    modes = [module.training for module in modules]
    stats = {offset: ReconstructionStats() for offset in (
        config.reconstruction_offsets if memory is not None else (0,)
    )}
    try:
        for module in modules:
            module.eval()
        for batch in tqdm(loader, desc="Validation", unit="batch", leave=False):
            frames = _depth_to_device(batch["depth"], device)
            targets = _depth_to_device(batch["target_depth"], device)
            if memory is None:
                stats[0].add(decoder(encoder(frames)), targets)
            else:
                b, t, c, h, w = frames.shape
                latent = encoder(frames.reshape(-1, c, h, w)).reshape(b, t, -1)
                predictions = decoder(memory(latent))
                targets, masks = build_reconstruction_targets(
                    targets, batch["valid"].to(device), config.reconstruction_offsets
                )
                for offset in stats:
                    stats[offset].add(predictions[offset][masks[offset]], targets[offset][masks[offset]])
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)
    return stats


def _log_reconstruction_epoch(writer, stage, epoch, train_stats, val_stats):
    values = {}
    for split, stats in (("train", train_stats), ("val", val_stats)):
        if stats is None:
            continue
        sse = sum(item.sse for item in stats.values())
        pixels = sum(item.pixels for item in stats.values())
        values[f"{split}_sse"] = sse
        values[f"{split}_mse"] = sse / pixels if pixels else float("nan")
        sae = sum(item.sae for item in stats.values())
        values[f"{split}_mae"] = sae / pixels if pixels else float("nan")
        if stage in {"memory", "memory_latent"}:
            for offset, item in stats.items():
                values[f"{split}_mse_t{offset:+d}"] = item.mse
                values[f"{split}_mae_t{offset:+d}"] = item.mae
    for name, value in values.items():
        writer.add_scalar(f"{stage}/epoch/{name}", value, epoch)
    summaries = []
    for split, stats in (("train", train_stats), ("val", val_stats)):
        if stats is None:
            continue
        lines = [
            f"{split}: MSE={values[f'{split}_mse']:.6f} "
            f"MAE={values[f'{split}_mae']:.6f}"
        ]
        if stage in {"memory", "memory_latent"}:
            lines.extend(
                f"  t{offset:+d}: MSE={values[f'{split}_mse_t{offset:+d}']:.6f} "
                f"MAE={values[f'{split}_mae_t{offset:+d}']:.6f}"
                for offset in stats
            )
        summaries.extend(lines)
    return "\n".join(summaries)


def train_vision(args: argparse.Namespace, _unknown_args: list[str]) -> None:
    device = _device(args.device)
    config = SSVMConfig.load(args.config)
    dataset = DepthFrameDataset(args.dataset, split="train", seed=args.seed)
    if not dataset:
        raise ValueError(f"no training frames found in {args.dataset}")
    if tuple(dataset[0]["depth"].shape) != (
        config.channels, config.image_height, config.image_width
    ):
        raise ValueError("dataset frame shape does not match the SSVM configuration")
    batch_size = args.batch_size or config.vision_batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **_offline_loader_options(args.workers, device),
    )
    print(
        f"[vision] device={device} frames={len(dataset)} "
        f"batch_size={batch_size} workers={args.workers} dataset=preloaded"
    )
    encoder = VisionMambaEncoder(
        config.image_size, channels=config.channels, patch_size=config.patch_size,
        latent_dim=config.latent_dim, depth=config.encoder_depth, d_state=config.d_state,
        drop_rate=config.drop_rate, drop_path_rate=config.drop_path_rate,
    ).to(device)
    decoder = VisionMambaDecoder(
        config.latent_dim,
        config.image_size,
        patch_size=config.patch_size,
        channels=config.channels,
        embed_dim=config.decoder_embed_dim,
        depth=config.decoder_depth,
        d_state=config.d_state,
        d_conv=config.d_conv, expand=config.expand,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()], lr=config.vision_lr
    )
    scheduler = _cosine_lr_scheduler(optimizer, config.vision_lrf, args.epochs)
    output = Path(args.output)
    writer = _writer(output / "tensorboard")
    step = 0
    validation = DataLoader(
        DepthFrameDataset(args.dataset, split="validation", seed=args.seed),
        batch_size=batch_size, shuffle=False,
        **_offline_loader_options(args.workers, device),
    )
    try:
        for epoch in range(args.epochs):
            stats = ReconstructionStats()
            epoch_loss_sum = 0.0
            epoch_updates = 0
            progress = tqdm(
                loader,
                desc=f"Vision Epoch {epoch + 1}/{args.epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in progress:
                frames = _depth_to_device(batch["depth"], device)
                targets = _depth_to_device(batch["target_depth"], device)
                prediction = decoder(encoder(frames))
                loss = _reconstruction_loss(
                    prediction,
                    targets,
                    config.vision_reconstruction_loss,
                    charbonnier_epsilon=config.charbonnier_epsilon,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                writer.add_scalar("vision/loss", float(loss.detach()), step)
                stats.add(prediction, targets)
                progress.set_postfix(batch_loss=f"{float(loss.detach()):.6f}", train_mse=f"{stats.mse:.6f}")
                epoch_loss_sum += float(loss.detach())
                epoch_updates += 1
                step += 1
            val_stats = _validate_reconstruction(validation, encoder, decoder, device, config)
            summary = _log_reconstruction_epoch(writer, "vision", epoch + 1, {0: stats}, val_stats)
            scheduler.step()
            mean_loss = epoch_loss_sum / epoch_updates if epoch_updates else float("nan")
            print(f"[vision] epoch={epoch + 1}/{args.epochs} loss={mean_loss:.6f} lr={optimizer.param_groups[0]['lr']:.3g}\n{summary}")
            atomic_torch_save({
                "stage": "vision", "model_version": MODEL_VERSION,
                "dataset_version": 1,
                "normalization": {"depth_divisor": 255.0},
                "epoch": epoch + 1, "config": config.to_dict(),
                "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, output / "vision_latest.pt")
    finally:
        writer.close()


@torch.no_grad()
def _cache_memory_latents(dataset, encoder, device, chunk_size=128):
    """Encode observed frames once; the same frozen representation is the target."""
    cached = []
    for sample in tqdm(dataset, desc="Caching episode latents", unit="episode"):
        frames = sample["depth"][sample["valid"]]
        z = torch.cat([encoder(_depth_to_device(chunk, device)).cpu()
                       for chunk in frames.split(chunk_size)])
        if not torch.isfinite(z).all():
            raise FloatingPointError("Non-finite cached Encoder features")
        cached.append({"latent": z, "valid": torch.ones(len(z), dtype=torch.bool)})
    return cached


def _valid_memory_anchors(cached, offsets):
    """Uniform sampling population: every (episode, t) with ALL targets present."""
    begin = max(0, -min(offsets))
    future = max(0, max(offsets))
    return [(episode, t) for episode, sample in enumerate(cached)
            for t in range(begin, len(sample["depth"] if "depth" in sample else sample["latent"]) - future)]


def _sample_memory_batch(cached, anchors, batch_size, offsets, rng):
    if not anchors or batch_size < 1:
        raise ValueError("Need valid memory anchors and a positive batch size")
    if batch_size > len(anchors):
        raise ValueError("batch_size cannot exceed the number of valid memory anchors")
    # Each batch contains distinct anchors; replacement happens only across batches.
    selected = [anchors[int(i)] for i in rng.choice(len(anchors), size=batch_size, replace=False)]
    return _gather_memory_batch(cached, selected, offsets)


def _gather_memory_batch(cached, selected, offsets):
    prefixes, targets, lengths = [], [], []
    for episode, t in selected:
        sample = cached[episode]
        z = sample["depth"] if "depth" in sample else sample["latent"]
        prefixes.append(z[:t+1])
        target_frames = sample.get("target_depth", z)
        targets.append(torch.stack([target_frames[t+k] for k in offsets]))
        lengths.append(t+1)
    return {"latent": torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True),
            "lengths": torch.tensor(lengths), "targets": torch.stack(targets)}


def _sampled_image_objective(encoder, memory, reconstructor, batch, device, config, stats):
    frames = batch["latent"]  # Causal image prefixes; shared sampler's payload key.
    b, t, c, h, w = frames.shape
    with torch.no_grad():
        latent = torch.cat([encoder(_depth_to_device(chunk, device))
                            for chunk in frames.reshape(-1, c, h, w).split(128)])
    # Outputs y_t already incorporate causal history. Internal recurrent states
    # are maintained by the scan, not passed to the reconstruction projection.
    output_features = memory(latent.reshape(b, t, -1))
    anchor_features = output_features[
        torch.arange(b, device=device), batch["lengths"].to(device)-1
    ]
    predictions = reconstructor(anchor_features)
    targets = _depth_to_device(batch["targets"], device)
    losses = []
    for i, offset in enumerate(config.reconstruction_offsets):
        stats[offset].add(predictions[offset], targets[:, i])
        losses.append(_reconstruction_loss(predictions[offset], targets[:, i],
                      config.reconstruction_loss, charbonnier_epsilon=config.charbonnier_epsilon))
    return torch.stack(losses).sum()


def _episode_image_objective(encoder, memory, reconstructor, sample, device, config, stats):
    """Compute one loss over every fully valid time in a complete episode."""
    frames = _depth_to_device(sample["depth"].unsqueeze(0), device)
    targets = _depth_to_device(sample["target_depth"].unsqueeze(0), device)
    valid = sample["valid"].unsqueeze(0).to(device)
    b, t, c, h, w = frames.shape
    with torch.no_grad():
        latent = torch.cat([
            encoder(chunk) for chunk in frames.reshape(-1, c, h, w).split(128)
        ]).reshape(b, t, -1)
    outputs = memory(latent)
    target_map, masks = build_reconstruction_targets(
        targets, valid, config.reconstruction_offsets
    )
    complete = torch.stack(list(masks.values())).all(dim=0)
    if not complete.any():
        return None
    predictions = reconstructor(outputs)
    losses = []
    for index, offset in enumerate(config.reconstruction_offsets):
        mask = complete
        prediction = predictions[offset][mask]
        target = target_map[offset][mask]
        stats[offset].add(prediction, target)
        losses.append(_reconstruction_loss(
            prediction, target, config.reconstruction_loss,
            charbonnier_epsilon=config.charbonnier_epsilon,
        ))
    return torch.stack(losses).sum()


def _sampled_memory_objective(memory, reconstructor, batch, device, offsets, stats):
    output_features = memory(batch["latent"].to(device))
    last = output_features[torch.arange(len(output_features), device=device), batch["lengths"].to(device)-1]
    codes = reconstructor.projection(last).reshape(len(output_features), len(offsets), -1)
    targets = batch["targets"].to(device)
    for i, offset in enumerate(offsets):
        stats[offset].add(codes[:, i], targets[:, i])
    return (codes-targets).square().mean(dim=(0, 2)).sum()


def _memory_latent_objective(memory, reconstructor, batch, device, offsets, stats, complete_only=False):
    latent = batch["latent"].to(device)
    valid = batch["valid"].to(device)
    output_features = memory(latent)
    codes = reconstructor.projection(output_features).reshape(*output_features.shape[:-1], len(offsets), -1)
    targets, masks = build_reconstruction_targets(latent, valid, offsets)
    if complete_only:
        common = torch.stack(list(masks.values())).all(dim=0)
        masks = {offset: common for offset in offsets}
    losses = []
    for index, offset in enumerate(offsets):
        if masks[offset].any():
            prediction = codes[:, :, index][masks[offset]]
            target = targets[offset][masks[offset]]
            stats[offset].add(prediction, target)
            losses.append(F.mse_loss(prediction, target))
    return torch.stack(losses).sum() if losses else None


def train_memory(args: argparse.Namespace, _unknown_args: list[str]) -> None:
    device = _device(args.device)
    source = load_checkpoint(args.vision_checkpoint, device)
    if source["stage"] != "vision":
        raise ValueError("memory training requires a vision-stage checkpoint")
    requested_config = SSVMConfig.load(args.config)
    config = _memory_stage_config(source["config"], requested_config)
    dataset = EpisodeSequenceDataset(
        args.dataset, sequence_length=None, split="train", seed=args.seed,
    )
    if not dataset:
        raise ValueError(f"no sequence windows found in {args.dataset}")
    if tuple(dataset[0]["depth"].shape[1:]) != (
        config.channels, config.image_height, config.image_width
    ):
        raise ValueError("dataset frame shape does not match the vision checkpoint")
    batch_size = args.batch_size or config.memory_batch_size
    if batch_size < 1:
        raise ValueError("Memory batch size must be positive")
    config.memory_batch_size = batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=pad_episode_batch,
                        **_offline_loader_options(args.workers, device))
    print(f"[memory] sampling=complete_episodes train_episodes={len(dataset)} "
          f"batch_size={batch_size} workers={args.workers} "
          f"offsets={config.reconstruction_offsets}")
    encoder = VisionMambaEncoder(
        config.image_size, channels=config.channels, patch_size=config.patch_size,
        latent_dim=config.latent_dim, depth=config.encoder_depth, d_state=config.d_state,
        drop_rate=config.drop_rate, drop_path_rate=config.drop_path_rate,
    ).to(device)
    encoder.load_state_dict(source["encoder"])
    encoder.eval().requires_grad_(False)
    validation_dataset = EpisodeSequenceDataset(
        args.dataset, sequence_length=None, split="validation", seed=args.seed,
    )
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=pad_episode_batch,
                            **_offline_loader_options(args.workers, device))
    memory = TemporalMambaMemory(
        config.latent_dim, config.memory_dim, depth=config.memory_depth,
        d_state=config.memory_d_state, d_conv=config.d_conv, expand=config.expand,
    ).to(device)
    decoder = VisionMambaDecoder(
        config.reconstruction_latent_dim,
        config.image_size,
        patch_size=config.patch_size,
        channels=config.channels,
        embed_dim=config.decoder_embed_dim,
        depth=config.decoder_depth,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
    ).to(device)
    if "decoder" not in source:
        raise ValueError("vision checkpoint does not contain its decoder")
    decoder.load_state_dict(source["decoder"])
    decoder.eval().requires_grad_(False)
    reconstructor = MultiFrameMambaReconstructor(
        config.memory_dim,
        config.reconstruction_latent_dim,
        config.reconstruction_offsets,
        decoder,
    ).to(device)
    optimizer = torch.optim.RAdam([
        {"params": memory.parameters(), "lr": config.memory_lr},
        {"params": [*reconstructor.projection.parameters()],
         "lr": config.memory_aux_lr},
    ])
    print(f"[memory] optimizer=RAdam objective=image_{config.reconstruction_loss}+latent; Encoder/Decoder frozen; training Memory/projection")
    scheduler = _cosine_lr_scheduler(optimizer, config.memory_lrf, args.epochs)
    print(f"[memory] scheduler=cosine temporal_lr={config.memory_lr:g} "
          f"aux_lr={config.memory_aux_lr:g} "
          f"temporal_final_lr={config.memory_lr * config.memory_lrf:g} "
          f"aux_final_lr={config.memory_aux_lr * config.memory_lrf:g} epochs={args.epochs}")
    output = Path(args.output)
    writer = _writer(output / "tensorboard")
    step = 0
    try:
        for epoch in range(args.epochs):
            stats = {offset: ReconstructionStats() for offset in config.reconstruction_offsets}
            epoch_loss_sum = 0.0
            epoch_updates = 0
            progress = tqdm(
                loader,
                desc=f"Memory Epoch {epoch + 1}/{args.epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in progress:
                frames = _depth_to_device(batch["depth"], device)
                target_frames = _depth_to_device(batch["target_depth"], device)
                valid = batch["valid"].to(device)
                b, length, channels, height, width = frames.shape
                with torch.no_grad():
                    latent = encoder(
                        frames.reshape(-1, channels, height, width)
                    ).reshape(b, length, -1)
                    target_latent = encoder(
                        target_frames.reshape(-1, channels, height, width)
                    ).reshape(b, length, -1)
                memories = memory(latent)
                targets, masks = build_reconstruction_targets(
                    target_frames, valid, config.reconstruction_offsets
                )
                predictions = reconstructor(memories)
                codes = reconstructor.project_codes(memories)
                losses = []
                for offset in config.reconstruction_offsets:
                    if masks[offset].any():
                        prediction = predictions[offset][masks[offset]]
                        target = targets[offset][masks[offset]]
                        stats[offset].add(prediction, target)
                        # Normalize image loss to the same scale as latent loss;
                        # otherwise pixel-count summation overwhelms the direct
                        # temporal recall signal.
                        image_loss = _reconstruction_loss(
                            prediction, target, config.reconstruction_loss,
                            charbonnier_epsilon=config.charbonnier_epsilon,
                        ) / prediction.numel()
                        latent_target = target_latent
                        if offset < 0:
                            shift = -offset
                            latent_prediction = codes[:, :, config.reconstruction_offsets.index(offset)][masks[offset]]
                            latent_target = target_latent[:, :-shift][valid[:, shift:] & valid[:, :-shift]]
                        elif offset > 0:
                            latent_prediction = codes[:, :, config.reconstruction_offsets.index(offset)][masks[offset]]
                            latent_target = target_latent[:, offset:][valid[:, :-offset] & valid[:, offset:]]
                        else:
                            latent_prediction = codes[:, :, config.reconstruction_offsets.index(offset)][masks[offset]]
                            latent_target = target_latent[masks[offset]]
                        latent_loss = F.mse_loss(latent_prediction, latent_target)
                        losses.append(image_loss + config.memory_latent_loss_weight * latent_loss)
                if not losses:
                    continue
                loss = _sum_memory_offset_losses(losses)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Memory image loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += float(loss.detach())
                epoch_updates += 1
                writer.add_scalar("memory/loss", float(loss.detach()), step)
                pixels = sum(item.pixels for item in stats.values())
                mse = sum(item.sse for item in stats.values()) / pixels
                mae = sum(item.sae for item in stats.values()) / pixels
                progress.set_postfix(batch_loss=f"{float(loss.detach()):.6f}", train_mse=f"{mse:.6f}", train_mae=f"{mae:.6f}")
                step += 1
            val_stats = None
            if (epoch + 1) % config.memory_validation_interval == 0 or epoch + 1 == args.epochs:
                val_stats = {offset: ReconstructionStats() for offset in config.reconstruction_offsets}
                memory.eval()
                reconstructor.eval()
                val_stats = _validate_reconstruction(
                    validation, encoder, reconstructor, device, config, memory
                )
                memory.train()
                reconstructor.train()
            summary = _log_reconstruction_epoch(writer, "memory", epoch + 1, stats, val_stats)
            scheduler.step()
            mean_loss = epoch_loss_sum / epoch_updates if epoch_updates else float("nan")
            print(f"[memory] epoch={epoch + 1}/{args.epochs} loss={mean_loss:.6f}\n"
                  f"{summary}\nlr_temporal={optimizer.param_groups[0]['lr']:.3g} "
                  f"lr_aux={optimizer.param_groups[1]['lr']:.3g}")
            atomic_torch_save({
                "stage": "memory", "model_version": MODEL_VERSION,
                "memory_objective": "image_and_latent_reconstruction", "decoder_frozen": True,
                "memory_sampling": "complete_episodes_all_valid_times",
                "optimizer_steps": step,
                "dataset_version": 1,
                "normalization": {"depth_divisor": 255.0},
                "epoch": epoch + 1, "config": config.to_dict(),
                "encoder": encoder.state_dict(), "memory": memory.state_dict(),
                "reconstructor": reconstructor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, output / "memory_latest.pt")
    finally:
        writer.close()


@torch.no_grad()
def evaluate(args: argparse.Namespace, unknown_args: list[str]) -> None:
    device = _device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config, perception = perception_from_checkpoint(checkpoint, device, freeze=True)
    env = _make_env(args, unknown_args)
    successes: list[float] = []
    returns: list[float] = []
    episode_records: list[dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=args.seed)
        _validate_observation(config, observation)
        agent = SSVMSACAgent(
            perception, observation["base"].size,
            env.action_space.low, env.action_space.high, config,
            device=device, seed=args.seed,
        )
        agent.load_sac_state(checkpoint)
        for episode in range(args.episodes):
            if episode:
                observation, _ = env.reset(seed=args.seed + episode)
            agent.reset()
            memory = agent.encode_frame(observation["depth"])
            episode_return = 0.0
            success = False
            for _ in range(args.episode_length):
                action = agent.select_action(observation["base"], memory, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                memory = agent.encode_frame(observation["depth"])
                episode_return += float(reward)
                success = success or bool(info.get("is_success", False))
                if terminated or truncated:
                    break
            returns.append(episode_return)
            successes.append(float(success))
            episode_records.append(
                {"episode": episode, "return": episode_return, "success": success}
            )
            print(f"[eval] episode={episode} return={episode_return:.3f} success={int(success)}")
        result = {
            "episodes": args.episodes,
            "mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
        }
        print(json.dumps(result, indent=2))
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation.json").write_text(
            json.dumps({**result, "episode_results": episode_records}, indent=2),
            encoding="utf-8",
        )
    finally:
        _close_env(env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent staged SSVM-SAC trainer",
        epilog="Unknown options are forwarded to the project's AirSim config parser.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default=str(Path(__file__).with_name("params.yaml")))
        subparser.add_argument("--seed", type=int, default=25)
        subparser.add_argument("--device", default="auto")
        subparser.add_argument("--output", default="runs/SSVM_SAC")

    def environment(subparser: argparse.ArgumentParser) -> None:
        common(subparser)
        subparser.add_argument("--max-steps", type=int, default=150_000)
        subparser.add_argument("--episode-length", type=int, default=300)
        subparser.add_argument(
            "--level", type=int, choices=range(4), default=3,
            help="Curriculum final level, or fixed level without curriculum (0-3; default: 3)",
        )
        subparser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="stage 1: SAC with random frozen perception"
    )
    environment(bootstrap)
    bootstrap.set_defaults(level=3, max_steps=20_000)
    bootstrap.add_argument("--learning-starts", type=int, default=1_000)
    bootstrap.add_argument(
        "--steps-per-update", "--steps_per_update", type=int, default=100
    )
    bootstrap.add_argument(
        "--gradient-steps",
        "--gradient_steps",
        "--updates-per-step",
        dest="gradient_steps",
        type=float,
        default=0.5,
    )
    bootstrap.add_argument("--checkpoint-interval", type=int, default=10_000)
    bootstrap.set_defaults(warm_start_sac=False)
    bootstrap.set_defaults(function=train_bootstrap)

    collect = subparsers.add_parser("collect", help="stage 2: collect complete episode files")
    common(collect)
    collect.add_argument("--policy-checkpoint", required=True)
    collect.add_argument("--dataset", required=True)
    collect.add_argument(
        "--target-frames",
        type=int,
        help="minimum number of collected frames before finishing the current episode; defaults to params.yaml collect_frames",
    )
    collect.add_argument("--episode-length", type=int, default=300)
    collect.add_argument("--max-steps", type=int, default=0)
    collect.add_argument("--level", type=int, choices=range(4), default=3)
    collect.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    collect.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=True,
        help="clear existing episode_*.npz files before collection (default: true)",
    )
    collect.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    collect.add_argument(
        "--clean-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="store noisy observations with AirSim clean depth as reconstruction targets",
    )
    collect.set_defaults(function=collect_sequences)

    vision = subparsers.add_parser("vision", help="stage 3: single-frame Vision Mamba autoencoder")
    common(vision)
    vision.add_argument("--dataset", required=True)
    vision.add_argument("--epochs", type=int, default=50)
    vision.add_argument("--batch-size", type=int)
    vision.add_argument("--workers", type=int, default=4)
    vision.set_defaults(function=train_vision)

    memory = subparsers.add_parser("memory", help="stage 4: self-supervised causal memory with configured temporal targets")
    common(memory)
    memory.add_argument("--dataset", required=True)
    memory.add_argument("--vision-checkpoint", required=True)
    memory.add_argument("--epochs", type=int, default=100)
    memory.add_argument("--batch-size", type=int)
    memory.add_argument("--workers", type=int, default=8)
    memory.set_defaults(function=train_memory)

    sac = subparsers.add_parser("sac", help="stage 5: reinitialized SAC on frozen memory")
    environment(sac)
    sac.set_defaults(curriculum=True)
    sac.add_argument("--perception-checkpoint", required=True)
    sac.add_argument("--learning-starts", type=int, default=1_000)
    sac.add_argument(
        "--steps-per-update", "--steps_per_update", type=int, default=100
    )
    sac.add_argument(
        "--gradient-steps",
        "--gradient_steps",
        "--updates-per-step",
        dest="gradient_steps",
        type=float,
        default=0.5,
    )
    sac.add_argument("--checkpoint-interval", type=int, default=10_000)
    sac.add_argument(
        "--results-root",
        default="results",
        help="root directory for SSVM-SAC final-stage curve data",
    )
    sac.add_argument(
        "--overwrite-results",
        action="store_true",
        help="replace an existing final-stage result directory for this seed",
    )
    sac.add_argument(
        "--warm-start-sac",
        action="store_true",
        help="load SAC heads when the perception checkpoint is a prior SAC checkpoint",
    )
    sac.set_defaults(function=train_final_sac)

    evaluation = subparsers.add_parser("eval", help="evaluate only a final SAC checkpoint")
    common(evaluation)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--episodes", type=int, default=100)
    evaluation.add_argument("--episode-length", type=int, default=300)
    evaluation.add_argument("--max-steps", type=int, default=0)
    evaluation.add_argument("--level", type=int, choices=range(4), default=3)
    evaluation.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    evaluation.set_defaults(function=evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    _seed_everything(args.seed)
    started = time.time()
    args.function(args, unknown_args)
    print(f"SSVM stage '{args.stage}' completed in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
