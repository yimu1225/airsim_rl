"""CLI and environment runner for CL-VSSM-SAC MambaLRP visualization."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_config
from eval.eval_common import close_env, resolve_checkpoint, set_agent_eval_mode
from eval.eval_env import SceneEvalAirSimEnv

from .attribution import (
    ACTION_LABELS,
    CaptureRecord,
    TrajectoryStep,
    _minimum_pair_gap,
    _obstacle_proximity,
    _physical_action,
    _prepare_depth,
    compute_mambalrp,
    select_spaced_top_indices,
)
from .plotting import (
    _render_action_frames,
    _render_four_frames,
    _render_summary,
    _save_record,
    _write_metadata,
)
from .rules import LRP_GAMMA


DEFAULT_MODEL_SEED = 25
DEFAULT_EPISODE_SEED = 25
DEFAULT_NUM_SAMPLES = 6
DEFAULT_MIN_SAMPLE_GAP = 10
OFFICIAL_MAMBALRP_COMMIT = "b4462a5f6d55ec38a1251683f7ca0f4d2a576e98"


def _load_actor_for_evaluation(agent, checkpoint_path: str) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    actor_encoder = checkpoint.get("actor_encoder")
    actor = checkpoint.get("actor")
    if not isinstance(actor_encoder, dict) or not isinstance(actor, dict):
        raise ValueError(
            "Checkpoint must contain actor_encoder and actor state dictionaries"
        )
    agent.actor_encoder.load_state_dict(actor_encoder, strict=True)
    agent.actor.load_state_dict(actor, strict=True)
    del checkpoint


def _default_checkpoint(model_seed: int) -> str:
    return str(
        REPO_ROOT
        / "models"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / "async_final.pth"
    )


def _default_output_dir(model_seed: int, episode_seed: int) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    return (
        REPO_ROOT
        / "results"
        / "explainability"
        / "mambalrp"
        / "test_scene"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / f"episode{int(episode_seed)}"
        / stamp
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate standalone paper-style MambaLRP explanations for "
            "CL-VSSM-SAC in the static test scene."
        )
    )
    parser.add_argument("--model_seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--episode_seed", type=int, default=DEFAULT_EPISODE_SEED)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--capture_steps", type=int, nargs="+", default=None)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--min_sample_gap", type=int, default=DEFAULT_MIN_SAMPLE_GAP
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.58)
    parser.add_argument(
        "--mask_value",
        type=float,
        default=255.0,
        help=(
            "Depth value used by patch flipping; 255 is free space in this "
            "repository's depth encoding."
        ),
    )
    parser.add_argument(
        "--skip_faithfulness",
        action="store_true",
        help="Skip the paper-style MoRF/LeRF patch-flipping evaluation.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--self_test", action="store_true")
    script_args, remaining = parser.parse_known_args(argv)
    if script_args.num_samples <= 0 or script_args.dpi <= 0:
        parser.error("--num_samples and --dpi must be positive")
    if script_args.min_sample_gap < 0:
        parser.error("--min_sample_gap must be non-negative")
    if not 0.0 <= script_args.overlay_alpha <= 1.0:
        parser.error("--overlay_alpha must be in [0, 1]")
    if script_args.capture_steps is not None:
        script_args.capture_steps = sorted(set(script_args.capture_steps))
        if script_args.capture_steps[0] < 0:
            parser.error("--capture_steps must be non-negative")

    args = get_config(remaining)
    args.algorithm_name = "CL-VSSM-SAC"
    args.seed = int(script_args.model_seed)
    params_path = (
        REPO_ROOT / "algorithm" / "SB_PER_VSSM_SAC" / "params.yaml"
    )
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle) or {}
    params = params.get("params", params)
    if not isinstance(params, dict):
        raise ValueError(f"Algorithm params must be a mapping: {params_path}")
    args.algorithm_params = dict(params)
    for key, value in params.items():
        setattr(args, key, value)
    return script_args, args


def run_visualization(script_args, args) -> Path:
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError(
            "VSSM-SAC MambaLRP requires CUDA because the encoder uses "
            "fused Mamba/Triton kernels."
        )
    from main_async import _configure_reproducibility, get_agent_class

    model_seed = int(script_args.model_seed)
    _configure_reproducibility(model_seed, args)
    output_dir = (
        Path(script_args.output_dir).resolve()
        if script_args.output_dir
        else _default_output_dir(model_seed, int(script_args.episode_seed))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(
        script_args.checkpoint, _default_checkpoint(model_seed)
    )

    env = SceneEvalAirSimEnv(
        takeoff_height=args.takeoff_height,
        config=args,
        stack_frames=int(args.n_frames),
    )
    trajectory: list[TrajectoryStep] = []
    records: list[CaptureRecord] = []
    termination = "max_steps"
    try:
        obs, _ = env.reset(seed=int(script_args.episode_seed))
        depth_shape = tuple(int(value) for value in obs["depth"].shape)
        expected = (int(args.n_frames), 128, 128)
        if depth_shape != expected:
            raise ValueError(
                f"Expected depth {expected}, got {depth_shape}"
            )

        agent_class = get_agent_class(args.algorithm_name)
        agent = agent_class(
            obs["base"].shape[0],
            (1, depth_shape[-2], depth_shape[-1]),
            env.action_space,
            args,
            device=torch.device("cuda"),
            seed=model_seed,
        )
        _load_actor_for_evaluation(agent, checkpoint)
        set_agent_eval_mode(agent)
        print(f"[MambaLRP] Loaded actor model: {checkpoint}")

        max_steps = int(getattr(args, "episode_length", 300))
        for step in range(max_steps):
            base = np.asarray(obs["base"], dtype=np.float32)
            depth = _prepare_depth(obs["depth"])
            action = _physical_action(agent, base, depth)
            trajectory.append(
                TrajectoryStep(
                    step=step,
                    base_state=base.copy(),
                    depth=depth.copy(),
                    physical_action=action.copy(),
                    obstacle_proximity=_obstacle_proximity(depth),
                )
            )
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if bool(info.get("is_success", False)):
                    termination = "success"
                elif bool(info.get("has_collided", False)):
                    termination = "collision"
                elif truncated:
                    termination = "timeout"
                else:
                    termination = "other_failure"
                print(
                    f"[MambaLRP] Episode ended at step {step}: "
                    f"{termination}"
                )
                break
        if not trajectory:
            raise RuntimeError("The test episode produced no trajectory")

        if script_args.capture_steps is None:
            selected_indices = select_spaced_top_indices(
                [item.obstacle_proximity for item in trajectory],
                count=int(script_args.num_samples),
                min_gap=int(script_args.min_sample_gap),
            )
            selection_method = (
                "top_obstacle_proximity_with_temporal_spacing"
            )
            missing_steps: list[int] = []
        else:
            requested = set(script_args.capture_steps)
            selected_indices = [
                index
                for index, item in enumerate(trajectory)
                if item.step in requested
            ]
            completed = {
                trajectory[index].step for index in selected_indices
            }
            missing_steps = sorted(requested - completed)
            selection_method = "explicit_capture_steps"
        if not selected_indices:
            raise RuntimeError("No selected samples exist in the trajectory")

        selected_steps = [
            trajectory[index].step for index in selected_indices
        ]
        print(
            "[MambaLRP] Selected steps: "
            + ", ".join(map(str, selected_steps))
        )
        for index in selected_indices:
            sample = trajectory[index]
            print(f"[MambaLRP] Explaining step {sample.step}")
            result = compute_mambalrp(
                agent,
                sample.base_state,
                sample.depth,
                evaluate_faithfulness=not script_args.skip_faithfulness,
                mask_value=float(script_args.mask_value),
            )
            record = CaptureRecord(sample=sample, result=result)
            records.append(record)
            _save_record(record, output_dir)
            _render_four_frames(
                record,
                output_dir / f"step_{sample.step:04d}_four_frames.png",
                alpha=float(script_args.overlay_alpha),
                dpi=int(script_args.dpi),
            )
            _render_action_frames(
                record,
                output_dir / f"step_{sample.step:04d}_actions.png",
                alpha=float(script_args.overlay_alpha),
                dpi=int(script_args.dpi),
            )
        _render_summary(
            records,
            output_dir / "current_frame_summary.png",
            alpha=float(script_args.overlay_alpha),
            dpi=int(script_args.dpi),
        )

        metadata = {
            "algorithm": "CL-VSSM-SAC",
            "method": "MambaLRP",
            "model_seed": model_seed,
            "checkpoint": os.path.abspath(checkpoint),
            "environment": "static_test_environment",
            "episode_seed": int(script_args.episode_seed),
            "termination": termination,
            "trajectory_steps": len(trajectory),
            "sample_selection": selection_method,
            "capture_steps_requested": script_args.capture_steps,
            "capture_steps_completed": selected_steps,
            "capture_steps_missing": missing_steps,
            "preferred_min_sample_gap": int(script_args.min_sample_gap),
            "effective_min_sample_gap": _minimum_pair_gap(selected_steps),
            "gamma": LRP_GAMMA,
            "signed_relevance": True,
            "raw_relevance_unit": "input_depth_pixel",
            "display_colormap": "jet",
            "display_range": [-1.0, 1.0],
            "display_interpolation": "none",
            "faithfulness_evaluation": (
                "disabled"
                if script_args.skip_faithfulness
                else "MoRF_and_LeRF_patch_flipping"
            ),
            "faithfulness_mask_value_depth_units": float(
                script_args.mask_value
            ),
            "action_labels": list(ACTION_LABELS),
            "method_details": {
                str(record.sample.step): record.result.details
                for record in records
            },
            "reference": {
                "paper": (
                    "Jafari et al., MambaLRP: Explaining Selective State "
                    "Space Sequence Models, NeurIPS 2024"
                ),
                "official_repository": (
                    "https://github.com/FarnoushRJ/MambaLRP"
                ),
                "official_repository_commit": OFFICIAL_MAMBALRP_COMMIT,
                "vision_configuration": (
                    "Appendix C.2-C.2.1: generalized LRP-gamma with "
                    "gamma=0.25 on Vision-Mamba convolution layers only"
                ),
            },
        }
        _write_metadata(output_dir / "metadata.json", metadata)
        if missing_steps:
            print(
                "[MambaLRP] Missing explicit steps: "
                + ", ".join(map(str, missing_steps))
            )
        print(f"[MambaLRP] Results saved to: {output_dir}")
        return output_dir
    finally:
        close_env(env, label="CL-VSSM-SAC MambaLRP visualization")



def main(argv=None) -> None:
    script_args, args = _parse_args(argv)
    if script_args.self_test:
        from .self_test import run_self_tests

        run_self_tests()
        return
    run_visualization(script_args, args)


if __name__ == "__main__":
    main()
