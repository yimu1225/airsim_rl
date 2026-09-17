#!/usr/bin/env python3
"""Standalone stage-3 Vision reconstruction test."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    import types
    from pathlib import Path

    _HERE = Path(__file__).resolve().parent.parent
    _ROOT = _HERE.parents[1]
    sys.path.insert(0, str(_ROOT))
    algorithm_package = types.ModuleType("algorithm")
    algorithm_package.__path__ = [str(_HERE.parent)]
    sys.modules.setdefault("algorithm", algorithm_package)
    ssvm_package = types.ModuleType("algorithm.SSVM_SAC")
    ssvm_package.__path__ = [str(_HERE)]
    sys.modules.setdefault("algorithm.SSVM_SAC", ssvm_package)
    visual_package = types.ModuleType("algorithm.SSVM_SAC.visualization")
    visual_package.__path__ = [str(_HERE / "visualization")]
    sys.modules.setdefault("algorithm.SSVM_SAC.visualization", visual_package)
    __package__ = "algorithm.SSVM_SAC.visualization"

import argparse
from pathlib import Path
import numpy as np
import torch
from .visualize_memory_reconstruction import (
    _PROJECT_ROOT, _device, _build_vision_decoder, _select_episode,
    _select_time_index, _image_array,
)
from ..checkpoints import load_checkpoint
from ..config import SSVMConfig
from ..dataset import EpisodeSequenceDataset
from ..networks import VisionMambaEncoder

@torch.inference_mode()
def visualize_vision(args: argparse.Namespace) -> Path:
    """Test the actual stage-3 encoder/decoder pair without a memory model."""
    device = _device(args.device)
    checkpoint = load_checkpoint(args.vision_checkpoint, device)
    if checkpoint["stage"] != "vision":
        raise ValueError("--vision-checkpoint must be a stage-3 checkpoint")
    config = SSVMConfig.from_mapping(checkpoint["config"])
    encoder = VisionMambaEncoder(
        config.image_size, channels=config.channels, patch_size=config.patch_size,
        latent_dim=config.latent_dim, depth=config.encoder_depth, d_state=config.d_state,
        drop_rate=config.drop_rate, drop_path_rate=config.drop_path_rate,
    ).to(device).eval()
    decoder = _build_vision_decoder(config).to(device).eval()
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    dataset = EpisodeSequenceDataset(
        args.dataset, sequence_length=None, split=args.split, seed=args.seed,
        use_clean_targets=not args.observed_targets,
    )
    if not dataset:
        raise ValueError(f"no episodes in {args.split} split")
    if args.episode_index is None:
        indices = np.linspace(0, len(dataset) - 1, min(4, len(dataset)), dtype=int)
    else:
        indices = [args.episode_index]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, len(indices), figsize=(4 * len(indices), 10), squeeze=False)
    try:
        for column, index in enumerate(indices):
            _, sample = _select_episode(dataset, (0,), int(index))
            t = _select_time_index(len(sample["depth"]), (0,), args.time_index)
            frame = sample["depth"][t:t+1].to(device=device, dtype=torch.float32) / 255.0
            target = sample["target_depth"][t].float().numpy() / 255.0
            prediction = decoder(encoder(frame))[0].cpu().numpy()
            mae = float(np.abs(prediction - target).mean() * 255.0)
            for row, (value, title) in enumerate((
                (frame[0].cpu().numpy() * 255.0, f"Observed: episode {index}, t={t}"),
                (target * 255.0, "Target (clean depth when available)"),
                (prediction * 255.0, f"Vision reconstruction\nMAE={mae:.2f}"),
            )):
                axes[row, column].imshow(_image_array(value), cmap=args.cmap, vmin=0, vmax=255)
                axes[row, column].set_title(title)
                axes[row, column].axis("off")
            print(f"[vision test] episode={index} t={t} mae={mae:.2f}")
        figure.suptitle(f"Stage 3 — checkpoint epoch {checkpoint.get('epoch', '?')} — {args.split}")
        figure.tight_layout()
        output = Path(args.output) if args.output else _PROJECT_ROOT / "runs/SSVM_SAC/vision/reconstruction_test.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
    finally:
        plt.close(figure)
    print(f"Saved Vision reconstruction grid to {output}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Test stage-3 Vision reconstruction")
    parser.add_argument("--dataset", default=str(_PROJECT_ROOT / "datasets/SSVM_SAC"))
    parser.add_argument("--vision-checkpoint", default=str(_PROJECT_ROOT / "runs/SSVM_SAC/vision/vision_latest.pt"))
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "runs/SSVM_SAC/vision/reconstruction_test.png"))
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--time-index", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--observed-targets", action="store_true")
    parser.add_argument(
        "--cmap", default="jet_r",
        help="matplotlib colormap for depth rendering (default: jet_r)",
    )
    return parser


if __name__ == "__main__":
    visualize_vision(build_parser().parse_args())
