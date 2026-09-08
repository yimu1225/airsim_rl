#!/usr/bin/env python3
"""Visualize stage-4 temporal reconstructions alongside their target frames."""

from __future__ import annotations

# Keep this entry point independent from algorithm/__init__.py, like train.py.
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
    mavm_package = types.ModuleType("algorithm.MAVM_SAC")
    mavm_package.__path__ = [str(_HERE)]
    sys.modules.setdefault("algorithm.MAVM_SAC", mavm_package)
    visual_package = types.ModuleType("algorithm.MAVM_SAC.visualization")
    visual_package.__path__ = [str(_HERE / "visualization")]
    sys.modules.setdefault("algorithm.MAVM_SAC.visualization", visual_package)
    __package__ = "algorithm.MAVM_SAC.visualization"

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..checkpoints import load_checkpoint, perception_from_checkpoint
from ..config import MAVMConfig
from ..dataset import EpisodeSequenceDataset
from ..networks import MultiFrameMambaReconstructor, VisionMambaDecoder


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _valid_time_bounds(length: int, offsets: Sequence[int]) -> tuple[int, int]:
    """Return inclusive t bounds for which every t+offset exists."""
    if length <= 0:
        raise ValueError("episode must contain at least one frame")
    minimum = max((max(0, -int(offset)) for offset in offsets), default=0)
    maximum = min(
        (min(length - 1, length - 1 - int(offset)) for offset in offsets),
        default=length - 1,
    )
    if minimum > maximum:
        raise ValueError(
            f"episode length {length} is too short for offsets {tuple(offsets)}"
        )
    return minimum, maximum


def _select_time_index(
    length: int, offsets: Sequence[int], requested: int | None
) -> int:
    minimum, maximum = _valid_time_bounds(length, offsets)
    if requested is None:
        return (minimum + maximum) // 2
    if not minimum <= requested <= maximum:
        raise ValueError(
            f"time index {requested} cannot show every target; "
            f"valid range is {minimum}..{maximum}"
        )
    return requested


def _display_offsets(offsets: Sequence[int]) -> tuple[int, ...]:
    """Use chronological display order and insert the current frame."""
    return tuple(sorted({0, *(int(offset) for offset in offsets)}))


def _build_vision_decoder(config: MAVMConfig) -> VisionMambaDecoder:
    return VisionMambaDecoder(
        config.latent_dim,
        config.image_size,
        patch_size=config.patch_size,
        channels=config.channels,
        embed_dim=config.decoder_embed_dim,
        depth=config.decoder_depth,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
    )


def _build_temporal_reconstructor(
    config: MAVMConfig,
) -> MultiFrameMambaReconstructor:
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
    )
    return MultiFrameMambaReconstructor(
        config.memory_dim,
        config.reconstruction_latent_dim,
        config.reconstruction_offsets,
        decoder,
    )


def _check_encoder_compatibility(
    vision_config: MAVMConfig, memory_config: MAVMConfig
) -> None:
    fields = (
        "image_height",
        "image_width",
        "channels",
        "patch_size",
        "latent_dim",
        "encoder_depth",
        "d_state",
    )
    mismatches = [
        field
        for field in fields
        if getattr(vision_config, field) != getattr(memory_config, field)
    ]
    if mismatches:
        raise ValueError(
            "vision and memory checkpoints have incompatible encoders: "
            + ", ".join(mismatches)
        )


def _select_episode(
    dataset: EpisodeSequenceDataset,
    offsets: Sequence[int],
    requested: int | None,
) -> tuple[int, dict[str, torch.Tensor]]:
    if requested is not None:
        if not 0 <= requested < len(dataset):
            raise IndexError(
                f"episode index {requested} is outside 0..{len(dataset) - 1}"
            )
        sample = dataset[requested]
        _valid_time_bounds(int(sample["valid"].sum()), offsets)
        return requested, sample

    for index in range(len(dataset)):
        sample = dataset[index]
        try:
            _valid_time_bounds(int(sample["valid"].sum()), offsets)
        except ValueError:
            continue
        return index, sample
    raise ValueError(f"no episode is long enough for offsets {tuple(offsets)}")


def _image_array(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] == 1:
        return image[0]
    if image.ndim == 3 and image.shape[0] in (3, 4):
        return np.moveaxis(image, 0, -1)
    return image


def _offset_label(offset: int) -> str:
    if offset == 0:
        return r"$I_t$"
    sign = "+" if offset > 0 else ""
    return rf"$I_{{t{sign}{offset}}}$"


def _save_grid(
    originals: dict[int, np.ndarray],
    reconstructions: dict[int, np.ndarray],
    output: str | Path,
    *,
    episode_index: int,
    time_index: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offsets = tuple(originals)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        len(offsets),
        figsize=(4.0 * len(offsets), 7.0),
        squeeze=False,
        constrained_layout=True,
    )
    for column, offset in enumerate(offsets):
        target = _image_array(originals[offset])
        reconstruction = _image_array(reconstructions[offset])
        mae = float(np.mean(np.abs(reconstruction - target)))
        axes[0, column].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(f"Target {_offset_label(offset)}")
        axes[1, column].imshow(
            reconstruction, cmap="gray", vmin=0.0, vmax=1.0
        )
        source = "Vision AE" if offset == 0 else "Temporal reconstruction"
        axes[1, column].set_title(f"{source}\nMAE={mae:.5f}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.suptitle(
        f"MAVM-SAC reconstruction — episode {episode_index}, t={time_index}"
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


@torch.inference_mode()
def visualize(args: argparse.Namespace) -> Path:
    device = _device(args.device)
    vision_checkpoint = load_checkpoint(args.vision_checkpoint, device)
    memory_checkpoint = load_checkpoint(args.memory_checkpoint, device)
    if vision_checkpoint["stage"] != "vision":
        raise ValueError("--vision-checkpoint must be a stage-3 checkpoint")
    if memory_checkpoint["stage"] != "memory":
        raise ValueError("--memory-checkpoint must be a stage-4 checkpoint")
    if "decoder" not in vision_checkpoint:
        raise ValueError("vision checkpoint does not contain its decoder")
    if "reconstructor" not in memory_checkpoint:
        raise ValueError("memory checkpoint does not contain its reconstructor")

    vision_config = MAVMConfig.from_mapping(vision_checkpoint["config"])
    memory_config, perception = perception_from_checkpoint(
        memory_checkpoint, device, freeze=True
    )
    _check_encoder_compatibility(vision_config, memory_config)
    current_decoder = _build_vision_decoder(vision_config).to(device)
    current_decoder.load_state_dict(vision_checkpoint["decoder"])
    current_decoder.eval().requires_grad_(False)
    temporal_reconstructor = _build_temporal_reconstructor(memory_config).to(device)
    temporal_reconstructor.load_state_dict(memory_checkpoint["reconstructor"])
    temporal_reconstructor.eval().requires_grad_(False)

    dataset = EpisodeSequenceDataset(
        args.dataset,
        sequence_length=None,
        split=args.split,
        seed=args.seed,
        use_clean_targets=not args.observed_targets,
    )
    if not dataset:
        raise ValueError(
            f"no episodes found in the {args.split!r} split of {args.dataset}"
        )
    episode_index, sample = _select_episode(
        dataset, memory_config.reconstruction_offsets, args.episode_index
    )
    length = int(sample["valid"].sum())
    time_index = _select_time_index(
        length, memory_config.reconstruction_offsets, args.time_index
    )

    frames = sample["depth"][: time_index + 1].to(
        device=device, dtype=torch.float32
    ).div_(255.0)
    target_frames = sample["target_depth"][:length].to(
        dtype=torch.float32
    ).div_(255.0)
    latents = perception.encoder(frames)
    memories = perception.memory(latents.unsqueeze(0))
    temporal_predictions = temporal_reconstructor(memories[:, -1])
    current_prediction = current_decoder(latents[-1:])

    display_offsets = _display_offsets(memory_config.reconstruction_offsets)
    originals = {
        offset: target_frames[time_index + offset].cpu().numpy()
        for offset in display_offsets
    }
    reconstructions = {
        offset: (
            current_prediction[0].cpu().numpy()
            if offset == 0
            else temporal_predictions[offset][0].cpu().numpy()
        )
        for offset in display_offsets
    }
    output = _save_grid(
        originals,
        reconstructions,
        args.output or _PROJECT_ROOT / "runs/MAVM_SAC/memory/reconstruction_test.png",
        episode_index=episode_index,
        time_index=time_index,
    )
    target_kind = "observed depth" if args.observed_targets else "training targets"
    print(
        f"Saved reconstruction grid to {output}\n"
        f"device={device} split={args.split} episode={episode_index} "
        f"t={time_index} offsets={display_offsets} targets={target_kind}"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render stage-4 MAVM temporal reconstruction results"
    )
    parser.add_argument(
        "--dataset", default=str(_PROJECT_ROOT / "datasets" / "MAVM_SAC")
    )
    parser.add_argument(
        "--vision-checkpoint",
        default=str(
            _PROJECT_ROOT
            / "runs"
            / "MAVM_SAC"
            / "vision"
            / "vision_latest.pt"
        ),
    )
    parser.add_argument(
        "--memory-checkpoint",
        default=str(
            _PROJECT_ROOT
            / "runs"
            / "MAVM_SAC"
            / "memory"
            / "memory_latest.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "all"), default="validation"
    )
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--time-index", type=int)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--observed-targets",
        action="store_true",
        help="compare against noisy observed depth instead of clean training targets",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    visualize(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
