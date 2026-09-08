#!/usr/bin/env python3
"""Fixed-frame SSE overfit diagnostic; no changes to production checkpoints."""
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
    mavm_package = types.ModuleType("algorithm.MAVM_SAC")
    mavm_package.__path__ = [str(_HERE)]
    sys.modules.setdefault("algorithm.MAVM_SAC", mavm_package)
    visual_package = types.ModuleType("algorithm.MAVM_SAC.visualization")
    visual_package.__path__ = [str(_HERE / "visualization")]
    sys.modules.setdefault("algorithm.MAVM_SAC.visualization", visual_package)
    __package__ = "algorithm.MAVM_SAC.visualization"


import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import trange
from ..config import MAVMConfig
from ..networks import VisionMambaEncoder
from ..dataset import EpisodeArchive, _split_files
from ..checkpoints import atomic_torch_save, MODEL_VERSION
from .visualize_memory_reconstruction import _PROJECT_ROOT, _build_vision_decoder

def select_frames(root, count, seed):
    files = _split_files(EpisodeArchive.list(root), "train", 0.1, seed)
    if count <= 0 or count > len(files):
        raise ValueError("samples must be positive and no greater than training episode count")
    rng = np.random.default_rng(seed)
    inputs, targets, records = [], [], []
    for index in rng.choice(len(files), count, replace=False):
        path = files[int(index)]
        with np.load(path, allow_pickle=False) as archive:
            t = len(archive["depth"]) // 2
            inputs.append(archive["depth"][t].copy())
            name = "clean_depth" if "clean_depth" in archive else "depth"
            targets.append(archive[name][t].copy())
            records.append({"episode": path.name, "frame": t, "target": name})
    return torch.from_numpy(np.stack(inputs)).float()/255, torch.from_numpy(np.stack(targets)).float()/255, records

def save_grid(path, inputs, targets, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(inputs)
    fig, axes = plt.subplots(3, n, figsize=(2.5*n, 7), squeeze=False)
    for i in range(n):
        for row, (data, label) in enumerate(((inputs,"Input"), (targets,"Target"), (predictions,"Reconstruction"))):
            axes[row,i].imshow(data[i,0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row,i].set_title(f"{label} #{i}")
            axes[row,i].axis("off")
        mae = (predictions[i]-targets[i]).abs().mean().item()
        axes[2,i].set_title(f"Reconstruction #{i}\nMAE={mae:.5f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(_PROJECT_ROOT/"datasets/MAVM_SAC"))
    parser.add_argument("--config", default=str(_PROJECT_ROOT/"algorithm/MAVM_SAC/params.yaml"))
    parser.add_argument("--output", default=str(_PROJECT_ROOT/"runs/MAVM_SAC/vision_overfit"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1:
        parser.error("steps and log-every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This real-Mamba diagnostic requires CUDA")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = MAVMConfig.load(args.config)
    x, y, records = select_frames(args.dataset, args.samples, args.seed)
    if tuple(x.shape[1:]) != (config.channels, *config.image_size):
        raise ValueError("dataset shape differs from config")
    (output/"samples.json").write_text(json.dumps(records, indent=2))
    (output/"config.json").write_text(json.dumps(config.to_dict(), indent=2))
    x, y = x.cuda(), y.cuda()
    encoder = VisionMambaEncoder(config.image_size, channels=config.channels,
        patch_size=config.patch_size, latent_dim=config.latent_dim,
        depth=config.encoder_depth, d_state=config.d_state,
        drop_rate=config.drop_rate, drop_path_rate=config.drop_path_rate).cuda()
    decoder = _build_vision_decoder(config).cuda()
    parameters = [*encoder.parameters(), *decoder.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config.vision_lr)
    with (output/"metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step","sse","mse","mae","grad_norm"])
        writer.writeheader()
        progress = trange(args.steps+1, desc="Vision overfit", unit="update")
        for step in progress:
            if step:
                encoder.train(); decoder.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = decoder(encoder(x))
                loss = (prediction-y).square().sum()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
                optimizer.step()
            if step % args.log_every == 0 or step == args.steps:
                encoder.eval(); decoder.eval()
                with torch.no_grad():
                    prediction = decoder(encoder(x))
                    error = prediction-y
                    metrics = dict(step=step, sse=error.square().sum().item(),
                        mse=error.square().mean().item(), mae=error.abs().mean().item(),
                        grad_norm=float(norm) if step else 0.)
                    writer.writerow(metrics); stream.flush()
                    save_grid(output/f"step_{step:06d}.png",x,y,prediction)
                progress.set_postfix(mse=metrics["mse"], mae=metrics["mae"])
                print(metrics, flush=True)
    atomic_torch_save(dict(stage="vision", model_version=MODEL_VERSION,
        config=config.to_dict(), encoder=encoder.state_dict(), decoder=decoder.state_dict(),
        optimizer=optimizer.state_dict(), step=args.steps, diagnostic=True), output/"overfit.pt")
    print(f"Saved diagnostic to {output}", flush=True)

if __name__ == "__main__":
    main()
