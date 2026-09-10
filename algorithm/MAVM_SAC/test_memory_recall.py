#!/usr/bin/env python3
"""Disposable latent-recall diagnostic. No production checkpoints are modified."""
from __future__ import annotations

import argparse
import csv
import json
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import trange


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=str(root / 'datasets/MAVM_SAC'))
    parser.add_argument('--vision-checkpoint', default=str(root / 'runs/MAVM_SAC/vision/vision_latest.pt'))
    parser.add_argument('--output', default=str(root / 'runs/MAVM_SAC/recall_tests' / datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--train-episodes', type=int, default=4)
    parser.add_argument('--val-episodes', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lag', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--encoder-batch', type=int, default=128)
    args = parser.parse_args()
    if min(args.steps, args.lag, args.train_episodes, args.val_episodes, args.log_every, args.encoder_batch) < 1 or args.lr <= 0:
        parser.error('counts, lag and learning rate must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Real Mamba diagnostic requires CUDA')
    # Reuse definitions without executing the training CLI or importing AirSim.
    definitions = runpy.run_path(str(Path(__file__).with_name('train.py')))
    from algorithm.MAVM_SAC.dataset import EpisodeArchive, _split_files
    config_type = definitions['MAVMConfig']
    source = definitions['load_checkpoint'](args.vision_checkpoint, 'cpu')
    if source['stage'] != 'vision':
        raise ValueError('A vision-stage checkpoint is required')
    config = config_type.from_mapping(source['config'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    encoder = definitions['VisionMambaEncoder'](
        config.image_size, channels=config.channels, patch_size=config.patch_size,
        latent_dim=config.latent_dim, depth=config.encoder_depth, d_state=config.d_state,
        drop_rate=config.drop_rate, drop_path_rate=config.drop_path_rate,
    ).cuda().eval().requires_grad_(False)
    encoder.load_state_dict(source['encoder'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    files = EpisodeArchive.list(args.dataset)
    rng = np.random.default_rng(args.seed)
    manifest = {}
    datasets = {}
    for split, count in [('train', args.train_episodes), ('validation', args.val_episodes)]:
        candidates = _split_files(files, split, 0.1, args.seed)
        selected, sequences = [], []
        for index in rng.permutation(len(candidates)):
            path = candidates[int(index)]
            with np.load(path, allow_pickle=False) as archive:
                frames = torch.from_numpy(archive['depth'].copy()).float().div_(255)
            if len(frames) <= args.lag:
                continue
            if tuple(frames.shape[1:]) != (config.channels, *config.image_size):
                raise ValueError(f'Frame dimensions mismatch: {path}')
            with torch.no_grad():
                z = torch.cat([encoder(chunk.cuda()) for chunk in frames.split(args.encoder_batch)])
            if not torch.isfinite(z).all():
                raise ValueError(f'Non-finite encoder output: {path}')
            sequences.append(z.detach())
            selected.append({'file': str(path.resolve()), 'frames': len(z)})
            if len(sequences) == count:
                break
        if len(sequences) != count:
            raise ValueError(f'Not enough valid {split} episodes for lag={args.lag}')
        datasets[split], manifest[split] = sequences, selected
    del encoder, source
    torch.cuda.empty_cache()
    memory = definitions['TemporalMambaMemory'](
        config.latent_dim, config.memory_dim, depth=config.memory_depth,
        d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,
    ).cuda()
    projection = nn.Linear(config.memory_dim, config.latent_dim).cuda()
    # A no-history learned baseline distinguishes recall from current-frame correlations.
    current_only = nn.Linear(config.latent_dim, config.latent_dim).cuda()
    parameters = [*memory.parameters(), *projection.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    baseline_optimizer = torch.optim.AdamW(current_only.parameters(), lr=args.lr, weight_decay=0)
    train_mean = torch.cat([z[:-args.lag] for z in datasets['train']]).mean(0)
    metadata = {'args': vars(args), 'config': config.to_dict(), 'episodes': manifest,
                'target': 'Encoder(observed_depth[t-lag]), not clean depth',
                'loss': 'latent MSE mean; constant LR; fresh Memory/projection; no Decoder'}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    @torch.no_grad()
    def evaluate():
        memory.eval()
        result = {}
        for split, sequences in datasets.items():
            sums = dict(recall=0., linear=0., copy=0., mean=0.)
            pixels = 0
            for z in sequences:
                target = z[:-args.lag]
                predictions = dict(
                    recall=projection(memory(z[None]))[0, args.lag:],
                    linear=current_only(z[args.lag:]), copy=z[args.lag:],
                    mean=train_mean.expand_as(target))
                for name, pred in predictions.items():
                    sums[name] += (pred-target).square().sum().item()
                pixels += target.numel()
            result.update({f'{split}_{name}_mse': value/pixels for name, value in sums.items()})
        memory.train()
        return result

    history = []
    print(f'[recall] output={output} lag={args.lag} train={args.train_episodes} validation={args.val_episodes}')
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = None
        for step in trange(args.steps+1, desc='Latent recall updates'):
            if step:
                z = datasets['train'][(step-1) % len(datasets['train'])]
                target = z[:-args.lag]
                prediction = projection(memory(z[None]))[0, args.lag:]
                loss = (prediction-target).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite recall loss at step {step}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                baseline_loss = (current_only(z[args.lag:])-target).square().mean()
                baseline_optimizer.zero_grad(set_to_none=True)
                baseline_loss.backward()
                baseline_optimizer.step()
            if step % args.log_every == 0 or step == args.steps:
                row = {'step': step, **evaluate()}
                history.append(row)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print('\n[recall] ' + ' '.join(f'{k}={v:.6g}' for k,v in row.items()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ('train', 'validation')):
        for name in ('recall', 'linear', 'copy', 'mean'):
            ax.plot([r['step'] for r in history], [r[f'{split}_{name}_mse'] for r in history], label=name)
        ax.set(title=split, xlabel='Optimizer updates', ylabel='Latent MSE', yscale='log')
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'recall_curves.png', dpi=150)
    plt.close(fig)
    torch.save({'diagnostic_only': True, 'memory': memory.state_dict(), 'projection': projection.state_dict(),
                'current_only': current_only.state_dict(), 'metadata': metadata}, output / 'diagnostic.pt')
    print(f'Diagnostic complete: {output}')


if __name__ == '__main__':
    main()
