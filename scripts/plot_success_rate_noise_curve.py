#!/usr/bin/env python3
"""
Plot the exploration-noise curve as a function of success rate.

The schedule uses a direct log function on [0.5, 1.0]:

    noise(s) = a + b * log(s)

with endpoint constraints:
    noise(0.5) = noise_max
    noise(1.0) = noise_min

which gives:
    a = noise_min
    b = (noise_max - noise_min) / log(0.5)

This guarantees monotonic decrease from noise_max to noise_min on [0.5, 1.0].
For s <= 0.5, noise keeps noise_max.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def noise_schedule(success_rate, noise_max, noise_min):
    """Compute noise intensity from success rate using noise = a + b*log(s)."""
    s = np.asarray(success_rate, dtype=np.float64)

    noise_max = float(noise_max)
    noise_min = float(noise_min)
    if noise_min > noise_max:
        raise ValueError("exploration_noise_final must be <= exploration_noise")

    noise = np.full_like(s, noise_max, dtype=np.float64)
    active = s > 0.5
    if np.any(active):
        a = noise_min
        b = (noise_max - noise_min) / np.log(0.5)
        noise[active] = a + b * np.log(s[active])

    return noise


def build_parser():
    parser = argparse.ArgumentParser(description="Plot success-rate vs exploration-noise curve")
    parser.add_argument("--exploration-noise", type=float, default=0.2, help="Maximum exploration noise")
    parser.add_argument("--exploration-noise-final", type=float, default=0.1, help="Minimum exploration noise")
    parser.add_argument("--num-points", type=int, default=1001, help="Number of points on x-axis")
    parser.add_argument("--dpi", type=int, default=200, help="Figure dpi")
    parser.add_argument(
        "--output",
        type=str,
        default="results/success_rate_noise_curve.png",
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Success Rate vs Exploration Noise",
        help="Figure title",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    return parser


def main():
    args = build_parser().parse_args()

    x = np.linspace(0.0, 1.0, max(2, int(args.num_points)))
    y = noise_schedule(
        success_rate=x,
        noise_max=args.exploration_noise,
        noise_min=args.exploration_noise_final,
    )

    noise_max = float(args.exploration_noise)
    noise_min = float(args.exploration_noise_final)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, linewidth=2.2, label="noise schedule")
    plt.axvline(0.5, linestyle="--", linewidth=1.4, color="tab:orange", label="decay start (0.5)")
    plt.axvline(1.0, linestyle="--", linewidth=1.4, color="tab:purple", label="decay end (1.0)")
    plt.axhline(noise_max, linestyle=":", linewidth=1.2, color="tab:green", label="noise max")
    plt.axhline(noise_min, linestyle=":", linewidth=1.2, color="tab:red", label="noise min")
    plt.xlabel("Success rate")
    plt.ylabel("Noise intensity")
    plt.title(args.title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    print(f"Saved figure: {output_path}")

    # Print key points for quick sanity check.
    for s in [0.0, 0.5, 0.75, 1.0]:
        val = noise_schedule([s], noise_max, noise_min)[0]
        print(f"success_rate={s:.3f} -> noise={val:.6f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
