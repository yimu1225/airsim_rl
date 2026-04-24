import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from tensorboard.backend.event_processing import event_accumulator

from config import get_config


METRIC_TAGS = {
    "meta_weight_entropy": "adaptive/meta_weight_entropy",
    "meta_weight_max": "adaptive/meta_weight_max",
    "adaptive_reg": "adaptive/reg",
}


def expand_algorithms(algo_str):
    groups = {
        "all": [
            "td3",
            "ddpg",
            "aetd3",
            "per_td3",
            "per_aetd3",
            "cfc_td3",
            "ST-VimTD3",
            "stv_patch_td3",
            "Vim-TD3",
            "PER-ST-VimTD3",
            "ST-SVimTD3",
            "mamba_td3",
            "gam_mamba_td3",
            "gam_td3",
            "ST_3DVimTD3",
        ],
        "base": ["td3", "ddpg", "aetd3", "per_td3", "per_aetd3"],
        "seq": ["cfc_td3", "ST-VimTD3", "stv_patch_td3", "Vim-TD3", "PER-ST-VimTD3", "ST-SVimTD3", "mamba_td3", "ST_3DVimTD3"],
    }

    if algo_str in groups:
        return groups[algo_str]
    if "," in algo_str:
        return [a.strip() for a in algo_str.split(",") if a.strip()]
    return [algo_str.strip()]


def smooth_values(values, method="moving", window=10, smooth_alpha=0.99):
    if len(values) == 0:
        return values
    if method == "moving":
        if window <= 1 or len(values) < window:
            return values
        series = pd.Series(values)
        return series.rolling(window, center=True, min_periods=1).mean().to_numpy()

    tb_val = float(np.clip(smooth_alpha, 0.0, 0.9999))
    alpha = 1.0 - tb_val
    return pd.Series(values).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def load_scalar_series(run_dir, tag):
    try:
        acc = event_accumulator.EventAccumulator(run_dir)
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            return None, None
        events = acc.Scalars(tag)
        if not events:
            return None, None
        steps = np.array([e.step for e in events], dtype=np.float64)
        values = np.array([e.value for e in events], dtype=np.float64)
        return steps, values
    except Exception:
        return None, None


def gather_runs(results_dir, algorithms, plot_cl=True, plot_non_cl=True, seeds_to_plot=None):
    selected = {}
    for algo in algorithms:
        variants = []
        if plot_non_cl:
            variants.append(algo)
        if plot_cl:
            variants.append(f"CL-{algo}")

        for variant in variants:
            run_dirs = sorted(glob.glob(os.path.join(results_dir, f"{variant}_seed*")))
            for run_dir in run_dirs:
                base = os.path.basename(run_dir)
                if "_seed" not in base:
                    continue
                seed = base.split("_seed", 1)[1]
                if seeds_to_plot is not None and seed not in seeds_to_plot:
                    continue
                selected.setdefault(variant, []).append(run_dir)
    return selected


def plot_metric(ax, runs, metric_key, smooth_method, smooth_window, smooth_alpha, ci_type, n_interp_points, color, label):
    tag = METRIC_TAGS[metric_key]
    curves = []
    all_steps = []

    for run in runs:
        steps, values = load_scalar_series(run, tag)
        if steps is None or values is None or len(steps) < 2:
            continue
        values = smooth_values(values, method=smooth_method, window=smooth_window, smooth_alpha=smooth_alpha)
        curves.append((steps, values))
        all_steps.append(steps)

    if len(curves) == 0:
        return False

    min_step = max(s[0] for s in all_steps)
    max_step = min(s[-1] for s in all_steps)
    if min_step >= max_step:
        return False

    x_common = np.linspace(min_step, max_step, n_interp_points)
    ys = []
    for steps, values in curves:
        f = interpolate.interp1d(steps, values, kind="linear", bounds_error=False, fill_value="extrapolate")
        ys.append(f(x_common))
    ys = np.array(ys)

    mean = np.mean(ys, axis=0)
    std = np.std(ys, axis=0)
    n = ys.shape[0]
    ci = std if ci_type == "std" else std / np.sqrt(max(n, 1))

    ax.plot(x_common, mean, linewidth=2.0, color=color, label=label)
    ax.fill_between(x_common, mean - ci, mean + ci, color=color, alpha=0.2)
    return True


def main():
    args = get_config()

    results_dir = "./results"
    algorithms = expand_algorithms(args.algorithm_name)
    seeds_to_plot = None
    if isinstance(args.seed, list):
        seeds_to_plot = [str(s) for s in args.seed]
    elif isinstance(args.seed, int):
        seeds_to_plot = [str(args.seed)]

    run_groups = gather_runs(
        results_dir,
        algorithms,
        plot_cl=args.plot_cl,
        plot_non_cl=args.plot_non_cl,
        seeds_to_plot=seeds_to_plot,
    )

    if not run_groups:
        print("No matching run folders found under ./results")
        return

    metric_titles = {
        "meta_weight_entropy": "Meta Weight Entropy",
        "meta_weight_max": "Meta Weight Max",
        "adaptive_reg": "Adaptive Reg",
    }

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_groups)))

    any_plotted = False
    for idx, (algo_variant, runs) in enumerate(sorted(run_groups.items())):
        color = colors[idx]
        for ax, metric_key in zip(axes, METRIC_TAGS.keys()):
            plotted = plot_metric(
                ax=ax,
                runs=runs,
                metric_key=metric_key,
                smooth_method=args.smooth_method,
                smooth_window=args.smooth_window,
                smooth_alpha=args.smooth_alpha,
                ci_type=args.ci_type,
                n_interp_points=600,
                color=color,
                label=algo_variant,
            )
            any_plotted = any_plotted or plotted

    if not any_plotted:
        print("No scalar data found for tags:", list(METRIC_TAGS.values()))
        print("Hint: run training after updating main_async.py to log adaptive/* tags.")
        return

    for ax, metric_key in zip(axes, METRIC_TAGS.keys()):
        ax.set_title(metric_titles[metric_key])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Total Timesteps")
    plt.tight_layout()

    out_path = os.path.join(results_dir, "aetd3_meta_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
