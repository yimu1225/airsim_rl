#!/usr/bin/env python3
"""Plot four grouped bar charts for fixed-density evaluation results.

The script reads real per-episode CSV files produced by
``eval.eval_training_density``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The execution environment may not provide a writable default matplotlib
# cache.  This must be configured before importing matplotlib.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/airsim_rl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algo_name_utils import to_output_algorithm_name, to_plot_algorithm_label
matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

DEFAULT_OBSTACLE_COUNTS = (160, 180, 200)
DEFAULT_RESULTS_ROOT = Path(REPO_ROOT) / "results" / "eval" / "training_density_path_length"
FIGURE_WIDTH_INCHES = 3.45
FIGURE_HEIGHT_INCHES = 2.40

PALETTE = [
    "#d62728",  # red
    "#17becf",  # cyan
    "#ff7f0e",  # orange
    "#e377c2",  # pink
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#1f77b4",  # blue
]


def _format_plot_label(algorithm: str) -> str:
    """Match plot_curves.py labels while hiding the curriculum prefix."""
    try:
        label = to_plot_algorithm_label(algorithm)
    except ValueError:
        label = str(algorithm).strip().replace("_", "-")
    return label[3:] if label.startswith("CL-") else label

PLOT_SPECS = {
    "mean_reward": {
        "ylabel": "Average Cumulative Reward",
        "percentage": False,
        "filename": "mean_reward.png",
    },
    "mean_path_length": {
        "ylabel": "Average Path Length /m",
        "percentage": False,
        "filename": "mean_path_length.png",
    },
    "success_rate": {
        "ylabel": "Average Success Rate",
        "percentage": True,
        "filename": "success_rate.png",
    },
    "collision_rate": {
        "ylabel": "Average Collision Rate",
        "percentage": True,
        "filename": "collision_rate.png",
    },
}

REQUIRED_EPISODE_COLUMNS = {
    "episode",
    "reward",
    "episode_length",
    "path_length",
    "is_success",
    "success_rate",
    "collision_rate",
}


def _condition_csv(root: Path, algorithm: str, model_seed: int, obstacle_count: int) -> Path:
    return root / algorithm / f"seed{model_seed}" / f"obstacles_{obstacle_count}_episodes.csv"


def _read_validated_episode_csv(csv_path: Path, expected_episodes: int) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_EPISODE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    if len(frame) != expected_episodes:
        raise ValueError(f"contains {len(frame)} episodes; expected {expected_episodes}")

    for column in ("reward", "episode_length", "path_length", "success_rate", "collision_rate"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"column {column!r} contains non-finite values")
    if (frame["path_length"] < 0.0).any():
        raise ValueError("column 'path_length' contains negative values")

    for column in ("success_rate", "collision_rate"):
        final_rate = float(frame[column].iloc[-1])
        if not 0.0 <= final_rate <= 1.0:
            raise ValueError(f"final {column} must be between 0 and 1, got {final_rate}")
    return frame


def discover_complete_results(
    root: Path,
    *,
    obstacle_counts: Sequence[int],
    algorithms: Optional[Sequence[str]] = None,
    model_seed: Optional[int] = None,
    expected_episodes: int = 100,
) -> Dict[str, Sequence[int]]:
    """Discover algorithms and seeds with all requested density CSV files."""
    if not root.exists():
        raise FileNotFoundError(f"Evaluation results directory does not exist: {root}")

    requested = set(algorithms) if algorithms is not None else None
    discovered: Dict[str, Sequence[int]] = {}
    for algorithm_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        algorithm = algorithm_dir.name
        if algorithm in {"figures", "manifests"}:
            continue
        if requested is not None and algorithm not in requested:
            continue

        complete_seeds = []
        for seed_dir in sorted(path for path in algorithm_dir.glob("seed*") if path.is_dir()):
            seed_text = seed_dir.name.removeprefix("seed")
            try:
                seed = int(seed_text)
            except ValueError:
                print(f"Skipping unrecognized seed directory: {seed_dir}")
                continue
            if model_seed is not None and seed != model_seed:
                continue

            missing_counts = [
                count
                for count in obstacle_counts
                if not _condition_csv(root, algorithm, seed, count).exists()
            ]
            if missing_counts:
                print(
                    f"Skipping incomplete result: algorithm={algorithm}, seed={seed}, "
                    f"missing obstacle counts={missing_counts}"
                )
                continue

            try:
                for count in obstacle_counts:
                    _read_validated_episode_csv(
                        _condition_csv(root, algorithm, seed, count),
                        expected_episodes,
                    )
            except Exception as error:
                print(
                    f"Skipping invalid result: algorithm={algorithm}, seed={seed}, "
                    f"reason={error}"
                )
                continue
            complete_seeds.append(seed)

        if complete_seeds:
            discovered[algorithm] = tuple(complete_seeds)

    if requested is not None:
        for missing_algorithm in sorted(requested.difference(discovered)):
            print(f"Skipping algorithm with no complete results: {missing_algorithm}")
    if not discovered:
        raise FileNotFoundError(
            f"No algorithm has complete data for obstacle counts {list(obstacle_counts)} under {root}."
        )
    return discovered


def aggregate_results(
    root: Path,
    *,
    algorithm_seeds: Mapping[str, Sequence[int]],
    obstacle_counts: Sequence[int],
    expected_episodes: int = 100,
) -> pd.DataFrame:
    """Aggregate all complete seeds into the four requested metrics."""
    if expected_episodes <= 0:
        raise ValueError("expected_episodes must be positive")
    records = []
    for algorithm, seeds in algorithm_seeds.items():
        for obstacle_count in obstacle_counts:
            reward_batches = []
            path_length_batches = []
            success_count = 0.0
            collision_count = 0.0
            total_episodes = 0
            for seed in seeds:
                csv_path = _condition_csv(root, algorithm, seed, obstacle_count)
                frame = _read_validated_episode_csv(csv_path, expected_episodes)
                episodes = len(frame)
                rewards = frame["reward"].to_numpy(dtype=float)
                path_lengths = frame.loc[
                    frame["is_success"] == 1,
                    "path_length",
                ].to_numpy(dtype=float)
                reward_batches.append(rewards)
                path_length_batches.append(path_lengths)
                success_count += float(frame["success_rate"].iloc[-1]) * episodes
                collision_count += float(frame["collision_rate"].iloc[-1]) * episodes
                total_episodes += episodes

            all_rewards = np.concatenate(reward_batches)
            all_path_lengths = np.concatenate(path_length_batches)
            records.append(
                {
                    "algorithm": algorithm,
                    "obstacle_count": int(obstacle_count),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "seed_count": len(seeds),
                    "episodes": total_episodes,
                    "mean_reward": float(np.mean(all_rewards)),
                    "mean_path_length": (
                        float(np.mean(all_path_lengths))
                        if all_path_lengths.size
                        else float("nan")
                    ),
                    "success_rate": success_count / total_episodes,
                    "collision_rate": collision_count / total_episodes,
                }
            )

    return pd.DataFrame.from_records(records)


def _validate_complete_grid(
    summary: pd.DataFrame,
    algorithms: Sequence[str],
    obstacle_counts: Sequence[int],
) -> None:
    expected = {(algorithm, int(count)) for algorithm in algorithms for count in obstacle_counts}
    actual = set(zip(summary["algorithm"], summary["obstacle_count"]))
    missing = expected.difference(actual)
    duplicates = summary.duplicated(["algorithm", "obstacle_count"]).any()
    if missing or duplicates:
        raise ValueError(f"Incomplete or duplicated algorithm-density grid; missing={sorted(missing)}")


def plot_metric(
    summary: pd.DataFrame,
    *,
    metric: str,
    algorithms: Sequence[str],
    obstacle_counts: Sequence[int],
    output_dir: Path,
) -> Path:
    spec = PLOT_SPECS[metric]
    x_positions = np.arange(len(obstacle_counts), dtype=float)
    bar_width = min(0.22, 0.8 / max(len(algorithms), 1))
    offsets = (np.arange(len(algorithms)) - (len(algorithms) - 1) / 2.0) * bar_width
    percentage = bool(spec["percentage"])

    fig, axis = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    plotted_values = []
    for algorithm_index, algorithm in enumerate(algorithms):
        rows = (
            summary.loc[summary["algorithm"] == algorithm]
            .set_index("obstacle_count")
            .loc[list(obstacle_counts)]
        )
        values = rows[metric].to_numpy(dtype=float)
        if percentage:
            values = values * 100.0
        plotted_values.extend(values.tolist())

        axis.bar(
            x_positions + offsets[algorithm_index],
            values,
            width=bar_width,
            color=PALETTE[algorithm_index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.7,
            label=_format_plot_label(algorithm),
            zorder=3,
        )
        for position, value in zip(
            x_positions + offsets[algorithm_index], values
        ):
            if not np.isfinite(value):
                axis.text(
                    position,
                    0.02,
                    "N/A",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.5,
                    fontweight="bold",
                )
    group_half_width = float(np.max(np.abs(offsets))) + bar_width / 2.0
    horizontal_padding = 0.20 * bar_width
    axis.set_xlim(
        x_positions[0] - group_half_width - horizontal_padding,
        x_positions[-1] + group_half_width + horizontal_padding,
    )
    axis.set_xlabel("Number of obstacles", fontsize=9.0, fontweight="bold", labelpad=1.0)
    axis.set_ylabel(str(spec["ylabel"]), fontsize=9.0, fontweight="bold", labelpad=1.5)
    axis.set_xticks(x_positions, [str(value) for value in obstacle_counts])
    axis.tick_params(axis="both", labelsize=8.5, pad=1.0)
    for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick_label.set_fontweight("bold")
    finite_values = np.asarray(plotted_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size:
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
    else:
        min_value, max_value = 0.0, 1.0
    if percentage:
        upper_limit = 105.0 if metric == "success_rate" else max(10.0, max_value * 1.22)
        axis.set_ylim(0.0, max(upper_limit, max_value * 1.25))
    else:
        value_span = max(max_value - min_value, abs(max_value) * 0.1, 1.0)
        lower_limit = min(0.0, min_value - 0.1 * value_span)
        upper_limit = max_value + 0.20 * (max_value - lower_limit)
        axis.set_ylim(lower_limit, upper_limit)
    for spine in axis.spines.values():
        spine.set_visible(True)
    # Keep the x-axis baseline above the white bar edges so its width is uniform.
    axis.spines["bottom"].set_zorder(4)
    handles, labels = axis.get_legend_handles_labels()
    legend_columns = min(3, len(algorithms))
    legend_rows = int(np.ceil(len(handles) / legend_columns))
    legend_order = [
        row * legend_columns + column
        for column in range(legend_columns)
        for row in range(legend_rows)
        if row * legend_columns + column < len(handles)
    ]
    axis.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        frameon=False,
        ncols=legend_columns,
        prop={"family": "Times New Roman", "weight": "bold", "size": 7.0},
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        borderaxespad=0.08,
        columnspacing=0.50,
        handlelength=1.2,
        handletextpad=0.25,
        labelspacing=0.10,
    )
    fig.tight_layout(pad=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(spec["filename"])
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def create_all_plots(
    summary: pd.DataFrame,
    *,
    algorithms: Sequence[str],
    obstacle_counts: Sequence[int],
    output_dir: Path,
) -> Iterable[Path]:
    _validate_complete_grid(summary, algorithms, obstacle_counts)
    for metric in PLOT_SPECS:
        yield plot_metric(
            summary,
            metric=metric,
            algorithms=algorithms,
            obstacle_counts=obstacle_counts,
            output_dir=output_dir,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot four bar charts from fixed-density AirSim evaluation CSV files."
    )
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Optional algorithm directory names; default discovers all complete algorithms.",
    )
    parser.add_argument(
        "--obstacle_counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_OBSTACLE_COUNTS),
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=None,
        help="Optional seed filter; default combines every complete seed found.",
    )
    parser.add_argument("--expected_episodes", type=int, default=100)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    results_root = args.results_root
    output_dir = args.output_dir or (results_root / "figures")
    obstacle_counts = tuple(args.obstacle_counts)

    requested_algorithms = (
        tuple(to_output_algorithm_name(name) for name in args.algorithms)
        if args.algorithms is not None
        else None
    )
    algorithm_seeds = discover_complete_results(
        results_root,
        obstacle_counts=obstacle_counts,
        algorithms=requested_algorithms,
        model_seed=args.model_seed,
        expected_episodes=args.expected_episodes,
    )
    algorithms = tuple(algorithm_seeds)
    # Ensure VSSM-SAC appears first
    if "CL-VSSM-SAC" in algorithms:
        algorithms = ("CL-VSSM-SAC",) + tuple(a for a in algorithms if a != "CL-VSSM-SAC")
    print(
        "Discovered complete results: "
        + ", ".join(
            f"{algorithm} (seeds={list(seeds)})"
            for algorithm, seeds in algorithm_seeds.items()
        )
    )
    summary = aggregate_results(
        results_root,
        algorithm_seeds=algorithm_seeds,
        obstacle_counts=obstacle_counts,
        expected_episodes=args.expected_episodes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "aggregated_metrics.csv"
    summary_for_csv = summary.copy()
    summary_for_csv["mean_path_length"] = summary_for_csv["mean_path_length"].map(
        lambda value: f"{value:.2f}"
    )
    summary_for_csv.to_csv(summary_path, index=False)
    print(f"Aggregated metrics saved to: {summary_path}")

    for figure_path in create_all_plots(
        summary,
        algorithms=algorithms,
        obstacle_counts=obstacle_counts,
        output_dir=output_dir,
    ):
        print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
