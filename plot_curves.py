import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import get_config
from algo_name_utils import (
    expand_algorithm_spec,
    split_curriculum_prefix,
    to_internal_algorithm_name,
    to_kebab_algorithm_name,
)


def _safe_to_internal_algo_name(name):
    try:
        return to_internal_algorithm_name(name)
    except ValueError:
        return str(name).strip()


def _safe_to_plot_label(name):
    try:
        return to_kebab_algorithm_name(name, upper=False)
    except ValueError:
        return str(name).strip().replace("_", "-").upper()

def _prepare_xy(x_values, y_values):
    """清洗并排序单条曲线，保证 x 严格递增。"""
    x_arr = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y_values, dtype=np.float64).reshape(-1)

    if x_arr.size != y_arr.size or x_arr.size < 2:
        return None

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    if x_arr.size < 2:
        return None

    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_idx]
    if unique_x.size < 2:
        return None

    return unique_x, unique_y


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=1e-8):
    """baselines 同款单侧 EMA 重采样。"""
    xolds = np.asarray(xolds, dtype=np.float64)
    yolds = np.asarray(yolds, dtype=np.float64)
    n = max(2, int(n))
    decay_steps = max(float(decay_steps), 1e-6)

    low = float(xolds[0]) if low is None else float(low)
    high = float(xolds[-1]) if high is None else float(high)

    if low >= high:
        xnews = np.linspace(low, low + 1.0, n)
        ys = np.full_like(xnews, yolds.mean(), dtype=np.float64)
        counts = np.ones_like(xnews, dtype=np.float64)
        return xnews, ys, counts

    luoi = 0
    sum_y = 0.0
    count_y = 0.0
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(-1.0 / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)

    for i, xnew in enumerate(xnews):
        sum_y *= interstep_decay
        count_y *= interstep_decay

        while luoi < len(xolds) and xolds[luoi] <= xnew:
            decay = np.exp(-(xnew - xolds[luoi]) / decay_period)
            sum_y += decay * yolds[luoi]
            count_y += decay
            luoi += 1

        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = np.divide(sum_ys, count_ys, out=np.full_like(sum_ys, np.nan), where=count_ys > 0)
    ys[count_ys < low_counts_threshold] = np.nan
    return xnews, ys, count_ys


def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=1e-8):
    """baselines 同款双侧 EMA 重采样（对称平滑）。"""
    xs, ys1, count_ys1 = one_sided_ema(
        xolds,
        yolds,
        low=low,
        high=high,
        n=n,
        decay_steps=decay_steps,
        low_counts_threshold=0.0,
    )
    _, ys2, count_ys2 = one_sided_ema(
        -xolds[::-1],
        yolds[::-1],
        low=-float(high),
        high=-float(low),
        n=n,
        decay_steps=decay_steps,
        low_counts_threshold=0.0,
    )
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = np.divide(
        ys1 * count_ys1 + ys2 * count_ys2,
        count_ys,
        out=np.full_like(count_ys, np.nan),
        where=count_ys > 0,
    )
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys


def aggregate_seed_curves(seed_curves, resample_points=512, smooth_step=1.0, ci_type="std"):
    """baselines 风格：交集区间 + EMA 重采样 + 均值/方差聚合。"""
    prepared = []
    for timesteps, values in seed_curves:
        cleaned = _prepare_xy(timesteps, values)
        if cleaned is not None:
            prepared.append(cleaned)

    if not prepared:
        return None

    low = max(xvals[0] for xvals, _ in prepared)
    high = min(xvals[-1] for xvals, _ in prepared)
    if low >= high:
        return None

    resample_points = max(2, int(resample_points))
    smoothed_ys = []
    usex = None
    for xvals, yvals in prepared:
        xs, ys, _ = symmetric_ema(
            xvals,
            yvals,
            low=low,
            high=high,
            n=resample_points,
            decay_steps=smooth_step,
        )
        usex = xs
        smoothed_ys.append(ys)

    ys = np.asarray(smoothed_ys, dtype=np.float64)
    finite_mask = np.all(np.isfinite(ys), axis=0)
    if not np.any(finite_mask):
        return None

    x_common = usex[finite_mask]
    ys = ys[:, finite_mask]

    mean = np.mean(ys, axis=0)
    std = np.std(ys, axis=0)
    stderr = std / np.sqrt(max(1, ys.shape[0]))

    if ci_type == "sem":
        ci = stderr
    else:
        ci = std

    lower = mean - ci
    upper = mean + ci
    return x_common, mean, lower, upper, std, stderr, ys.shape[0]

def plot_curves(algorithms, seeds_to_plot=None, save_path="learning_curves.png",
                smooth_window=10, smooth_method="moving", smooth_alpha=0.6, smooth_beta=0.1,
                plot_cl=True, plot_non_cl=True, n_interpolate_points=512, smooth_step=1.0, ci_type="std"):
    """
    Plots learning curves for specified algorithms on the same figures.
    Reads CSV logs from 'results' directory.
    All algorithms are plotted together: one figure for reward, one for success rate.
    
    Args:
        algorithms: 算法列表
        seeds_to_plot: 要绘制的随机种子列表（如 ['1', '2', '3']），None 表示绘制所有种子
        save_path: 保存路径
        smooth_window: 兼容旧参数，baselines 风格聚合中不使用
        smooth_method: 兼容旧参数，baselines 风格聚合中不使用
        smooth_alpha: 兼容旧参数，baselines 风格聚合中不使用
        smooth_beta: 兼容旧参数，baselines 风格聚合中不使用
        plot_cl: 是否绘制带 CL- 前缀的算法
        plot_non_cl: 是否绘制不带 CL- 前缀的算法
        n_interpolate_points: baselines 风格重采样点数（默认 512）
        smooth_step: baselines 风格 EMA 的 decay_steps 参数
    """
    results_dir = "./results"
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    # 收集所有数据
    all_data = []
    
    for algo in algorithms:
        # 根据参数决定搜索哪些算法变体
        algo_variants = []
        if plot_non_cl:
            algo_variants.append(algo)
        if plot_cl:
            algo_variants.append(f"CL-{algo}")
        
        if not algo_variants:
            print(f"Warning: plot_cl and plot_non_cl are both False for {algo}, skipping...")
            continue
        
        csv_files = []
        for algo_variant in algo_variants:
            search_patterns = [
                os.path.join(results_dir, f"{algo_variant}*", f"{algo_variant}_seed*_log.csv"),
                os.path.join(results_dir, f"{algo_variant}*", "**", f"{algo_variant}_seed*_log.csv"),
                os.path.join(results_dir, f"{algo_variant}*", "*_log.csv"),
            ]
            for pattern in search_patterns:
                csv_files.extend(glob.glob(pattern, recursive=True))
        csv_files = sorted(set(csv_files))
        
        if not csv_files:
            print(f"No logs found for algorithm: {algo}")
            continue
            
        print(f"Found {len(csv_files)} logs for {algo}")
        
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # 从文件名中提取实际的算法名和种子
                file_basename = os.path.basename(file)
                if '_seed' in file_basename:
                    parts = file_basename.split('_seed')
                    actual_algo = _safe_to_internal_algo_name(parts[0])
                    # 提取种子号
                    seed_part = parts[1].split('_')[0] if len(parts) > 1 else 'unknown'
                else:
                    actual_algo = _safe_to_internal_algo_name(algo)
                    seed_part = 'unknown'
                
                # 如果指定了种子列表，则只加载指定的种子
                if seeds_to_plot is not None and seed_part not in seeds_to_plot:
                    print(f"  Skipped: {file_basename} (seed {seed_part} not in {seeds_to_plot})")
                    continue
                
                df['Algorithm'] = actual_algo
                df['AlgorithmLabel'] = _safe_to_plot_label(actual_algo)
                df['Seed'] = seed_part
                all_data.append(df)
                print(f"  Loaded: {file_basename} -> {actual_algo} (seed={seed_part})")
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    if not all_data:
        print("No data loaded. Exiting.")
        return
    
    # 合并所有数据
    full_df = pd.concat(all_data, ignore_index=True)
    
    # 获取所有唯一的算法名
    unique_algorithms = full_df['Algorithm'].unique()
    print(f"\nAlgorithms to plot: {list(unique_algorithms)}")
    
    # 为每个算法分配颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_algorithms)))
    color_map = {algo: colors[i] for i, algo in enumerate(unique_algorithms)}
    
    # 为不同算法分配不同的线型
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1))]
    line_style_map = {algo: line_styles[i % len(line_styles)] for i, algo in enumerate(unique_algorithms)}
    
    # === Plot Reward ===
    fig_reward, ax_reward = plt.subplots(figsize=(12, 8))
    
    for algo_name in unique_algorithms:
        algo_df = full_df[full_df['Algorithm'] == algo_name]
        if algo_df.empty:
            continue
        
        # 获取该算法的所有种子
        seeds = algo_df['Seed'].unique()
        print(f"\n{algo_name}: {len(seeds)} seeds")
        
        # 收集每个种子的数据
        seed_curves = []
        
        for seed in seeds:
            seed_df = algo_df[algo_df['Seed'] == seed].sort_values('total_timesteps')
            if len(seed_df) == 0:
                continue
            
            timesteps = seed_df['total_timesteps'].values
            rewards = seed_df['reward'].values

            seed_curves.append((timesteps, rewards))

        if not seed_curves:
            continue

        aggregated = aggregate_seed_curves(
            seed_curves,
            resample_points=n_interpolate_points,
            smooth_step=smooth_step,
            ci_type=ci_type,
        )
        if aggregated is None:
            print(f"  Warning: {algo_name} 各 seed 时间轴交集为空或有效点不足，跳过")
            continue
        x_common, mean_reward, lower, upper, _, _, n_seed_curves = aggregated
        
        color = color_map[algo_name]
        
        display_label = str(algo_df["AlgorithmLabel"].iloc[0])

        # 绘制均值曲线
        ax_reward.plot(x_common, mean_reward,
                      label=display_label, linewidth=2.5, color=color,
                      linestyle=line_style_map[algo_name])
        
        # baselines 风格：均值曲线 + 组内离散度阴影
        ax_reward.fill_between(x_common, lower, upper, color=color, alpha=0.2)
    
    ax_reward.set_xlabel("Total Timesteps", fontfamily='Arial', fontsize=20)
    ax_reward.set_ylabel("Reward", fontfamily='Arial', fontsize=20)
    ax_reward.set_title("Learning Curves - Reward Comparison", fontfamily='Arial', fontsize=20, fontweight='bold')
    ax_reward.legend(prop={'family': 'Arial', 'size': 20}, loc='best')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.tick_params(axis='both', labelsize=18)  # 设置坐标轴刻度文字大小
    
    plt.tight_layout()
    reward_save_path = os.path.join(results_dir, "combined_reward_curves.png")
    plt.savefig(reward_save_path, dpi=600, bbox_inches='tight')
    print(f"\nCombined reward plot saved to {reward_save_path}")
    
    # === Plot Success Rate ===
    fig_success, ax_success = plt.subplots(figsize=(12, 8))
    
    for algo_name in unique_algorithms:
        algo_df = full_df[full_df['Algorithm'] == algo_name]
        if algo_df.empty:
            continue
        
        # 获取该算法的所有种子
        seeds = algo_df['Seed'].unique()
        
        # 收集每个种子的数据
        seed_curves = []
        
        for seed in seeds:
            seed_df = algo_df[algo_df['Seed'] == seed].sort_values('total_timesteps')
            if len(seed_df) == 0:
                continue
            
            timesteps = seed_df['total_timesteps'].values
            success_rates = np.clip(seed_df['success_rate'].values, 0.0, 1.0)

            seed_curves.append((timesteps, success_rates))

        if not seed_curves:
            continue

        aggregated = aggregate_seed_curves(
            seed_curves,
            resample_points=n_interpolate_points,
            smooth_step=smooth_step,
            ci_type=ci_type,
        )
        if aggregated is None:
            continue
        x_common, mean_success, lower, upper, _, _, _ = aggregated
        mean_success = np.clip(mean_success, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
        lower = np.clip(lower, 0.0, 1.0)
        
        color = color_map[algo_name]
        
        display_label = str(algo_df["AlgorithmLabel"].iloc[0])

        # 绘制均值曲线
        ax_success.plot(x_common, mean_success,
                       label=display_label, linewidth=2.5, color=color,
                       linestyle=line_style_map[algo_name])
        
        # 绘制阴影区域（标准差或标准误差）
        ax_success.fill_between(x_common, lower, upper,
                               color=color, alpha=0.2)
    
    ax_success.set_xlabel("Total Timesteps", fontfamily='Arial', fontsize=20)
    ax_success.set_ylabel("Success Rate", fontfamily='Arial', fontsize=20)
    # ax_success.set_title("Learning Curves - Success Rate Comparison", fontfamily='Arial', fontsize=20, fontweight='bold')
    ax_success.set_ylim(0, 1.05)
    ax_success.legend(prop={'family': 'Arial', 'size': 20}, loc='best')
    ax_success.grid(True, alpha=0.3)
    ax_success.tick_params(axis='both', labelsize=18)  # 设置坐标轴刻度文字大小
    
    plt.tight_layout()
    success_save_path = os.path.join(results_dir, "combined_success_rate_curves.png")
    plt.savefig(success_save_path, dpi=600, bbox_inches='tight')
    print(f"Combined success rate plot saved to {success_save_path}")
    
    # Show both plots
    plt.show()
    plt.close('all')

def main():
    args = get_config()
    
    algo_list_input = args.algorithm_name

    expanded_algorithms = expand_algorithm_spec(algo_list_input)
    algos_to_plot = []
    for algorithm_name in expanded_algorithms:
        _, core_name = split_curriculum_prefix(algorithm_name)
        if core_name not in algos_to_plot:
            algos_to_plot.append(core_name)

    display_algos = [to_kebab_algorithm_name(name, upper=False) for name in algos_to_plot]
    print(f"Plotting curves for: {display_algos}")
    print(f"Plot CL algorithms: {args.plot_cl}")
    print(f"Plot non-CL algorithms: {args.plot_non_cl}")
    
    if not args.plot_cl and not args.plot_non_cl:
        print("Error: Both --plot_cl and --plot_non_cl are False. Nothing to plot.")
        return
    
    # 解析要绘制的种子
    seeds_to_plot = None
    if hasattr(args, 'seed') and args.seed:
        if isinstance(args.seed, str):
            seeds_to_plot = [s.strip() for s in args.seed.split(',')]
        elif isinstance(args.seed, (list, tuple)):
            seeds_to_plot = [str(s) for s in args.seed]
        else:
            seeds_to_plot = [str(args.seed)]
        print(f"Seeds to plot: {seeds_to_plot}")
    
    plot_curves(
        algos_to_plot,
        seeds_to_plot=seeds_to_plot,
        smooth_window=args.smooth_window,
        smooth_method=args.smooth_method,
        smooth_alpha=args.smooth_alpha,
        smooth_beta=args.smooth_beta,
        plot_cl=args.plot_cl,
        plot_non_cl=args.plot_non_cl,
        n_interpolate_points=args.resample_points,
        smooth_step=args.curve_smooth_step,
        ci_type=args.ci_type)

if __name__ == "__main__":
    main()
