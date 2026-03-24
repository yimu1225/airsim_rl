import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import get_config

def smooth_curve(values, window=10):
    """对曲线进行平滑处理。

    原来的实现使用 ``np.convolve``+``mode='same'``，该模式会在
    序列两端假定零填充，因此边缘点会被错误地拉低，尤其在
    窗口较大时会产生不自然的下降或上升。
    
    这里改用 pandas 的滚动平均，``center=True`` 保证窗口
    居中，``min_periods=1`` 在边缘处仍然返回合理值。结果长度
   与输入保持一致，不会出现边缘失真。
    """
    if window <= 1 or len(values) < window:
        return values
    # 使用 pandas rolling 计算居中移动平均，自动处理边界
    series = pd.Series(values)
    smoothed = series.rolling(window, center=True, min_periods=1).mean().to_numpy()
    return smoothed


def zero_phase_double_exponential_smoothing(data, alpha=0.3, beta=0.1):
    """
    零相位双重指数平滑（Zero-Phase Double Exponential Smoothing）
    
    结合双重指数平滑（Holt's方法）和零相位滤波技术：
    1. 双重指数平滑：处理带有趋势的时间序列，同时平滑水平和趋势分量
    2. 零相位滤波：通过前向+后向滤波消除相位延迟
    
    参数:
        data: 输入数据（1D numpy数组或列表）
        alpha: 水平平滑因子 (0~1)，越大越关注近期数据
        beta: 趋势平滑因子 (0~1)，越大越关注近期趋势变化
    
    返回:
        平滑后的数据（与输入等长）
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n <= 1:
        return data
    
    # 第一阶段：正向双重指数平滑
    s_fwd = np.zeros(n)  # 水平分量
    b_fwd = np.zeros(n)  # 趋势分量
    
    # 初始化
    s_fwd[0] = data[0]
    b_fwd[0] = data[1] - data[0] if n > 1 else 0
    
    for t in range(1, n):
        s_fwd[t] = alpha * data[t] + (1 - alpha) * (s_fwd[t-1] + b_fwd[t-1])
        b_fwd[t] = beta * (s_fwd[t] - s_fwd[t-1]) + (1 - beta) * b_fwd[t-1]
    
    # 第二阶段：反向双重指数平滑（消除相位延迟）
    s_rev = s_fwd[::-1]
    s_bwd = np.zeros(n)
    b_bwd = np.zeros(n)
    
    s_bwd[0] = s_rev[0]
    b_bwd[0] = s_rev[1] - s_rev[0] if n > 1 else 0
    
    for t in range(1, n):
        s_bwd[t] = alpha * s_rev[t] + (1 - alpha) * (s_bwd[t-1] + b_bwd[t-1])
        b_bwd[t] = beta * (s_bwd[t] - s_bwd[t-1]) + (1 - beta) * b_bwd[t-1]
    
    return s_bwd[::-1]


def interpolate_to_grid(x_src, y_src, x_dst):
    """将单条曲线插值到统一网格，原始范围外填充 NaN。"""
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    x_dst = np.asarray(x_dst, dtype=np.float64)

    if x_src.size == 0 or y_src.size == 0:
        return np.full_like(x_dst, np.nan, dtype=np.float64)

    # 保证 x 严格递增且去重，避免 np.interp 在重复点处行为不稳定
    order = np.argsort(x_src)
    x_sorted = x_src[order]
    y_sorted = y_src[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_idx]

    if unique_x.size == 1:
        y_interp = np.full_like(x_dst, np.nan, dtype=np.float64)
        y_interp[np.isclose(x_dst, unique_x[0])] = unique_y[0]
        return y_interp

    return np.interp(x_dst, unique_x, unique_y, left=np.nan, right=np.nan)


def aggregate_seed_curves(smoothed_curves, n_points=1000, ci_type="std", min_valid_count=1):
    """
    将多个 seed 曲线对齐到“并集时间轴”，并用 NaN 感知统计量聚合。

    返回:
        x_common, mean, lower, upper, valid_mask, valid_counts
    """
    if not smoothed_curves:
        return None

    starts = [timesteps[0] for timesteps, _ in smoothed_curves if len(timesteps) > 0]
    ends = [timesteps[-1] for timesteps, _ in smoothed_curves if len(timesteps) > 0]
    if not starts or not ends:
        return None

    min_timestep = min(starts)
    max_timestep = max(ends)
    if min_timestep >= max_timestep:
        return None

    x_common = np.linspace(min_timestep, max_timestep, n_points)
    interpolated_curves = np.array(
        [interpolate_to_grid(timesteps, values, x_common) for timesteps, values in smoothed_curves],
        dtype=np.float64,
    )

    valid_counts = np.sum(np.isfinite(interpolated_curves), axis=0)
    valid_mask = valid_counts >= max(1, int(min_valid_count))
    if not np.any(valid_mask):
        return None

    mean = np.full_like(x_common, np.nan, dtype=np.float64)
    std = np.full_like(x_common, np.nan, dtype=np.float64)

    mean[valid_mask] = (
        np.nansum(interpolated_curves[:, valid_mask], axis=0) / valid_counts[valid_mask]
    )
    centered = interpolated_curves[:, valid_mask] - mean[valid_mask]
    centered = np.where(np.isfinite(interpolated_curves[:, valid_mask]), centered, np.nan)
    std[valid_mask] = np.sqrt(
        np.nansum(centered ** 2, axis=0) / valid_counts[valid_mask]
    )

    if ci_type == "std":
        ci = std
    else:  # sem
        ci = std / np.sqrt(np.maximum(valid_counts, 1))

    upper = mean + ci
    lower = mean - ci
    return x_common, mean, lower, upper, valid_mask, valid_counts

def plot_curves(algorithms, seeds_to_plot=None, save_path="learning_curves.png", 
                smooth_window=10, smooth_method="moving", smooth_alpha=0.6, smooth_beta=0.1,
                plot_cl=True, plot_non_cl=True, n_interpolate_points=1000, ci_type="std"):
    """
    Plots learning curves for specified algorithms on the same figures.
    Reads CSV logs from 'results' directory.
    All algorithms are plotted together: one figure for reward, one for success rate.
    
    Args:
        algorithms: 算法列表
        seeds_to_plot: 要绘制的随机种子列表（如 ['1', '2', '3']），None 表示绘制所有种子
        save_path: 保存路径
        smooth_window: 平滑窗口大小（仅对 ``moving`` 方法有效）
        smooth_method: 平滑方法，"moving" 或 "zero_phase_des"（零相位双重指数平滑）。
        smooth_alpha: 对于 ``zero_phase_des`` 使用，水平平滑因子 (0~1)，越大越关注近期数据
        smooth_beta: 对于 ``zero_phase_des`` 使用，趋势平滑因子 (0~1)，越大越关注近期趋势变化
        plot_cl: 是否绘制带 CL- 前缀的算法
        plot_non_cl: 是否绘制不带 CL- 前缀的算法
        n_interpolate_points: 插值点数，用于对齐多个种子的曲线（并集时间轴）
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
                    actual_algo = parts[0]
                    # 提取种子号
                    seed_part = parts[1].split('_')[0] if len(parts) > 1 else 'unknown'
                else:
                    actual_algo = algo
                    seed_part = 'unknown'
                
                # 如果指定了种子列表，则只加载指定的种子
                if seeds_to_plot is not None and seed_part not in seeds_to_plot:
                    print(f"  Skipped: {file_basename} (seed {seed_part} not in {seeds_to_plot})")
                    continue
                
                df['Algorithm'] = actual_algo
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
        smoothed_curves = []
        seed_end_timesteps = []
        
        for seed in seeds:
            seed_df = algo_df[algo_df['Seed'] == seed].sort_values('total_timesteps')
            if len(seed_df) == 0:
                continue
            
            timesteps = seed_df['total_timesteps'].values
            rewards = seed_df['reward'].values
            
            # 对该种子的数据进行平滑
            if smooth_method == "moving":
                rewards_smooth = smooth_curve(rewards, smooth_window)
            else:  # zero_phase_des（零相位双重指数平滑）
                rewards_smooth = zero_phase_double_exponential_smoothing(
                    rewards, alpha=smooth_alpha, beta=smooth_beta
                )
            
            smoothed_curves.append((timesteps, rewards_smooth))
            seed_end_timesteps.append(timesteps[-1])
        
        if not smoothed_curves:
            continue

        min_end = min(seed_end_timesteps)
        max_end = max(seed_end_timesteps)
        if max_end > min_end * 3:
            print(
                f"  Note: {algo_name} seed 时长差异较大 ({int(min_end)} ~ {int(max_end)}), "
                "已使用并集时间轴 + NaN 感知均值避免短 seed 截断整条曲线。"
            )

        n_seed_curves = len(smoothed_curves)
        min_valid_for_plot = 1 if n_seed_curves <= 1 else max(2, int(np.ceil(0.5 * n_seed_curves)))

        aggregated = aggregate_seed_curves(
            smoothed_curves,
            n_points=n_interpolate_points,
            ci_type=ci_type,
            min_valid_count=min_valid_for_plot,
        )
        if aggregated is None:
            print(f"  Warning: {algo_name} 聚合失败，跳过")
            continue
        x_common, mean_reward, lower, upper, valid_mask, valid_counts = aggregated
        if valid_counts[-1] < min_valid_for_plot:
            print(
                f"  Note: {algo_name} 末端可用 seed 数不足（{int(valid_counts[-1])}/{n_seed_curves}），"
                f"已隐藏覆盖不足的尾段以避免终点跳变。"
            )
        
        color = color_map[algo_name]
        
        # 绘制均值曲线
        ax_reward.plot(x_common[valid_mask], mean_reward[valid_mask], 
                      label=algo_name.upper(), linewidth=2.5, color=color, 
                      linestyle=line_style_map[algo_name])
        
        # 绘制阴影区域（标准差或标准误差）
        ax_reward.fill_between(x_common[valid_mask], lower[valid_mask], upper[valid_mask],
                              color=color, alpha=0.2)
    
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
        smoothed_curves = []
        seed_end_timesteps = []
        
        for seed in seeds:
            seed_df = algo_df[algo_df['Seed'] == seed].sort_values('total_timesteps')
            if len(seed_df) == 0:
                continue
            
            timesteps = seed_df['total_timesteps'].values
            success_rates = seed_df['success_rate'].values
            
            # 对该种子的数据进行平滑
            if smooth_method == "moving":
                success_smooth = smooth_curve(success_rates, smooth_window)
            else:  # zero_phase_des（零相位双重指数平滑）
                success_smooth = zero_phase_double_exponential_smoothing(
                    success_rates, alpha=smooth_alpha, beta=smooth_beta
                )
            # 成功率是有界指标，平滑后裁剪回 [0, 1] 避免 Holt 趋势项导致越界
            success_smooth = np.clip(success_smooth, 0.0, 1.0)
            
            smoothed_curves.append((timesteps, success_smooth))
            seed_end_timesteps.append(timesteps[-1])
        
        if not smoothed_curves:
            continue

        min_end = min(seed_end_timesteps)
        max_end = max(seed_end_timesteps)
        if max_end > min_end * 3:
            print(
                f"  Note: {algo_name} seed 时长差异较大 ({int(min_end)} ~ {int(max_end)}), "
                "已使用并集时间轴 + NaN 感知均值避免短 seed 截断整条曲线。"
            )

        n_seed_curves = len(smoothed_curves)
        min_valid_for_plot = 1 if n_seed_curves <= 1 else max(2, int(np.ceil(0.5 * n_seed_curves)))

        aggregated = aggregate_seed_curves(
            smoothed_curves,
            n_points=n_interpolate_points,
            ci_type=ci_type,
            min_valid_count=min_valid_for_plot,
        )
        if aggregated is None:
            continue
        x_common, mean_success, lower, upper, valid_mask, valid_counts = aggregated
        if valid_counts[-1] < min_valid_for_plot:
            print(
                f"  Note: {algo_name} 末端可用 seed 数不足（{int(valid_counts[-1])}/{n_seed_curves}），"
                f"已隐藏覆盖不足的尾段以避免终点跳变。"
            )
        mean_success = np.clip(mean_success, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
        lower = np.clip(lower, 0.0, 1.0)
        
        color = color_map[algo_name]
        
        # 绘制均值曲线
        ax_success.plot(x_common[valid_mask], mean_success[valid_mask], 
                       label=algo_name.upper(), linewidth=2.5, color=color,
                       linestyle=line_style_map[algo_name])
        
        # 绘制阴影区域（标准差或标准误差）
        ax_success.fill_between(x_common[valid_mask], lower[valid_mask], upper[valid_mask],
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
    
    supported_algos = [
        'td3', 'ddpg', 'aetd3', 'per_td3', 'per_aetd3',
        'gru_td3', 'lstm_td3', 'gru_aetd3', 'lstm_aetd3', 'cfc_td3',
        'st_mamba_td3', 'ST-VimTD3', 'ST-SVimTD3', 'st_cnn_td3', 'gam_mamba_td3'
    ]
    
    if algo_list_input == 'all':
        algos_to_plot = supported_algos
    else:
        algos_to_plot = [a.strip() for a in algo_list_input.split(',')]
        
    print(f"Plotting curves for: {algos_to_plot}")
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
        ci_type=args.ci_type)

if __name__ == "__main__":
    main()
