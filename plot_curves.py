import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
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

def plot_curves(algorithms, seeds_to_plot=None, save_path="learning_curves.png", 
                smooth_window=10, smooth_method="moving", smooth_alpha=0.6,
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
        smooth_method: 平滑方法，"moving" 或 "ema"。
        smooth_alpha: 对于 ``ema`` 使用，遵循 TensorBoard 语义——
            越大越平滑（0.0 无平滑，1.0 完全保留过去）。
            内部会转换成 pandas 所需的 ``alpha = 1 - smooth_alpha``。
        plot_cl: 是否绘制带 CL- 前缀的算法
        plot_non_cl: 是否绘制不带 CL- 前缀的算法
        n_interpolate_points: 插值点数，用于对齐多个种子的曲线
    """
    from scipy import interpolate
    
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
    
    # === Plot Reward ===
    fig_reward, ax_reward = plt.subplots(figsize=(12, 8))
    
    for algo_name in unique_algorithms:
        algo_df = full_df[full_df['Algorithm'] == algo_name]
        if algo_df.empty:
            continue
        
        # 获取该算法的所有种子
        seeds = algo_df['Seed'].unique()
        print(f"\n{algo_name}: {len(seeds)} seeds")
        
        # 收集每个种子的数据，找到共同的 timestep 范围
        all_timesteps = []
        smoothed_curves = []
        
        for seed in seeds:
            seed_df = algo_df[algo_df['Seed'] == seed].sort_values('total_timesteps')
            if len(seed_df) == 0:
                continue
            
            timesteps = seed_df['total_timesteps'].values
            rewards = seed_df['reward'].values
            
            # 对该种子的数据进行平滑
            if smooth_method == "moving":
                rewards_smooth = smooth_curve(rewards, smooth_window)
            else:  # ema
                tb_val = smooth_alpha
                alpha = 1.0 - tb_val
                series = pd.Series(rewards)
                rewards_smooth = series.ewm(alpha=alpha, adjust=False).mean().to_numpy()
            
            all_timesteps.append(timesteps)
            smoothed_curves.append((timesteps, rewards_smooth))
        
        if not smoothed_curves:
            continue
        
        # 找到所有种子的共同 timestep 范围
        min_timestep = max(t[0] for t in all_timesteps)  # 最大的起始点
        max_timestep = min(t[-1] for t in all_timesteps)  # 最小的结束点
        
        if min_timestep >= max_timestep:
            print(f"  Warning: {algo_name} 的种子时间范围不重叠，跳过")
            continue
        
        # 创建统一的 timestep 网格
        x_common = np.linspace(min_timestep, max_timestep, n_interpolate_points)
        
        # 将每个种子的曲线插值到统一网格上
        interpolated_curves = []
        for timesteps, rewards in smoothed_curves:
            # 使用线性插值
            f = interpolate.interp1d(timesteps, rewards, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
            y_interp = f(x_common)
            interpolated_curves.append(y_interp)
        
        interpolated_curves = np.array(interpolated_curves)  # shape: (n_seeds, n_points)
        
        # 计算均值和标准差（跨种子）
        mean_reward = np.mean(interpolated_curves, axis=0)
        std_reward = np.std(interpolated_curves, axis=0)
        n_seeds = len(interpolated_curves)
        
        # 使用标准差或标准误差
        if ci_type == "std":
            ci = std_reward
        else:  # sem
            ci = std_reward / np.sqrt(n_seeds)
        upper = mean_reward + ci
        lower = mean_reward - ci
        
        color = color_map[algo_name]
        
        # 绘制均值曲线
        ax_reward.plot(x_common, mean_reward, 
                      label=algo_name, linewidth=2.5, color=color)
        
        # 绘制阴影区域（标准差或标准误差）
        ax_reward.fill_between(x_common, lower, upper,
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
        all_timesteps = []
        smoothed_curves = []
        
        for seed in seeds:
            seed_df = algo_df[algo_df['Seed'] == seed].sort_values('total_timesteps')
            if len(seed_df) == 0:
                continue
            
            timesteps = seed_df['total_timesteps'].values
            success_rates = seed_df['success_rate'].values
            
            # 对该种子的数据进行平滑
            if smooth_method == "moving":
                success_smooth = smooth_curve(success_rates, smooth_window)
            else:  # ema
                tb_val = smooth_alpha
                alpha = 1.0 - tb_val
                series = pd.Series(success_rates)
                success_smooth = series.ewm(alpha=alpha, adjust=False).mean().to_numpy()
            
            all_timesteps.append(timesteps)
            smoothed_curves.append((timesteps, success_smooth))
        
        if not smoothed_curves:
            continue
        
        # 找到所有种子的共同 timestep 范围
        min_timestep = max(t[0] for t in all_timesteps)
        max_timestep = min(t[-1] for t in all_timesteps)
        
        if min_timestep >= max_timestep:
            continue
        
        # 创建统一的 timestep 网格
        x_common = np.linspace(min_timestep, max_timestep, n_interpolate_points)
        
        # 将每个种子的曲线插值到统一网格上
        interpolated_curves = []
        for timesteps, success in smoothed_curves:
            f = interpolate.interp1d(timesteps, success, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            y_interp = f(x_common)
            interpolated_curves.append(y_interp)
        
        interpolated_curves = np.array(interpolated_curves)
        
        # 计算均值和标准差（跨种子）
        mean_success = np.mean(interpolated_curves, axis=0)
        std_success = np.std(interpolated_curves, axis=0)
        n_seeds = len(interpolated_curves)
        
        # 使用标准差或标准误差
        if ci_type == "std":
            ci = std_success
        else:  # sem
            ci = std_success / np.sqrt(n_seeds)
        upper = np.clip(mean_success + ci, 0, 1)
        lower = np.clip(mean_success - ci, 0, 1)
        
        color = color_map[algo_name]
        
        # 绘制均值曲线
        ax_success.plot(x_common, mean_success, 
                       label=algo_name, linewidth=2.5, color=color)
        
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
    
    supported_algos = [
        'td3', 'ddpg', 'aetd3', 'per_td3', 'per_aetd3',
        'gru_td3', 'lstm_td3', 'gru_aetd3', 'lstm_aetd3', 'cfc_td3',
        'st_mamba_td3', 'ST-VimTD3', 'ST-VimTD3-Safety', 'st_cnn_td3', 'gam_mamba_td3'
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
        print(f"Seeds to plot: {seeds_to_plot}")
    
    plot_curves(
        algos_to_plot,
        seeds_to_plot=seeds_to_plot,
        smooth_window=args.smooth_window,
        smooth_method=args.smooth_method,
        smooth_alpha=args.smooth_alpha,
        plot_cl=args.plot_cl,
        plot_non_cl=args.plot_non_cl,
        ci_type=args.ci_type)

if __name__ == "__main__":
    main()
