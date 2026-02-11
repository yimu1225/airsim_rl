#!/usr/bin/env python3
"""
衰减OU噪声可视化脚本

此脚本专门演示Ornstein-Uhlenbeck (OU) 噪声的衰减特性，模拟1万个step。
衰减OU噪声在强化学习中常用于逐渐减少探索，使智能体在训练后期更趋于确定性策略。
"""

import os
# 设置OpenMP环境变量以避免运行时冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithm.ou_noise import OUNoise


def generate_decay_noise_sequence(ou_noise, num_steps, decay_type='linear'):
    """生成衰减OU噪声序列"""
    sequence = []
    sigma_values = []
    
    for step in range(num_steps):
        # 计算当前step的衰减比例
        if decay_type == 'linear':
            decay_ratio = step / num_steps
        elif decay_type == 'exponential':
            decay_ratio = 1 - np.exp(-2 * step / num_steps)
        elif decay_type == 'cosine':
            decay_ratio = 0.5 * (1 + np.cos(np.pi * (1 - step / num_steps)))
        elif decay_type == 'sigmoid':
            # Sigmoid-like decay: 前期慢，中期快，后期平缓
            # 使用调整的sigmoid函数
            x = 10 * (step / num_steps - 0.6)  # 调整sigmoid的中心和陡度
            sigmoid_decay = 1 / (1 + np.exp(-x))
            decay_ratio = sigmoid_decay
        else:
            decay_ratio = step / num_steps
            
        # 直接计算sigma值：sigma从初始值衰减到最小值
        initial_sigma = ou_noise.initial_sigma
        sigma_min = ou_noise.sigma_min if ou_noise.sigma_min is not None else 0.01
        current_sigma = initial_sigma - (initial_sigma - sigma_min) * decay_ratio
        ou_noise.sigma = current_sigma
        
        # 记录当前sigma值和噪声样本
        sigma_values.append(ou_noise.sigma)
        noise = ou_noise.sample()
        sequence.append(noise[0])  # 假设是1维噪声，取第一个元素
    
    return np.array(sequence), np.array(sigma_values)


def plot_decay_comparison():
    """绘制不同衰减策略的对比"""
    num_steps = 10000
    time_steps = np.arange(num_steps)
    
    # 衰减策略配置
    decay_strategies = [
        {'type': 'sigmoid', 'label': 'Sigmoid衰减', 'color': 'blue'},
        {'type': 'exponential', 'label': '指数衰减', 'color': 'red'},
        {'type': 'cosine', 'label': '余弦衰减', 'color': 'green'},
        {'type': 'none', 'label': '无衰减', 'color': 'gray'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, strategy in enumerate(decay_strategies):
        # 创建OU噪声实例
        ou = OUNoise(
            size=1, 
            mu=0.0, 
            theta=0.15, 
            sigma=0.2,  # 初始较大的sigma以便观察衰减效果
            sigma_min=0.01,  # 最小sigma
            dt=1.0
        )
        
        if strategy['type'] == 'none':
            # 无衰减情况
            sequence = []
            sigma_values = []
            for step in range(num_steps):
                sigma_values.append(ou.sigma)
                noise = ou.sample()
                sequence.append(noise[0])
            sequence = np.array(sequence)
            sigma_values = np.array(sigma_values)
        else:
            sequence, sigma_values = generate_decay_noise_sequence(
                ou, num_steps, strategy['type']
            )
        
        # 绘制噪声序列
        ax = axes[0, 0]
        ax.plot(time_steps, sequence, 
                color=strategy['color'], 
                alpha=0.7, 
                linewidth=0.8,
                label=strategy['label'])
        
        # 绘制sigma衰减曲线
        ax = axes[0, 1]
        ax.plot(time_steps, sigma_values, 
                color=strategy['color'], 
                linewidth=2,
                label=strategy['label'])
        
        # 绘制噪声分布直方图（前1000步 vs 后1000步）
        ax = axes[1, 0]
        if strategy['type'] == 'none':
            ax.hist(sequence[:1000], bins=30, alpha=0.5, 
                   color=strategy['color'], density=True,
                   label=f'{strategy["label"]} (前1000步)')
            ax.hist(sequence[-1000:], bins=30, alpha=0.5, 
                   color=strategy['color'], density=True, linestyle='--',
                   label=f'{strategy["label"]} (后1000步)')
        else:
            ax.hist(sequence[:1000], bins=30, alpha=0.5, 
                   color=strategy['color'], density=True,
                   label=f'{strategy["label"]} (前1000步)')
            ax.hist(sequence[-1000:], bins=30, alpha=0.5, 
                   color=strategy['color'], density=True, linestyle='--',
                   label=f'{strategy["label"]} (后1000步)')
        
        # 绘制移动平均和移动标准差
        ax = axes[1, 1]
        window_size = 100
        if len(sequence) >= window_size:
            moving_avg = np.convolve(sequence, np.ones(window_size)/window_size, mode='valid')
            moving_std = []
            for i in range(len(moving_avg)):
                start_idx = i
                end_idx = i + window_size
                moving_std.append(np.std(sequence[start_idx:end_idx]))
            moving_std = np.array(moving_std)
            
            moving_time = time_steps[window_size-1:]
            ax.plot(moving_time, moving_avg, 
                   color=strategy['color'], linewidth=2,
                   label=f'{strategy["label"]} 移动平均')
            ax.fill_between(moving_time, 
                           moving_avg - moving_std, 
                           moving_avg + moving_std,
                           color=strategy['color'], alpha=0.2)
    
    # 设置各个子图的标题和标签
    axes[0, 0].set_title('不同衰减策略的OU噪声序列 (10000步)', fontsize=14)
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('噪声值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Sigma衰减曲线', fontsize=14)
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('Sigma值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('噪声分布对比 (前1000步 vs 后1000步)', fontsize=14)
    axes[1, 0].set_xlabel('噪声值')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('移动平均和移动标准差 (窗口=100)', fontsize=14)
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('噪声值')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decay_ou_noise_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_decay_analysis():
    """详细分析衰减OU噪声的特性"""
    num_steps = 10000
    time_steps = np.arange(num_steps)
    
    # 使用sigmoid衰减进行详细分析
    ou = OUNoise(
        size=1, 
        mu=0.0, 
        theta=0.15, 
        sigma=0.3,
        sigma_min=0.01,
        dt=1.0
    )
    
    sequence, sigma_values = generate_decay_noise_sequence(ou, num_steps, 'sigmoid')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 完整噪声序列
    ax = axes[0, 0]
    ax.plot(time_steps, sequence, 'b-', linewidth=0.8, alpha=0.7)
    ax.set_title('完整噪声序列 (10000步)', fontsize=12)
    ax.set_xlabel('时间步')
    ax.set_ylabel('噪声值')
    ax.grid(True, alpha=0.3)
    
    # 添加分段高亮
    segments = [(0, 1000, '初期'), (1000, 5000, '中期'), (5000, 10000, '后期')]
    colors = ['red', 'orange', 'green']
    for (start, end, label), color in zip(segments, colors):
        ax.axvspan(start, end, alpha=0.2, color=color, label=label)
    ax.legend()
    
    # 2. Sigma衰减过程
    ax = axes[0, 1]
    ax.plot(time_steps, sigma_values, 'r-', linewidth=2)
    ax.set_title('Sigma衰减过程', fontsize=12)
    ax.set_xlabel('时间步')
    ax.set_ylabel('Sigma值')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.01, color='g', linestyle='--', alpha=0.7, label='最小Sigma')
    ax.legend()
    
    # 3. 不同阶段的噪声分布
    ax = axes[0, 2]
    for (start, end, label), color in zip(segments, colors):
        segment_data = sequence[start:end]
        ax.hist(segment_data, bins=50, alpha=0.6, color=color, 
                density=True, label=f'{label} (均值={np.mean(segment_data):.3f})')
    ax.set_title('不同阶段的噪声分布', fontsize=12)
    ax.set_xlabel('噪声值')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 滑动窗口统计量
    ax = axes[1, 0]
    window_sizes = [100, 500, 1000]
    for window_size, color in zip(window_sizes, ['blue', 'red', 'green']):
        moving_std = []
        for i in range(len(sequence) - window_size + 1):
            window_data = sequence[i:i+window_size]
            moving_std.append(np.std(window_data))
        moving_std = np.array(moving_std)
        moving_time = time_steps[window_size-1:]
        ax.plot(moving_time, moving_std, color=color, linewidth=2, 
                label=f'窗口大小={window_size}')
    
    ax.set_title('滑动窗口标准差', fontsize=12)
    ax.set_xlabel('时间步')
    ax.set_ylabel('标准差')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 自相关函数随时间的变化
    ax = axes[1, 1]
    max_lag = 50
    autocorr_evolution = []
    
    # 在不同时间点计算自相关
    sample_points = [0, 2500, 5000, 7500, 9999]
    colors_corr = ['blue', 'green', 'orange', 'red', 'purple']
    
    for sample_point, color in zip(sample_points, colors_corr):
        if sample_point + max_lag < len(sequence):
            segment = sequence[sample_point:sample_point+max_lag+1]
            autocorr = np.correlate(segment - np.mean(segment), 
                                  segment - np.mean(segment), mode='full')
            autocorr = autocorr[autocorr.size // 2:]  # 只取正延迟部分
            autocorr = autocorr / autocorr[0]  # 归一化
            
            lags = np.arange(len(autocorr))
            ax.plot(lags, autocorr, color=color, linewidth=2, 
                   label=f'时间点={sample_point}')
    
    ax.set_title('不同时间点的自相关函数', fontsize=12)
    ax.set_xlabel('延迟 (时间步)')
    ax.set_ylabel('自相关系数')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 功率谱密度
    ax = axes[1, 2]
    from scipy import signal
    
    # 对不同阶段计算功率谱密度
    for (start, end, label), color in zip(segments, colors):
        segment_data = sequence[start:end]
        freqs, psd = signal.welch(segment_data, fs=1.0, nperseg=min(256, len(segment_data)))
        ax.semilogy(freqs, psd, color=color, linewidth=2, label=label)
    
    ax.set_title('不同阶段的功率谱密度', fontsize=12)
    ax.set_xlabel('频率')
    ax.set_ylabel('功率谱密度')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decay_ou_noise_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("衰减OU噪声可视化脚本")
    print("=" * 50)
    print("模拟步数: 10000")
    print("=" * 50)
    
    # 设置中文字体（如果需要的话）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        print("1. 生成不同衰减策略的对比图...")
        plot_decay_comparison()
        
        print("2. 生成详细的衰减OU噪声分析图...")
        plot_decay_analysis()
        
        print("\n可视化完成！生成的图片文件：")
        print("- decay_ou_noise_comparison.png: 衰减策略对比")
        print("- decay_ou_noise_analysis.png: 详细分析")
        print("\n分析要点：")
        print("- 线性衰减：sigma随时间线性减少")
        print("- 指数衰减：sigma按指数规律快速衰减")
        print("- 余弦衰减：sigma按余弦函数平滑衰减")
        print("- 无衰减：sigma保持恒定作为对照")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装必要的依赖包: numpy, matplotlib, scipy")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
