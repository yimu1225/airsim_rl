import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import get_config

def plot_curves(algorithms, save_path="learning_curves.png", smooth_window=10):
    """
    Plots learning curves for specified algorithms on the same figures.
    Reads CSV logs from 'results' directory.
    All algorithms are plotted together: one figure for reward, one for success rate.
    """
    results_dir = "./results"
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    # Collect all data from all algorithms
    all_data = []
    
    for algo in algorithms:
        # Data gathering for this algorithm
        algo_data = []
        
        # Find all log files for this algorithm
        search_patterns = [
            os.path.join(results_dir, f"{algo}*", f"{algo}_log.csv"),
            os.path.join(results_dir, f"{algo}*", "**", f"{algo}_log.csv"),
            os.path.join(results_dir, f"{algo}*", "*_log.csv"),
        ]
        csv_files = []
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
                # Add algorithm name for grouping
                df['Algorithm'] = algo.upper()
                algo_data.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if not algo_data:
            print(f"No data loaded for {algo}. Skipping.")
            continue

        # Combine all dataframes for this algorithm
        algo_df = pd.concat(algo_data, ignore_index=True)
        
        # Apply smoothing to this algorithm's data
        if smooth_window > 1:
            algo_df['reward_smooth'] = algo_df['reward'].rolling(window=smooth_window, min_periods=1).mean()
            algo_df['success_rate_smooth'] = algo_df['success_rate'].rolling(window=smooth_window, min_periods=1).mean()
        else:
            algo_df['reward_smooth'] = algo_df['reward']
            algo_df['success_rate_smooth'] = algo_df['success_rate']
        
        all_data.append(algo_df)
    
    if not all_data:
        print("No data loaded for any algorithm. Exiting.")
        return
    
    # Combine all data from all algorithms
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Plot Reward - Single Figure for all algorithms
    fig_reward, ax_reward = plt.subplots(figsize=(12, 8))
    
    # Plot each algorithm as a separate line with mean values
    for algo in algorithms:
        algo_df = full_df[full_df['Algorithm'] == algo.upper()]
        if not algo_df.empty:
            # Calculate mean reward across all runs for this algorithm
            grouped = algo_df.groupby('total_timesteps', sort=True)['reward_smooth']
            mean_reward = grouped.mean()
            std_reward = grouped.std().fillna(0.0)
            
            # Plot mean line
            ax_reward.plot(mean_reward.index, mean_reward.values, 
                          label=algo.upper(), linewidth=2.5, alpha=0.8)
            
            # Add confidence interval (optional)
            ax_reward.fill_between(mean_reward.index, 
                                  mean_reward.values - std_reward.values,
                                  mean_reward.values + std_reward.values,
                                  alpha=0.2)
    
    ax_reward.set_xlabel("Total Timesteps", fontfamily='Arial', fontsize=14)
    ax_reward.set_ylabel("Reward", fontfamily='Arial', fontsize=14)
    ax_reward.set_title("Learning Curves - Reward Comparison", fontfamily='Arial', fontsize=16, fontweight='bold')
    ax_reward.legend(prop={'family': 'Arial', 'size': 12}, loc='best')
    ax_reward.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    reward_save_path = os.path.join(results_dir, "combined_reward_curves.png")
    plt.savefig(reward_save_path, dpi=600, bbox_inches='tight')
    print(f"Combined reward plot saved to {reward_save_path}")
    
    # Plot Success Rate - Single Figure for all algorithms
    fig_success, ax_success = plt.subplots(figsize=(12, 8))
    
    # Plot each algorithm as a separate line with mean values
    for algo in algorithms:
        algo_df = full_df[full_df['Algorithm'] == algo.upper()]
        if not algo_df.empty:
            # Calculate mean success rate across all runs for this algorithm
            grouped = algo_df.groupby('total_timesteps', sort=True)['success_rate_smooth']
            mean_success = grouped.mean()
            std_success = grouped.std().fillna(0.0)
            
            # Plot mean line
            ax_success.plot(mean_success.index, mean_success.values, 
                           label=algo.upper(), linewidth=2.5, alpha=0.8)
            
            # Add confidence interval (optional)
            ax_success.fill_between(mean_success.index, 
                                   mean_success.values - std_success.values,
                                   mean_success.values + std_success.values,
                                   alpha=0.2)
    
    ax_success.set_xlabel("Total Timesteps", fontfamily='Arial', fontsize=14)
    ax_success.set_ylabel("Success Rate", fontfamily='Arial', fontsize=14)
    ax_success.set_title("Learning Curves - Success Rate Comparison", fontfamily='Arial', fontsize=16, fontweight='bold')
    ax_success.set_ylim(0, 1.05)
    ax_success.legend(prop={'family': 'Arial', 'size': 12}, loc='best')
    ax_success.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    success_save_path = os.path.join(results_dir, "combined_success_rate_curves.png")
    plt.savefig(success_save_path, dpi=600, bbox_inches='tight')
    print(f"Combined success rate plot saved to {success_save_path}")
    
    # Show both plots in windows
    plt.show()
    
    # Close figures to free memory after showing
    plt.close('all')

def main():
    # Use config parser to get parameters
    args = get_config()
    
    algo_list_input = args.algorithm_name
    
    # Define groups same as config.py hint implies
    supported_algos = [
        'td3', 'aetd3', 'per_td3', 'per_aetd3',
        'gru_td3', 'lstm_td3', 'gru_aetd3', 'lstm_aetd3', 'cfc_td3',
        'vmamba_td3', 'vmamba_td3_no_cross', 'st_vmamba_td3', 'st_mamba_td3', 'ST-VimTD3', 'st_cnn_td3'
    ]
    
    if algo_list_input == 'all':
        algos_to_plot = supported_algos
    else:
        # Split by comma and strip whitespace
        algos_to_plot = [a.strip() for a in algo_list_input.split(',')]
        
    print(f"Plotting curves for: {algos_to_plot}")
    plot_curves(algos_to_plot, smooth_window=args.smooth_window)

if __name__ == "__main__":
    main()
