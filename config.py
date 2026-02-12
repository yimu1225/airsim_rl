import argparse
import numpy as np
import os

# Set CUDA memory allocator configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



def _parse_seed_value(seed_value):
    if isinstance(seed_value, int):
        return seed_value
    if seed_value is None:
        return seed_value
    seed_str = str(seed_value).strip()
    if seed_str == "":
        return seed_value
    if "," in seed_str:
        seeds = [s.strip() for s in seed_str.split(",") if s.strip() != ""]
        return [int(s) for s in seeds]
    return int(seed_str)


def get_config(argv=None):
    # get the parameters
    parser = argparse.ArgumentParser(description='AirSim_RL')
    # 环境 (Environment)
    parser.add_argument("--env_name", type=str, default='AirSimEnv-v42', help="要训练的环境名称")  # AirSimEnv-v42  CartPole-v0

    # 算法选择 (Algorithm Selection)
    parser.add_argument("--algorithm_name", type=str, default='td3',
                        help="要训练的算法。支持: td3, aetd3, per_td3, per_aetd3, gru_td3, lstm_td3, gru_aetd3, lstm_aetd3, cfc_td3, vmamba_td3, vmamba_td3_no_cross, st_vmamba_td3, st_mamba_td3, ST-VimTD3, st_cnn_td3。可以是单个，多个（逗号分隔），或组名 ('all', 'base', 'seq')")
    parser.add_argument("--smooth_window", type=int, default=1000, help="平滑窗口大小，用于平滑学习曲线")

    # 训练设置 (Training Setup)
    parser.add_argument("--seed", type=str, default="25", help="随机种子 (支持逗号分隔多个种子)")
    parser.add_argument("--cuda", action='store_false', default=True, help="是否使用CUDA")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="CUDA是否确定性")
    parser.add_argument("--n_training_threads", type=int, default=1, help="训练线程数")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="Rollout线程数（在AirSim环境中必须为1）")
    parser.add_argument("--max_timesteps", type=int, default=200000, help='要训练的环境步数 (默认: 10e6)')
    parser.add_argument("--buffer_size", type=int, default=20000, help='经验池大小 (注意内存占用: 30000步约占用4GB)')
    parser.add_argument("--learning_starts", type=int, default=000, help="训练开始前的时间步数 (兼容 start_timesteps)")
    parser.add_argument("--gradient_steps", type=int, default=1, help="每次更新的梯度步数")
    parser.add_argument("--episode_length", type=int, default=200, help='每个环境中的最大回合长度')
    parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率")
    parser.add_argument("--feature_dim", type=int, default=128, help="特征维度")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--exploration_noise", type=float, default=0.3, help="探索噪声")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子") 
    parser.add_argument("--tau", type=float, default=0.005, help="软更新参数")
    parser.add_argument("--actor_lr", type=float, default=4e-4, help="Actor学习率")
    parser.add_argument("--critic_lr", type=float, default=4e-4, help="Critic学习率")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="策略噪声")
    parser.add_argument("--noise_clip", type=float, default=0.2, help="噪声裁剪")
    parser.add_argument("--policy_freq", type=int, default=10, help="策略更新频率")
    parser.add_argument("--grad_clip", type=float, default=2.0, help="梯度裁剪")

    # 可视化 (Visualization)
    parser.add_argument("--render_window", action='store_true', default=False, help="显示实时可视化窗口 (默认开启，可用 --no-render_window 关闭)")
    parser.add_argument("--need_render", action='store_true', default=False, help="启用2D轨迹渲染窗口 (Gym rendering，用于debug)")
    
    
        
    # 循环网络参数 (LSTM/GRU)
    parser.add_argument("--stack_frames", type=int, default=4, help="非时序算法的图像堆叠帧数")
    parser.add_argument("--stack_frames_recurrent", type=int, default=1, help="时序算法的图像堆叠帧数 (通常为1，使用外部历史队列)")
    
    parser.add_argument("--seq_len", type=int, default=4, help="循环网络的序列长度，即输入到时序算法的连续帧数 (默认: 16)")

    # TD3 的 OU 噪声 (OU Noise for TD3)
    parser.add_argument("--ou_theta", type=float, default=0.15, help="OU噪声的theta参数")
    parser.add_argument("--ou_sigma", type=float, default=0.1, help="OU噪声的sigma参数")
    parser.add_argument("--ou_sigma_min", type=float, default=0.01, help="OU噪声的最小sigma")
    parser.add_argument("--ou_dt", type=float, default=1.0, help="OU噪声的时间步长")

    # GRU-TD3
    parser.add_argument("--gru_hidden_dim", type=int, default=128, help="GRU 的隐藏层维度")
    parser.add_argument("--gru_num_layers", type=int, default=1, help="GRU 层数")

    # LSTM-TD3
    parser.add_argument("--lstm_hidden_dim", type=int, default=128, help="LSTM 的隐藏层维度")
    # CfC
    parser.add_argument("--cfc_lr", type=float, default=1e-2, help="CfC 时间序列模块学习率")
    parser.add_argument("--cfc_units", type=int, default=32, help="NCPs 拓扑总神经元数")
    parser.add_argument("--cfc_motor_units", type=int, default=8, help="NCPs 拓扑输出 (运动) 神经元数")

    # VMamba-TD3 参数
    parser.add_argument("--vmamba_patch_size", type=int, default=2, help="VMamba 图像切分的 patch 大小")
    parser.add_argument("--vmamba_hidden_dim", type=int, default=64, help="VMamba 基础隐藏层维度")
    parser.add_argument("--vmamba_num_vss_blocks", type=int, nargs='+', default=[2, 2, 5, 2], help="VMamba 每个阶段的 VSSBlock 数量，列表长度决定 Stage 数量")
    parser.add_argument("--vmamba_drop_path_rate", type=float, default=0.1, help="VMamba DropPath 比率")
    parser.add_argument("--vmamba_layer_scale_init", type=float, default=1e-6, help="VMamba LayerScale 初始化值")
    parser.add_argument("--vmamba_ssm_d_state", type=int, default=32, help="VMamba SSM 状态维度")
    parser.add_argument("--vmamba_ssm_ratio", type=float, default=2.0, help="VMamba SSM 比率")
    parser.add_argument("--vmamba_mlp_ratio", type=float, default=4.0, help="VMamba MLP 比率")
    parser.add_argument("--vmamba_num_heads", type=int, default=4, help="VMamba CrossAttention 头数")
    parser.add_argument("--state_feature_dim", type=int, default=32, help="状态特征维度")
    
    # 时序Mamba参数 (Temporal Mamba Parameters)
    parser.add_argument("--mamba_d_state", type=int, default=16, help="时序Mamba SSM状态维度")
    parser.add_argument("--mamba_d_conv", type=int, default=4, help="时序Mamba卷积核大小")
    parser.add_argument("--mamba_expand", type=int, default=2, help="时序Mamba扩展因子")
    parser.add_argument("--attention_dropout", type=float, default=0.0, help="自注意力dropout率")

    # ST-Mamba 参数
    parser.add_argument("--st_mamba_embed_dim", type=int, default=192, help="ST-Mamba 嵌入维度")
    parser.add_argument("--st_mamba_depth", type=int, default=6, help="ST-Mamba Block 数量")
    parser.add_argument("--st_mamba_patch_size", type=int, default=16, help="ST-Mamba Patch 大小")
    parser.add_argument("--st_mamba_d_state", type=int, default=16, help="ST-Mamba SSM 状态维度")
    parser.add_argument("--st_mamba_d_conv", type=int, default=4, help="ST-Mamba SSM 卷积宽度")
    parser.add_argument("--st_mamba_expand", type=int, default=2, help="ST-Mamba Block 扩展因子")
    parser.add_argument("--st_mamba_drop_rate", type=float, default=0.1, help="ST-Mamba Dropout 率 (pos_drop)")
    parser.add_argument("--st_mamba_drop_path_rate", type=float, default=0.1, help="ST-Mamba Drop Path 率 (stochastic depth)")
    parser.add_argument("--st_mamba_temporal_depth", type=int, default=3, help="ST-Mamba-VimTokens 时序 Mamba Block 数量")


    # Adaptive Ensemble TD3
    parser.add_argument("--adaptive_k", type=int, default=5, help="Ensemble critics 的数量")
    parser.add_argument("--adaptive_reg", type=float, default=0.001, help="Adaptive ensemble 的正则化系数")
    parser.add_argument("--adaptive_meta_lr", type=float, default=1e-3, help="元网络 (Meta network) 学习率")
    # 连续控制参数 (Continuous Control Parameters)
    parser.add_argument("--min_forward_speed", type=float, default=0.0, help="最小前进速度 (m/s)")
    parser.add_argument("--max_forward_speed", type=float, default=2.0, help="最大前进速度 (m/s)")
    parser.add_argument("--max_vertical_speed", type=float, default=0.5, help="最大垂直速度 (m/s)")
    parser.add_argument("--max_yaw_rate", type=float, default=np.pi/12, help="最大偏航角速度 (rad/s)")
    parser.add_argument("--takeoff_height", type=float, default=-2.0, help="起飞目标高度 (NED坐标系中负值为向上)")
    parser.add_argument("--action_duration", type=float, default=0.5, help="在时钟缩放之前的每个速度指令的基础持续时间 (秒)")
    parser.add_argument("--clock_speed_factor", type=float, default=1.0, help="AirSim 设置中配置的 ClockSpeed 因子；持续时间将除以此值")

    # 飞行高度限制 (Flight Altitude Limits)
    parser.add_argument("--max_flight_altitude", type=float, default=4.0, help="最大飞行高度 (米, 正值为向上)")
    parser.add_argument("--min_flight_altitude", type=float, default=0.0, help="最小飞行高度 (米, 正值为向上)")
    parser.add_argument("--min_altitude_penalty", type=float, default=0.5, help="低于最小高度的惩罚")
    parser.add_argument("--max_altitude_penalty", type=float, default=3.5, help="高于最大高度的惩罚")
    parser.add_argument("--altitude_penalty_value", type=float, default=0.5, help="飞出高度惩罚范围时的固定惩罚值")

    # 保存 (Saving)
    parser.add_argument("--save_interval", type=int, default=20, help="检查点保存频率 (单位: epoch)")
    parser.add_argument("--continue_last", default=False, help="是否继续上次的训练")
    
    # 训练循环参数 (Training Loop Parameters)
    parser.add_argument("--steps_per_update", type=int, default=100, help='每次更新前收集的步数')
    parser.add_argument("--step_penalty", type=float, default=0.05, help="每步惩罚，以鼓励更快完成")

    # 日志 (Logging)
    parser.add_argument("--log_interval", type=int, default=1, help="日志记录间隔")
    parser.add_argument("--steps_per_epoch", type=int, default=50, help="每个 worker 每个 epoch 的步数 (Tune 迭代)")

    # 评估 (Evaluation)
    parser.add_argument("--eval", action='store_true', default=False, help="是否进行评估")
    parser.add_argument("--save_gifs", action='store_true', default=False, help="是否保存GIF")
    parser.add_argument("--ifi", type=float, default=0.333333, help="帧间间隔")
    parser.add_argument("--eval_episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--model_dir", type=str, default='results/AirSimEnv-v42/ppo-lstm3/run5/models', help="模型目录")

    

    
    # AIRSIM 连接参数 (AIRSIM CONNECTION PARAMETERS) 
    parser.add_argument("--airsim_ip", type=str, default="172.20.176.1", help="AirSim 服务器 IP 地址")
    parser.add_argument("--airsim_port", type=int, default=41451, help="AirSim 服务器端口")
    parser.add_argument("--settings_file", type=str, default="", help="AirSim settings.json 文件路径 (可选)")
    parser.add_argument("--load_model", type=str, default="", help="要加载的模型路径")



    # =============================================================================
    # 已注释的 PPO 算法参数 (COMMENTED PPO ALGORITHM PARAMETERS)
    # =============================================================================
    
    # parser.add_argument("--hidden_size", type=int, default=512, help="PPO 网络中隐藏层的维度大小 (默认: 512)")
    # parser.add_argument("--ppo_epoch", type=int, default=8, help='PPO 迭代次数 (默认: 4)')
    # parser.add_argument("--use_clipped_value_loss", action='store_false', default=True)
    # parser.add_argument("--clip_param", type=float, default=0.15, help='PPO 裁剪参数 (默认: 0.2)')
    # parser.add_argument("--num_mini_batch", type=int, default=1, help='PPO 的批次数量 (默认: 32)')
    # parser.add_argument("--entropy_coef", type=float, default=0.01, help='熵项系数 (默认: 0.01)')
    # parser.add_argument("--value_loss_coef", type=float, default=1.0, help='价值损失系数 (默认: 0.5)')
    # parser.add_argument("--lr", type=float, default=5e-4, help='学习率 (默认: 7e-4)')
    # parser.add_argument("--eps", type=float, default=1e-5, help='RMSprop 优化器 epsilon (默认: 1e-5)')
    # parser.add_argument("--max-grad-norm", type=float, default=5, help='梯度最大范数 (默认: 0.5)')
    # parser.add_argument("--use-gae", action='store_false', default=True, help='使用广义优势估计 (GAE)')
    # parser.add_argument("--gamma", type=float, default=0.99, help='奖励折扣因子 (默认: 0.99)')
    # parser.add_argument("--gae-lambda", type=float, default=0.95, help='GAE lambda 参数 (默认: 0.95)')
    # parser.add_argument("--use-proper-time-limits", action='store_true', default=False, help='计算回报时考虑时间限制')
    # parser.add_argument("--use_huber_loss", action='store_false', default=False)
    # parser.add_argument("--huber_delta", type=float, default=10.0)
    # parser.add_argument("--episode_length", type=int, default=512, help='A2C 中的前向步数 (默认: 5)')
    # parser.add_argument("--steps_per_update", type=int, default=100, help='每次更新前收集的步数')
    # parser.add_argument("--use-linear-lr-decay", action='store_false', default=False, help='使用线性学习率衰减')
    # parser.add_argument("--step_penalty", type=float, default=0.01, help="每步惩罚，以鼓励更快完成")


    # Ray Tune workers inject additional CLI args; ignore unknowns for compatibility.
    args, _ = parser.parse_known_args(args=argv)
    args.seed = _parse_seed_value(args.seed)
    

    return args
