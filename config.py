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
    parser.add_argument("--env_name", type=str, default='AirSimEnv-v42', help="要训练的环境名称")  # AirSimEnv-v42  AirSimEnv-Gradient-v1

    # 算法选择 (Algorithm Selection)
    parser.add_argument("--algorithm_name", type=str, default='CL-ST-VimTD3',
                        help="要训练的算法。支持: td3, noisy_td3, noisy_td3_type2, ddpg, aetd3, per_td3, per_aetd3, cfc_td3, ST-VimTD3, stv_patch_td3, stv_vim_td3, stv_per_vim_td3, ST-SVimTD3, mamba_td3, ST_3DVimTD3, gam_mamba_td3, gam_td3, ST-DualVimTD3, sac, ppo。可以是单个，多个（逗号分隔），或组名 ('all', 'base', 'seq')")
    parser.add_argument("--smooth_window", type=int, default=300, help="平滑窗口大小，用于平滑学习曲线 (仅对移动平均有效)")
    parser.add_argument("--smooth_method", type=str, default="moving", choices=["moving","zero_phase_des"], help="曲线平滑方法: moving=滑动平均, zero_phase_des=零相位双重指数平滑")
    parser.add_argument("--smooth_alpha", type=float, default=0.05, help="零相位双重指数平滑的水平平滑因子 (0-1)，越大越关注近期数据")
    parser.add_argument("--smooth_beta", type=float, default=0.3, help="零相位双重指数平滑的趋势平滑因子 (0-1)，越大越关注近期趋势变化")
    parser.add_argument("--plot_cl", action='store_true', default=False, help="绘图时是0否检索带 CL- 前缀的算法 (默认: True)")
    parser.add_argument("--plot_non_cl", action='store_true', default=True, help="绘图时是否检索常规算法 (默认: True)")
    parser.add_argument("--use_percentile", action='store_true', default=False, help="使用四分位范围作为阴影带而不是均值加置信区间")
    parser.add_argument("--ci_type", type=str, default="std", choices=["std", "sem"], help="阴影区域类型: std=标准差, sem=标准误差")
    parser.add_argument("--resample_points", type=int, default=512, help="baselines 风格曲线聚合的重采样点数")
    parser.add_argument("--curve_smooth_step", type=float, default=1.0, help="baselines 风格 EMA 重采样 smooth_step")

    # 训练设置 (Training Setup)
    parser.add_argument("--seed", type=str, default="3,4,5", help="随机种子 (支持逗号分隔多个种子)")
    parser.add_argument("--curriculum_start_level", type=int, default=0, choices=[0, 1, 2, 3], help="课程学习起始等级 (0-3, 默认: 0)。注意：算法名以 'CL-' 前缀开头时自动启用课程学习")
    parser.add_argument("--non_curriculum_level", type=int, default=2, choices=[0, 1, 2, 3], help="非课程学习时的固定难度等级 (0-3, 默认: 3)")
    parser.add_argument("--steps_per_update", type=int, default=100, help='每次更新前收集的步数')
    parser.add_argument("--cuda", action='store_false', default=True, help="是否使用CUDA")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="CUDA是否确定性")
    parser.add_argument("--n_training_threads", type=int, default=1, help="训练线程数")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="Rollout线程数（在AirSim环境中必须为1）")
    parser.add_argument("--max_timesteps", type=int, default=60000, help='要训练的环境步数 (默认: 10e6)')
    parser.add_argument("--buffer_size", type=int, default=20000, help='经验池大小 (注意内存占用: 30000步约占用4GB)')
    parser.add_argument("--learning_starts", type=int, default=2000, help="训练开始前的时间步数 (兼容 start_timesteps)")
    parser.add_argument("--gradient_steps", type=float, default=0.5, help="每次收集数据后的梯度更新倍数")
    parser.add_argument("--episode_length", type=int, default=300, help='每个环境中的最大回合长度')
    parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--base_feature_dim", type=int, default=32, help="基础状态先映射到该维度，再与视觉特征拼接")
    parser.add_argument("--exploration_noise", type=float, default=0.20, help="探索噪声")
    parser.add_argument("--exploration_noise_final", type=float, default=0.10, help="探索噪声最终值（用于线性递减）")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子") 
    parser.add_argument("--tau", type=float, default=0.005, help="软更新参数")
    parser.add_argument("--actor_lr", type=float, default=7e-4, help="Actor学习率")
    parser.add_argument("--critic_lr", type=float, default=7e-4, help="Critic学习率")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="策略噪声")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="噪声裁剪")
    parser.add_argument("--policy_freq", type=int, default=2, help="策略更新频率")
    parser.add_argument("--noisy_td3_sigma_init", type=float, default=0.5, help="NoisyTD3: NoisyLinear 初始噪声系数")
    parser.add_argument("--grad_clip", type=float, default=4.0, help="梯度裁剪")

    # 可视化 (Visualization)
    parser.add_argument("--render_window", action='store_true', default=False, help="显示实时可视化窗口 (默认开启，可用 --no-render_window 关闭)")
    parser.add_argument("--depth_view_scale", type=float, default=2.5, help="深度图显示窗口放大倍数")
    
    
        
    # 图像帧数参数 (所有算法统一的帧堆叠/序列长度)
    parser.add_argument("--n_frames", type=int, default=4, help="图像帧数（非时序算法为堆叠帧数，时序算法为序列长度）")

    # SAC 参数 (Soft Actor-Critic)
    parser.add_argument("--auto_entropy_tuning", action='store_true', default=True, help="SAC: 是否自动调整熵温度系数")
    parser.add_argument("--alpha", type=float, default=0.2, help="SAC: 固定的熵温度系数 (当auto_entropy_tuning=False时使用)")
    
    # CfC
    parser.add_argument("--cfc_lr", type=float, default=1e-2, help="CfC 时间序列模块学习率")
    parser.add_argument("--cfc_units", type=int, default=32, help="NCPs 拓扑总神经元数")
    parser.add_argument("--cfc_motor_units", type=int, default=8, help="NCPs 拓扑输出 (运动) 神经元数")
    
    # 时序Mamba参数 (Temporal Mamba Parameters)
    parser.add_argument("--mamba_d_state", type=int, default=16, help="时序Mamba SSM状态维度")
    parser.add_argument("--mamba_d_conv", type=int, default=4, help="时序Mamba卷积核大小")
    parser.add_argument("--mamba_expand", type=int, default=2, help="时序Mamba扩展因子")
    parser.add_argument("--attention_dropout", type=float, default=0.0, help="自注意力dropout率")
    parser.add_argument("--mamba_td3_temporal_depth", type=int, default=2, help="mamba_td3 中时序 Mamba 堆叠层数")

    # GAM-Mamba-TD3 参数
    parser.add_argument("--gam_mamba_layers", type=int, default=2, help="GAM-Mamba-TD3中Mamba块堆叠层数")
    parser.add_argument("--gam_mamba_d_state", type=int, default=16, help="GAM-Mamba-TD3中Mamba的SSM状态维度")
    parser.add_argument("--gam_mamba_d_conv", type=int, default=4, help="GAM-Mamba-TD3中Mamba卷积核宽度")
    parser.add_argument("--gam_mamba_expand", type=int, default=2, help="GAM-Mamba-TD3中Mamba扩展因子")

    # ST-Mamba 参数
    parser.add_argument("--st_mamba_embed_dim", type=int, default=32, help="ST-Mamba 嵌入维度")
    parser.add_argument("--st_mamba_depth", type=int, default=1, help="ST-Mamba Block 数量")
    parser.add_argument("--st_mamba_patch_size", type=int, default=32, help="ST-Mamba Patch 大小")
    parser.add_argument("--st_mamba_d_state", type=int, default=32, help="ST-Mamba SSM 状态维度")
    parser.add_argument("--st_mamba_d_conv", type=int, default=4, help="ST-Mamba SSM 卷积宽度")
    parser.add_argument("--st_mamba_expand", type=int, default=2, help="ST-Mamba Block 扩展因子")
    parser.add_argument("--st_mamba_drop_rate", type=float, default=0.05, help="ST-Mamba Dropout 率 (pos_drop)")
    parser.add_argument("--st_mamba_drop_path_rate", type=float, default=0.05, help="ST-Mamba Drop Path 率 (stochastic depth)")
    parser.add_argument("--st_mamba_temporal_depth", type=int, default=1, help="ST-Mamba-VimTokens 时序 Mamba Block 数量")

    # ST-3DVimTD3 参数 (ST-3DVimTD3 Parameters)
    parser.add_argument("--st_3d_patch_size", type=str, default="2,8,8", help="ST-3DVimTD3 3D Patch 大小 (时间,高度,宽度)，用逗号分隔，如 '2,4,4'")

    # ST-VimTD3 Safety Layer 参数

    parser.add_argument("--use_vim_safety_layer", dest="use_vim_safety_layer", action='store_true', help="启用基于Vim隐空间的Safety Layer")
    parser.add_argument("--no_vim_safety_layer", dest="use_vim_safety_layer", action='store_false', help="禁用基于Vim隐空间的Safety Layer")
    parser.set_defaults(use_vim_safety_layer=True)
    parser.add_argument("--safety_lr", type=float, default=5e-4, help="Safety Constraint Head 学习率")
    parser.add_argument("--safety_loss_coef", type=float, default=1.0, help="安全监督损失系数")
    parser.add_argument("--safety_actor_penalty_coef", type=float, default=0.05, help="Actor 的约束违反惩罚系数")
    parser.add_argument("--safety_warmup_steps", type=int, default=0, help="开始训练Safety Head前的迭代步数")
    parser.add_argument("--safety_end_to_end", action='store_true', default=False, help="是否让Safety损失回传并更新Vim Encoder")
    parser.add_argument("--safety_label_mode", type=str, default="collision", choices=["collision"], help="Safety标签来源：真实碰撞标记")


    # Adaptive Ensemble TD3
    parser.add_argument("--adaptive_k", type=int, default=5, help="Ensemble critics 的数量")
    parser.add_argument("--adaptive_reg", type=float, default=0.001, help="Adaptive ensemble 的正则化系数")
    parser.add_argument("--adaptive_reg_final", type=float, default=0.0001, help="Adaptive ensemble 正则化系数的最终值 (用于线性衰减)")
    parser.add_argument("--adaptive_meta_lr", type=float, default=1e-3, help="元网络 (Meta network) 学习率")

    # PER-TD3 双经验池参数 (Dual-Buffer PER-TD3)
    parser.add_argument("--per_td3_success_capacity_ratio", type=float, default=0.3, help="成功经验池容量占总buffer比例")
    parser.add_argument("--per_td3_alpha", type=float, default=0.6, help="PER 优先级指数 alpha")
    parser.add_argument("--per_td3_priority_eps", type=float, default=1e-6, help="PER 优先级最小平滑项 eps")
    parser.add_argument("--per_td3_beta_start", type=float, default=0.4, help="PER 重要性采样权重 beta 初始值")
    parser.add_argument("--per_td3_beta_final", type=float, default=1.0, help="PER 重要性采样权重 beta 最终值")
    parser.add_argument("--per_td3_mu_low", type=float, default=0.15, help="成功经验优先采样比例 mu 的早期值")
    parser.add_argument("--per_td3_mu_mid", type=float, default=0.30, help="成功经验优先采样比例 mu 的中期值")
    parser.add_argument("--per_td3_mu_high", type=float, default=0.45, help="成功经验优先采样比例 mu 的后期值 (不宜过高)")
    parser.add_argument("--per_td3_mu_step1", type=float, default=0.25, help="mu 阶梯函数第一阈值 (训练进度比例)")
    parser.add_argument("--per_td3_mu_step2", type=float, default=0.65, help="mu 阶梯函数第二阈值 (训练进度比例)")
    # 连续控制参数 (Continuous Control Parameters)
    parser.add_argument("--min_forward_speed", type=float, default=0.0, help="最小前进速度 (m/s)")
    parser.add_argument("--max_forward_speed", type=float, default=2.0, help="最大前进速度 (m/s)")
    parser.add_argument("--max_vertical_speed", type=float, default=0.5, help="最大垂直速度 (m/s)")
    parser.add_argument("--max_yaw_rate", type=float, default=np.pi/3, help="最大偏航角速度 (rad/s)")
    parser.add_argument("--takeoff_height", type=float, default=-2.0, help="起飞目标高度 (NED坐标系中负值为向上)")
    parser.add_argument("--action_duration", type=float, default=0.5, help="在时钟缩放之前的每个速度指令的基础持续时间 (秒)")
    parser.add_argument("--clock_speed_factor", type=float, default=1.0, help="AirSim 设置中配置的 ClockSpeed 因子；持续时间将除以此值")

    # 飞行高度限制 (Flight Altitude Limits)
    parser.add_argument("--max_flight_altitude", type=float, default=4.0, help="最大飞行高度 (米, 正值为向上)")
    parser.add_argument("--min_flight_altitude", type=float, default=0.0, help="最小飞行高度 (米, 正值为向上)")
    parser.add_argument("--min_altitude_penalty", type=float, default=1.0, help="低于最小高度的惩罚")
    parser.add_argument("--max_altitude_penalty", type=float, default=3.0, help="高于最大高度的惩罚")
    parser.add_argument("--altitude_penalty_value", type=float, default=0.5, help="飞出高度惩罚范围时的固定惩罚值")
    parser.add_argument("--takeoff_obstacle_threshold_m", type=float, default=2.0, help="起飞后四向避障最小安全距离阈值 (米)")
    parser.add_argument("--takeoff_lidar_name", type=str, default="LidarSensor1", help="起飞后避障检查使用的激光雷达名称")
    parser.add_argument("--takeoff_obstacle_reset_retries", type=int, default=3, help="起飞后近障触发重置的最大重试次数")
    parser.add_argument("--reward_lidar_name", type=str, default="LidarSensor1", help="奖励计算使用的激光雷达名称")
    parser.add_argument("--lidar_safe_distance_m", type=float, default=2.0, help="激光雷达避障惩罚安全距离阈值 (米)")
    parser.add_argument("--lidar_log_penalty_weight", type=float, default=1.0, help="激光雷达对数惩罚权重")
    parser.add_argument("--lidar_log_penalty_min", type=float, default=-3.0, help="激光雷达对数惩罚最小值(下限)")
    parser.add_argument("--lidar_penalty_eps", type=float, default=1e-3, help="激光雷达惩罚数值稳定项")
    parser.add_argument("--lidar_distance_cap_m", type=float, default=5.0, help="激光雷达距离裁剪上限 (米)")
    parser.add_argument("--lidar_query_max_attempts", type=int, default=1, help="每步激光雷达查询最大重试次数")
    parser.add_argument("--lidar_query_retry_sleep", type=float, default=0.02, help="激光雷达查询重试间隔 (秒)")
    parser.add_argument("--lidar_h_bins", type=int, default=36, help="激光雷达水平角离散束数（NavRL风格，建议36）")
    parser.add_argument("--lidar_v_bins", type=int, default=3, help="激光雷达垂直层数（你当前需求建议3）")
    parser.add_argument("--lidar_vfov_min_deg", type=float, default=-10.0, help="激光雷达垂直视场最小角(度)")
    parser.add_argument("--lidar_vfov_max_deg", type=float, default=20.0, help="激光雷达垂直视场最大角(度)")

    # =============================================================================
    # AirGym_GradientReward 奖励函数参数 (Gradient Reward Parameters)
    # =============================================================================
    # 核心权重
    parser.add_argument("--grad_goal_weight", type=float, default=1.2, help="梯度奖励：目标距离项权重")
    parser.add_argument("--grad_heading_weight", type=float, default=0.35, help="梯度奖励：朝向误差项权重")
    parser.add_argument("--grad_obstacle_weight", type=float, default=0.50, help="梯度奖励：障碍物风险项权重")
    parser.add_argument("--grad_altitude_weight", type=float, default=0.30, help="梯度奖励：高度误差项权重")
    parser.add_argument("--grad_progress_weight", type=float, default=8.0, help="梯度奖励：进度项权重")
    
    # 惩罚与裁剪
    parser.add_argument("--grad_step_penalty", type=float, default=0.2, help="梯度奖励：每步时间成本")
    parser.add_argument("--grad_reward_clip", type=float, default=8.0, help="梯度奖励：奖励裁剪上限")
    parser.add_argument("--grad_cost_clip", type=float, default=8.0, help="梯度奖励：代价裁剪上限")
    parser.add_argument("--grad_shaping_gamma", type=float, default=1.0, help="梯度奖励：势能折扣因子")
    
    # 深度图参数
    parser.add_argument("--depth_max_distance", type=float, default=15.0, help="深度图最大距离(米)")
    parser.add_argument("--grad_safe_depth_m", type=float, default=1.0, help="梯度奖励：安全深度阈值(米)")
    parser.add_argument("--grad_depth_floor_m", type=float, default=0.15, help="梯度奖励：深度最小值(米)")
    parser.add_argument("--grad_depth_max_m", type=float, default=15.0, help="梯度奖励：深度最大值(米)")
    parser.add_argument("--grad_depth_percentile", type=float, default=15.0, help="梯度奖励：深度统计百分位")
    parser.add_argument("--grad_obstacle_decay_m", type=float, default=2.0, help="梯度奖励：障碍物风险衰减距离(米)")
    parser.add_argument("--grad_obstacle_balance_weight", type=float, default=0.25, help="梯度奖励：左右障碍不平衡惩罚权重")
    
    # 高度参数
    parser.add_argument("--grad_altitude_band_m", type=float, default=2.0, help="梯度奖励：高度误差归一化带宽(米)")
    
    # 跨课程学习尺度归一化
    parser.add_argument("--grad_distance_scale_min", type=float, default=2.0, help="梯度奖励：距离尺度最小值")
    parser.add_argument("--grad_distance_arena_ratio", type=float, default=0.25, help="梯度奖励：距离与场地比例")
    
    # 平滑控制
    parser.add_argument("--grad_smoothness_weight", type=float, default=0.25, help="梯度奖励：动作平滑性惩罚权重")
    parser.add_argument("--grad_smoothness_deadzone", type=float, default=0.15, help="梯度奖励：平滑性死区")
    
    # 终止条件奖励
    parser.add_argument("--grad_success_reward", type=float, default=20.0, help="梯度奖励：成功到达目标奖励")
    parser.add_argument("--grad_collision_reward", type=float, default=-20.0, help="梯度奖励：碰撞惩罚")
    parser.add_argument("--grad_timeout_reward", type=float, default=-30.0, help="梯度奖励：超时惩罚")

    # 保存 (Saving)
    parser.add_argument("--save_interval", type=int, default=20, help="检查点保存频率 (单位: epoch)")
    parser.add_argument("--continue_last", default=False, help="是否继续上次的训练")
    
    # 训练循环参数 (Training Loop Parameters)
    
    parser.add_argument("--step_penalty", type=float, default=0.5, help="每步惩罚，以鼓励更快完成")

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
    parser.add_argument("--airsim_ip", type=str, default="127.0.0.1", help="AirSim 服务器 IP 地址")
    parser.add_argument("--airsim_port", type=int, default=41451, help="AirSim 服务器端口")
    parser.add_argument("--ue4_rpc_fail_threshold", type=int, default=2, help="UE4健康检测中，连续RPC失败达到该次数后触发强制重启")
    parser.add_argument("--ue4_health_check_interval", type=float, default=10.0, help="UE4健康检查最小间隔秒数，降低对训练速度的影响")
    parser.add_argument("--ue4_window_check_interval", type=float, default=10.0, help="窗口状态检测间隔秒数（较慢但开销更大，建议大于健康检查间隔）")
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

    # =============================================================================
    # 新增 PPO 算法参数 (NEW PPO ALGORITHM PARAMETERS)
    # =============================================================================
    parser.add_argument("--rollout_buffer_size", type=int, default=2048, help="PPO Rollout Buffer 大小 (默认: 2048)")
    parser.add_argument("--ppo_epochs", type=int, default=10, help='PPO 每次数据迭代次数 (默认: 10)')
    parser.add_argument("--ppo_batch_size", type=int, default=64, help='PPO Mini-batch 大小 (默认: 64)')
    parser.add_argument("--clip_range", type=float, default=0.2, help='PPO 裁剪参数 epsilon (默认: 0.2)')
    parser.add_argument("--vf_coef", type=float, default=0.5, help='价值损失系数 (默认: 0.5)')
    parser.add_argument("--ent_coef", type=float, default=0.0, help='熵正则化系数 (默认: 0.0)')
    parser.add_argument("--target_kl", type=float, default=None, help='KL散度阈值，超过则提前停止 (默认: None)')


    # Ray Tune workers inject additional CLI args; ignore unknowns for compatibility.
    args, _ = parser.parse_known_args(args=argv)
    args.seed = _parse_seed_value(args.seed)
    

    return args
