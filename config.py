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
    parser.add_argument("--env_name", type=str, default='AirSimEnv-v42', help="要训练的环境名称")

    # 算法选择 (Algorithm Selection)
    parser.add_argument("--algorithm_name", type=str, default='CL-ST-DualVimTD3,CL-td3,CL-ST-VimTD3,CL-per_td3',
                        help="要训练的算法。支持: td3, noisy_td3, noisy_td3_type2, ddpg, aetd3, per_td3, per_aetd3, cfc_td3, ST-VimTD3, stv_patch_td3, stv_vim_td3, st_seq_vim_td3 (或 ST-SeqVimTD3), stv_seq_vim_td3 (或 STV-SeqVimTD3), stv_per_vim_td3, ST-SVimTD3, mamba_td3, ST_3DVimTD3, gam_mamba_td3, gam_td3, ST-DualVimTD3, sac, ppo。可以是单个，多个（逗号分隔），或组名 ('all', 'base', 'seq')")
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
    parser.add_argument("--steps_per_update", type=int, default=32, help='每次更新前收集的步数')
    parser.add_argument("--cuda", action='store_false', default=True, help="是否使用CUDA")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="CUDA是否确定性")
    parser.add_argument("--n_training_threads", type=int, default=1, help="训练线程数")
    parser.add_argument("--n_rollout_threads", type=int, default=1, help="Rollout线程数（在AirSim环境中必须为1）")
    parser.add_argument("--max_timesteps", type=int, default=200000, help='要训练的环境步数 (默认: 10e6)')
    parser.add_argument("--buffer_size", type=int, default=40000, help='经验池大小 (注意内存占用: 30000步约占用4GB)')
    parser.add_argument("--learning_starts", type=int, default=5000, help="训练开始前的时间步数 (兼容 start_timesteps)")
    parser.add_argument("--gradient_steps", type=float, default=0.8, help="每次收集数据后的梯度更新倍数")
    parser.add_argument("--episode_length", type=int, default=600, help='每个环境中的最大回合长度')
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
    parser.add_argument("--grad_clip", type=float, default=4.0, help="梯度裁剪")

    # 可视化 (Visualization)
    parser.add_argument("--render_window", action='store_true', default=False, help="显示实时可视化窗口 (默认开启，可用 --no-render_window 关闭)")
    parser.add_argument("--depth_view_scale", type=float, default=2.5, help="深度图显示窗口放大倍数")
    
    
        
    # 图像帧数参数 (所有算法统一的帧堆叠/序列长度)
    parser.add_argument("--n_frames", type=int, default=4, help="图像帧数（非时序算法为堆叠帧数，时序算法为序列长度）")

    # 算法专属参数已迁移到各算法目录下的 params.yaml，
    # 例如 algorithm/td3/params.yaml、algorithm/sac/params.yaml。
    # 这里只保留公共参数定义。
    # 连续控制参数 (Continuous Control Parameters)
    parser.add_argument("--min_forward_speed", type=float, default=0.0, help="最小前进速度 (m/s)")
    parser.add_argument("--max_forward_speed", type=float, default=2.0, help="最大前进速度 (m/s)")
    parser.add_argument("--max_vertical_speed", type=float, default=0.5, help="最大垂直速度 (m/s)")
    parser.add_argument("--max_yaw_rate", type=float, default=np.pi/3, help="最大偏航角速度 (rad/s)")
    parser.add_argument("--takeoff_height", type=float, default=-2.0, help="起飞目标高度 (NED坐标系中负值为向上)")
    parser.add_argument("--action_duration", type=float, default=0.1, help="在时钟缩放之前的每个速度指令的基础持续时间 (秒)")
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
        
    # 停滞惩罚 (Stagnation Penalty) —— 基于滑动窗口位移
    parser.add_argument("--use_stagnation_penalty", action='store_true', default=True, help="是否启用停滞惩罚")
    parser.add_argument("--stagnation_window", type=int, default=10, help="停滞惩罚滑动窗口步数")
    parser.add_argument("--stagnation_window_threshold", type=float, default=0.5, help="滑动窗口内最低累计位移（米），低于此值触发惩罚")
    parser.add_argument("--stagnation_weight", type=float, default=10.0, help="停滞惩罚权重")

    # 保存 (Saving)
    parser.add_argument("--save_interval", type=int, default=20, help="检查点保存频率 (单位: epoch)")
    parser.add_argument("--continue_last", default=False, help="是否继续上次的训练")
    
    # 训练循环参数 (Training Loop Parameters)
    
    parser.add_argument("--step_penalty", type=float, default=0.1, help="每步惩罚，以鼓励更快完成")

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
    parser.add_argument("--disable_game_restart", action='store_true', default=False, help="禁用 UE4 进程重启，仅尝试 AirSim 客户端重连")
    parser.add_argument("--ue4_rpc_fail_threshold", type=int, default=2, help="UE4健康检测中，连续RPC失败达到该次数后触发强制重启")
    parser.add_argument("--ue4_health_check_interval", type=float, default=10.0, help="UE4健康检查最小间隔秒数，降低对训练速度的影响")
    parser.add_argument("--ue4_window_check_interval", type=float, default=10.0, help="窗口状态检测间隔秒数（较慢但开销更大，建议大于健康检查间隔）")
    parser.add_argument("--settings_file", type=str, default="", help="AirSim settings.json 文件路径 (可选)")
    parser.add_argument("--load_model", type=str, default="", help="要加载的模型路径")
    # Ray Tune workers inject additional CLI args; ignore unknowns for compatibility.
    args, _ = parser.parse_known_args(args=argv)
    args.seed = _parse_seed_value(args.seed)
    

    return args
