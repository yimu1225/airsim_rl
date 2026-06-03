import argparse
import numpy as np
import os
from algo_name_utils import normalize_algorithm_name_for_config

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
    parser.add_argument("--algorithm_name", type=str, default='CL-DDPG,CL-TD3',
                        help="要训练的算法。支持: TD3, DDPG, DPER_TD3, ST_Vim_TD3, STV_Patch_TD3, Vim_TD3, ST_Seq_Vim_TD3, STV_Seq_Vim_TD3, DPER_ST_Vim_TD3, ST_SVim_TD3, Mamba_TD3, ST_DualVim_TD3, AETD3, SAC, SAC_Beta, LSTM_SAC, ST_Vim_SAC, ST_SVim_SAC, ST_Vim_SAC_Beta, DPER_ST_Vim_SAC, DPER_ST_Vim_SAC_Beta, PPO, ST_Vim_PPO, PL_ST_Vim_PPO, PL_TD3, PL_DPER_TD3, PL_ST_Vim_TD3, PL_SAC, PL_SAC_Beta, PL_ST_Vim_SAC, PL_PER_ST_Vim_SAC, PL_DPER_ST_Vim_SAC, PL_DPER_ST_Vim_SAC_Beta, PL_DPER_ST_Vim_TD3,MM_ST_Vim_SAC,MambaCSJA_SAC")
    parser.add_argument("--smooth_window", type=int, default=300, help="平滑窗口大小，用于平滑学习曲线 (仅对移动平均有效)")
    parser.add_argument("--smooth_method", type=str, default="moving", choices=["moving","zero_phase_des"], help="曲线平滑方法: moving=滑动平均, zero_phase_des=零相位双重指数平滑")
    parser.add_argument("--smooth_alpha", type=float, default=0.05, help="零相位双重指数平滑的水平平滑因子 (0-1)，越大越关注近期数据")
    parser.add_argument("--smooth_beta", type=float, default=0.3, help="零相位双重指数平滑的趋势平滑因子 (0-1)，越大越关注近期趋势变化")
    parser.add_argument("--plot_cl", action='store_true', default=True, help="绘图时是否检索带 CL- 前缀的算法 (默认: True)")
    parser.add_argument("--plot_non_cl", action='store_true', default=True, help="绘图时是否检索常规算法 (默认: True)")
    parser.add_argument("--use_percentile", action='store_true', default=False, help="使用四分位范围作为阴影带而不是均值加置信区间")
    parser.add_argument("--ci_type", type=str, default="std", choices=["std", "sem"], help="阴影区域类型: std=标准差, sem=标准误差")
    parser.add_argument("--resample_points", type=int, default=512, help="baselines 风格曲线聚合的重采样点数")
    parser.add_argument("--curve_smooth_step", type=float, default=1.0, help="baselines 风格 EMA 重采样 smooth_step")

    # 训练设置 (Training Setup)
    parser.add_argument("--seed", type=str, default="1,2,3", help="随机种子 (支持逗号分隔多个种子)")
    parser.add_argument("--curriculum_start_level", type=int, default=0, choices=[0, 1, 2, 3], help="课程学习起始等级 (0-3, 默认: 0)。注意：算法名以 'CL-' 前缀开头时自动启用课程学习")
    parser.add_argument("--curriculum_mode", type=str, default="progress", choices=["progress", "success"], help="课程学习模式: progress=按训练进度连续增加难度, success=按成功率离散切换难度")
    parser.add_argument("--curriculum_progress_max_ratio", type=float, default=0.9, help="progress课程达到最大难度所需的训练进度比例")
    parser.add_argument("--non_curriculum_level", type=int, default=2, choices=[0, 1, 2, 3], help="非课程学习时的固定难度等级 (0-3, 默认: 3)")
    parser.add_argument("--steps_per_update", type=int, default=100, help='每次更新前收集的步数')
    parser.add_argument("--cuda", action='store_false', default=True, help="是否使用CUDA")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="CUDA是否确定性")
    parser.add_argument("--max_timesteps", type=int, default=120000, help='要训练的环境步数 (默认: 10e6)')
    parser.add_argument("--buffer_size", type=int, default=30000, help='经验池大小 (注意内存占用: 30000步约占用4GB)')
    parser.add_argument("--learning_starts", type=int, default=3000, help="训练开始前的时间步数 (兼容 start_timesteps)。在此步数之前使用随机动作探索，之后改用策略网络采样。")
    parser.add_argument("--update_after", type=int, default=3000, help="开始网络更新的时间步数。默认与 learning_starts 相同。在 learning_starts 之后、update_after 之前，将使用策略网络采集经验，但仍不进行训练更新。")
    parser.add_argument("--gradient_steps", type=float, default=0.5, help="每次收集数据后的梯度更新倍数")
    parser.add_argument("--episode_length", type=int, default=300, help='每个环境中的最大回合长度 ')
    parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--base_feature_dim", type=int, default=32, help="基础状态先映射到该维度，再与视觉特征拼接")
    parser.add_argument("--exploration_noise", type=float, default=0.20, help="探索噪声")
    # parser.add_argument("--exploration_noise_final", type=float, default=0.10, help="探索噪声最终值（用于线性递减）")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子") 
    parser.add_argument("--tau", type=float, default=0.003, help="软更新参数")
    parser.add_argument("--actor_lr", type=float, default=4e-4, help="Actor学习率")
    parser.add_argument("--critic_lr", type=float, default=4e-4, help="Critic学习率")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="策略噪声")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="噪声裁剪")
    parser.add_argument("--policy_freq", type=int, default=2, help="策略更新频率")
    parser.add_argument("--grad_clip", type=float, default=1e6, help="梯度裁剪")

    # 可视化 (Visualization)
    parser.add_argument("--render_window", action='store_true', default=False, help="显示实时可视化窗口 (默认开启，可用 --no-render_window 关闭)")
    parser.add_argument("--depth_view_scale", type=float, default=2.5, help="深度图显示窗口放大倍数")

    # 观测图像噪声 (Observation Image Noise)
    parser.add_argument("--enable_observation_noise", action='store_true', default=False, help="是否对传给算法的深度图加噪声")
    parser.add_argument("--disable_observation_noise", action='store_false', dest="enable_observation_noise", help="关闭传给算法的深度图噪声")
    parser.add_argument("--depth_noise_gaussian_std", type=float, default=10.0, help="深度图高斯噪声标准差，作用在0-255图像强度上")
    parser.add_argument("--depth_noise_gaussian_clip", type=float, default=30.0, help="深度图高斯噪声裁剪范围 [-clip, clip]")
    parser.add_argument("--depth_noise_salt_pepper_prob", type=float, default=0.01, help="深度图椒盐噪声总概率，一半salt一半pepper")
    parser.add_argument("--depth_noise_motion_blur_kernel_size", type=int, default=3, help="深度图水平运动模糊kernel大小，<=1表示关闭")
    
    
        
    # 图像帧数参数 (所有算法统一的帧堆叠/序列长度)
    parser.add_argument("--n_frames", type=int, default= 4, help="图像帧数（非时序算法为堆叠帧数，时序算法为序列长度）")

    # 算法专属参数已迁移到各算法目录下的 params.yaml，
    # 例如 algorithm/TD3/params.yaml、algorithm/SAC/params.yaml。
    # 这里只保留公共参数定义。
    # 连续控制参数 (Continuous Control Parameters)
    parser.add_argument("--min_forward_speed", type=float, default=-2.0, help="最小机体系x轴速度 (m/s)")
    parser.add_argument("--max_forward_speed", type=float, default=2.0, help="最大机体系x轴速度 (m/s)")
    parser.add_argument("--max_vertical_speed", type=float, default=0.3, help="最大垂直速度 (m/s)")
    parser.add_argument("--max_yaw_rate", type=float, default=np.pi/3, help="最大偏航角速度 (rad/s)")
    parser.add_argument("--takeoff_height", type=float, default=-1.0, help="起飞目标高度 (NED坐标系中负值为向上)")
    parser.add_argument("--action_duration", type=float, default=0.15, help="每个速度指令的仿真持续时间 (秒)")

    # 飞行高度限制 (Flight Altitude Limits)
    parser.add_argument("--max_flight_altitude", type=float, default=2.5, help="最大飞行高度 (米, 正值为向上)")
    parser.add_argument("--min_flight_altitude", type=float, default=0.0, help="最小飞行高度 (米, 正值为向上)")
    parser.add_argument("--min_altitude_penalty", type=float, default=0.5, help="低于最小高度的惩罚")
    parser.add_argument("--max_altitude_penalty", type=float, default=2.0, help="高于最大高度的惩罚")
    parser.add_argument("--altitude_penalty_value", type=float, default=1.0, help="飞出高度惩罚范围时的固定惩罚值")
    parser.add_argument("--enable_takeoff_obstacle_check", type=str.lower, choices=["true", "false"], default="false", help="是否启用起飞后距离传感器避障检查 (true/false)")
    parser.add_argument("--takeoff_obstacle_reset_retries", type=int, default=3, help="起飞后近障触发重置的最大重试次数")
    parser.add_argument("--distance_sensor_count", type=int, default=108, help="3层距离传感器总数 (每层36个, 俯仰-15°/0°/+15°)")
    parser.add_argument("--distance_sensor_prefix", type=str, default="DistanceSensor", help="距离传感器自动命名前缀")
    parser.add_argument("--distance_sensor_start_index", type=int, default=0, help="距离传感器自动命名起始编号")
    parser.add_argument("--distance_sensor_names", type=str, default="", help="距离传感器名称列表，逗号分隔；为空时按 prefix+编号生成")

    parser.add_argument("--distance_sensor_log_penalty_min", type=float, default=-5.0, help="距离传感器对数惩罚最小值(下限)")
    parser.add_argument("--distance_sensor_penalty_max_distance", type=float, default=2.0, help="静态障碍物惩罚的最大距离 (米)")
    parser.add_argument("--distance_sensor_penalty_eps", type=float, default=1e-3, help="距离传感器惩罚数值稳定项")
    parser.add_argument("--distance_sensor_query_max_attempts", type=int, default=1, help="每步距离传感器查询最大重试次数")
    parser.add_argument("--distance_sensor_query_retry_sleep", type=float, default=0.02, help="距离传感器查询重试间隔 (秒)")
        
    # 停滞惩罚 (Stagnation Penalty) —— 基于滑动窗口位移
    parser.add_argument("--use_stagnation_penalty", action='store_true', default=True, help="是否启用停滞惩罚")
    parser.add_argument("--stagnation_window", type=int, default=10, help="停滞惩罚滑动窗口步数")
    parser.add_argument("--stagnation_window_threshold", type=float, default=0.5, help="滑动窗口内最低累计位移（米），低于此值触发惩罚")
    parser.add_argument("--stagnation_weight", type=float, default=5.0, help="停滞惩罚权重")

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
    args = parser.parse_args(args=argv)
    args.seed = _parse_seed_value(args.seed)
    args.enable_takeoff_obstacle_check = (args.enable_takeoff_obstacle_check == "true")
    args.algorithm_name = normalize_algorithm_name_for_config(args.algorithm_name)
    

    return args
