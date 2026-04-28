from settings_folder import settings
import msgs
import numpy as np
import math
import time
from environment_randomization.game_config_handler_class import *
from game_handling.game_handler_class import GameHandler
import gymnasium as gym
import collections
from gymnasium import spaces
from gymnasium.utils import seeding
from gym_airsim.envs.airlearningclient import *
from common.utils import *


class AirSimEnv(gym.Env):
    """
    AirSim 强化学习环境。
    该类根据 OpenAI Gym 的接口标准实现了 AirSim 环境的包装。
    
    Attributes:
        stack_frames (int): 堆叠帧数，用于作为状态输入 (默认4帧)。
        observation_space (gymnasium.spaces.Box): 观测空间，基于堆叠后的深度图像。
        action_space: 动作空间。可以是离散的（8个动作）或连续的（速度控制）。
        game_handler (GameHandler): 游戏处理器，用于启动/重启 AirSim。
        game_config_handler (GameConfigHandler): 游戏配置处理器，用于管理环境配置。
        airgym (AirLearningClient): 与 AirSim 交互的客户端实例。
        goal (np.array): 目标点坐标。
    """
    def __init__(self, takeoff_height=-0.9, config=None, stack_frames=4):
        """
        初始化 AirSim 环境。
        
        Args:
            takeoff_height (float): 起飞高度 (NED坐标，负数为高度)。
            config: 配置对象，包含高度限制等参数。
            stack_frames (int): 堆叠帧数 (默认4帧)。如果为1，则不进行堆叠（用于RNN/LSTM）。
        """

        self.config = config  # Store config for reward calculations

        # 从config读取高度限制参数，如果没有config则使用默认值
        
        self.max_altitude = config.max_flight_altitude
        self.min_altitude = config.min_flight_altitude
        self.min_altitude_penalty = config.min_altitude_penalty
        self.max_altitude_penalty = config.max_altitude_penalty
        self.altitude_penalty_value = config.altitude_penalty_value
    
        # 停滞惩罚参数
        self.use_stagnation_penalty = config.use_stagnation_penalty
        self.stagnation_window = config.stagnation_window
        self.stagnation_window_threshold = config.stagnation_window_threshold
        self.stagnation_weight = config.stagnation_weight
        self.displacement_window = collections.deque(maxlen=self.stagnation_window)
    
        self.stack_frames = stack_frames
        self.episode_reward = 0

        self.base_dim = 11
        self.depth_shape = None  # 将在创建AirSim客户端后根据实际分辨率设置
        self.observation_space = None  # 延迟设置

        self.depth_stack = collections.deque(maxlen=self.stack_frames)
        self.clean_depth_stack = collections.deque(maxlen=self.stack_frames)

        # 速度需要大于 2 或者持续时间大于 0.4
        # 否则效果不佳！
        if (settings.control_mode == "moveByVelocity"):
            self.action_space = spaces.Box(np.array([-0.3, -0.3], dtype=np.float32),
                                           np.array([+0.3, +0.3], dtype=np.float32),
                                           dtype=np.float32)
        elif (settings.control_mode == "Continuous"):
            # Continuous action space: [forward_speed, z_velocity, yaw_rate]
            fwd_min = config.min_forward_speed
            fwd_max = config.max_forward_speed
            z_max = config.max_vertical_speed
            yaw_max = config.max_yaw_rate

            self.action_space = spaces.Box(
                low=np.array([fwd_min, -z_max, -yaw_max], dtype=np.float32),
                high=np.array([fwd_max, z_max, yaw_max], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(8)

        # 课程学习设置（提前判断，因为需要在创建 GameConfigHandler 时使用）
        # 通过算法名前缀判断是否使用课程学习 (CL-前缀表示启用)
        algorithm_name = config.algorithm_name
        self.use_curriculum = algorithm_name.upper().startswith("CL-")
        self.curriculum_start_level = config.curriculum_start_level
        self.non_curriculum_level = config.non_curriculum_level
        
        # Level 到配置字典的映射
        level_config_map = {
            0: "settings.easy_range_dic",
            1: "settings.medium_range_dic",
            2: "settings.hard_range_dic",
            3: "settings.dynamic_obstacles_dic"
        }
        
        # 确定目标 level
        if self.use_curriculum:
            target_level = self.curriculum_start_level
        else:
            target_level = self.non_curriculum_level
        self.level = target_level
        
        # 关键：先创建 GameConfigHandler 并写入 JSON，确保 UE4 启动时能读取正确的配置
        # UE4 环境配置（必须在 restart_game 之前，这样 UE4 启动时会读取正确的 JSON）
        config_name = level_config_map[target_level]
        self.game_config_handler = GameConfigHandler(range_dic_name=config_name)
        
        # 保存基础种子用于确定性环境采样
        self.base_seed = int(config.seed)
        self.seed(self.base_seed)
        
        # 环境变化计数器（第几次环境变化）
        self.change_counter = 0
        
        # 立即采样并写入 JSON，使用确定性采样（change_counter=0 表示初始环境）
        # 这必须在 UE4 启动之前完成！
        sample_vars = ["Seed", "ArenaSize", "NumberOfObjects", "End", "Walls1", "MinimumDistance"]
        # Always sample dynamic object count to avoid carrying stale values from previous runs.
        # For non-dynamic levels, the configured range is [0], so this deterministically clears dynamics.
        sample_vars.append("NumberOfDynamicObjects")
        self.game_config_handler.sample(*sample_vars, change_counter=0, base_seed=self.base_seed)

        # 现在启动 UE4，它会读取上面写入的 JSON
        disable_game_restart = config.disable_game_restart
        self.game_handler = None if disable_game_restart else GameHandler()
        if self.game_handler is not None:
            self.game_handler.restart_game()

        print("Scene initialization complete")

        # 无人机 API，传入config中的ip和port参数
        client_ip = config.airsim_ip if config is not None else None
        client_port = config.airsim_port if config is not None else None
        self.airgym = AirLearningClient(z=takeoff_height, ip=client_ip, port=client_port)

        # 从AirSim获取一次深度图，确定原始分辨率，并设置observation_space
        try:
            depth_sample = self.airgym.getScreenDepth(max_attempts=3)
            STATE_DEPTH_H, STATE_DEPTH_W = depth_sample.shape
            print(f"Detected AirSim depth resolution: {STATE_DEPTH_W}x{STATE_DEPTH_H}")
        except Exception as e:
            print(f"WARNING: Failed to get depth sample during init, falling back to 128x128: {e}")
            STATE_DEPTH_H, STATE_DEPTH_W = 128, 128
        
        self.depth_shape = (self.stack_frames, STATE_DEPTH_H, STATE_DEPTH_W)
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=np.float32(0), high=np.float32(255), shape=self.depth_shape, dtype=np.float32),
            "base": spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_dim,), dtype=np.float32)
        })

        # 动作持续时间使用仿真时间，不再跟 AirSim ClockSpeed 绑定。
        self.action_duration = config.action_duration

        # 重置环境变量
        self.success_count = 0
        self.episodeN = 0
        self.stepN = 0
        self.goal = airsimize_coordinates(self.game_config_handler.get_cur_item("End"))


        self.init_state_f()
        self.prev_state = self.get_obs()
        self.init_state = self.prev_state
        self.success = False
        
        # 使用训练配置中的 seed 初始化随机数生成器，确保首次环境采样可复现
        self.success_deque = collections.deque(maxlen=256)

        self.ue4_rpc_fail_count = 0
        self.ue4_rpc_fail_threshold = config.ue4_rpc_fail_threshold
        self.ue4_health_check_interval = config.ue4_health_check_interval
        self.ue4_window_check_interval = config.ue4_window_check_interval
        self.ue4_process_check_interval = max(3.0, self.ue4_health_check_interval * 3.0)
        self._last_ue4_health_check_ts = 0.0
        self._last_process_check_ts = 0.0
        self._last_window_check_ts = 0.0
        self._cached_process_alive = True
        self._cached_window_alive = None

        self.enable_takeoff_obstacle_check = bool(config.enable_takeoff_obstacle_check)
        self.takeoff_obstacle_threshold_m = float(config.takeoff_obstacle_threshold_m)
        self.takeoff_obstacle_reset_retries = max(0, int(config.takeoff_obstacle_reset_retries))
        self.takeoff_lidar_name = str(config.takeoff_lidar_name).strip()
        self.reward_lidar_name = str(config.reward_lidar_name).strip()
        if self.reward_lidar_name == "":
            self.reward_lidar_name = self.takeoff_lidar_name
        self.lidar_safe_distance_m = max(0.05, float(config.lidar_safe_distance_m))
        self.lidar_log_penalty_weight = float(config.lidar_log_penalty_weight)
        self.lidar_log_penalty_min = float(config.lidar_log_penalty_min)
        self.lidar_penalty_eps = max(1e-6, float(config.lidar_penalty_eps))
        self.lidar_distance_cap_m = max(
            self.lidar_safe_distance_m,
            float(config.lidar_distance_cap_m),
        )
        self.lidar_query_max_attempts = max(1, int(config.lidar_query_max_attempts))
        self.lidar_query_retry_sleep = max(0.0, float(config.lidar_query_retry_sleep))
        self.lidar_h_bins = max(4, int(config.lidar_h_bins))
        self.lidar_v_bins = max(1, int(config.lidar_v_bins))
        self.lidar_vfov_min_deg = float(config.lidar_vfov_min_deg)
        self.lidar_vfov_max_deg = float(config.lidar_vfov_max_deg)
        self.last_lidar_obstacle_penalty = 0.0
        self.last_lidar_scan_distance = np.full(
            (self.lidar_h_bins, self.lidar_v_bins),
            self.lidar_distance_cap_m,
            dtype=np.float32,
        )

        # Initialize previous action for jerk penalty calculation
        if hasattr(self.action_space, 'shape') and self.action_space.shape:
            self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        else:
            self.prev_action = 0  # For discrete actions
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.prev_pos_xy = None

    def _reconnect_airsim_client(self, reason=""):
        """
        Reconnect AirSim client with retries after UE recovery.
        Returns True on success, False on failure.
        """
        z = -1.0
        ip = None
        port = None
        if hasattr(self, 'airgym'):
            z = self.airgym.z
        if hasattr(self, 'config') and self.config:
            ip = self.config.airsim_ip
            port = self.config.airsim_port

        max_retry = 5
        for attempt in range(max_retry):
            try:
                self.airgym = AirLearningClient(z=z, ip=ip, port=port)
                self.ue4_rpc_fail_count = 0
                return True
            except Exception as e:
                if attempt == max_retry - 1:
                    print(f"WARNING: AirSim reconnect failed after UE recovery ({reason or 'unknown reason'}): {e}")
                time.sleep(1.5)
        return False

    def _is_airsim_rpc_alive(self):
        """
        Lightweight RPC health check for AirSim.
        Returns True when simulator RPC responds; False otherwise.
        """
        if not hasattr(self, 'airgym') or self.airgym is None or not hasattr(self.airgym, 'client'):
            return False

        try:
            # 优化：使用更轻量的日志消息API代替姿态查询
            # 原方法：simGetVehiclePose() 需要获取完整的车辆状态信息，较慢
            # self.airgym.client.simGetVehiclePose()
            
            # 新方法：simPrintLogMessage() 仅发送简短日志消息，更快
            self.airgym.client.simPrintLogMessage("health_check", "", 0)
            return True
        except Exception:
            return False

    def getGoal(self):
        return self.goal

    def get_space(self):
        return self.observation_space,self.action_space

    def check_ue4_status(self, force_restart=False, reason=""):
        """
        Check if UE process/RPC is healthy. If not, force restart and reconnect client.
        Returns True if restart/recovery happened.
        """
        now_ts = time.monotonic()
        elapsed = now_ts - self._last_ue4_health_check_ts
        if (not force_restart) and (elapsed < self.ue4_health_check_interval):
            return False
        self._last_ue4_health_check_ts = now_ts

        # Fast path: process check first (cheap)
        if self.game_handler:
            if (now_ts - self._last_process_check_ts) >= self.ue4_process_check_interval:
                self._last_process_check_ts = now_ts
                self._cached_process_alive = self.game_handler.is_game_process_alive()

            process_alive = self._cached_process_alive
            if not process_alive:
                restarted = self.game_handler.check_and_recover_game(
                    force_restart=True,
                    reason="process_missing",
                    check_window=False,
                )
                if restarted:
                    self._cached_process_alive = True
                    self._last_process_check_ts = now_ts
                    return self._reconnect_airsim_client(reason="process_missing")
                return False

            # Window check is expensive, run at a lower frequency.
            if (now_ts - self._last_window_check_ts) >= self.ue4_window_check_interval:
                self._last_window_check_ts = now_ts
                self._cached_window_alive = self.game_handler.is_game_window_alive()

            if self._cached_window_alive is False:
                restarted = self.game_handler.check_and_recover_game(
                    force_restart=True,
                    reason="ue_window_closed",
                    check_window=False,
                )
                self._cached_window_alive = None
                if restarted:
                    self._cached_process_alive = True
                    self._last_process_check_ts = now_ts
                    return self._reconnect_airsim_client(reason="ue_window_closed")
                return False

        rpc_alive = self._is_airsim_rpc_alive()
        if rpc_alive:
            self.ue4_rpc_fail_count = 0
        else:
            self.ue4_rpc_fail_count += 1

        rpc_force_restart = self.ue4_rpc_fail_count >= self.ue4_rpc_fail_threshold
        if rpc_force_restart and not force_restart:
            force_restart = True
            reason = f"rpc_unhealthy_{self.ue4_rpc_fail_count}_times"

        if self.game_handler and force_restart:
            restarted = self.game_handler.check_and_recover_game(
                force_restart=force_restart,
                reason=reason,
                check_window=False,
            )
            if restarted:
                self._cached_process_alive = True
                self._last_process_check_ts = now_ts
                self._cached_window_alive = None
                return self._reconnect_airsim_client(reason=reason or "forced_restart")
        elif force_restart:
            # Fallback when restart is disabled: reconnect client only.
            if self._reconnect_airsim_client(reason=reason or "rpc_unhealthy_no_game_handler"):
                return True
            return False
        return False

    def seed(self, seed=None):
        """
        设置随机数种子。
        """
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def print_msg_of_inspiration(self):
        if (self.success_count %2 == 0):
            print("---------------:) :) :) Success, Be Happy (: (: (:------------ !!!\n")
        elif (self.success_count %3 == 0):
            print("---------------:) :) :) Success, Shake Your Butt (: (: (:------------ !!!\n")
        else:
            print("---------------:) :) :) Success, Oh Yeah! (: (: (:------------ !!!\n")

    def get_obs(self):
        inform = self.state()
        # 归一化base state
        # inform = self.normalize_base_state(inform)
        
        depth_stack_np = np.array(self.depth_stack, dtype=np.float32)
        return {
            "depth": depth_stack_np,
            "base": inform
        }

    def _get_zero_depth(self):
        """返回与当前深度图像分辨率一致的全零数组"""
        h, w = self.depth_shape[-2], self.depth_shape[-1]
        return np.zeros((h, w), dtype=np.float32)

    def _get_latest_clean_depth(self, noisy_depth):
        clean = getattr(self.airgym, "last_depth_clean", None)
        if clean is None:
            clean = noisy_depth
        clean = np.asarray(clean, dtype=np.float32)
        if clean.ndim == 3:
            clean = clean[0] if clean.shape[0] > 0 else self._get_zero_depth()
        expected_hw = self.depth_shape[-2:]
        if clean.shape != expected_hw:
            return self._get_zero_depth()
        return clean

    def init_state_f(self):
        self.depth_stack.clear()
        self.clean_depth_stack.clear()
        for _ in range(self.stack_frames):
            depth = None
            depth_fetch_failed = False
            try:
                depth = self.airgym.getScreenDepth(max_attempts=3)
            except Exception as e:
                depth_fetch_failed = True
                print(f"init_state_f: getScreenDepth failed after 3 attempts: {e}. Restarting game...")
                if self.game_handler:
                    self.game_handler.restart_game()
                    time.sleep(10)
                    self.airgym = AirLearningClient(
                        z=self.airgym.z,
                        ip=self.config.airsim_ip if self.config else None,
                        port=self.config.airsim_port if self.config else None
                    )
                    try:
                        depth = self.airgym.getScreenDepth(max_attempts=3)
                        depth_fetch_failed = False
                    except Exception as e2:
                        print(f"init_state_f: still failed after restart: {e2}. Using zero depth.")
                        depth = self._get_zero_depth()
                        depth_fetch_failed = True
                else:
                    print("init_state_f: no game handler available. Using zero depth.")
                    depth = self._get_zero_depth()
                    depth_fetch_failed = True

            if depth is None or depth.shape != self.depth_shape[-2:]:
                print("init_state_f: invalid depth shape, using zero depth.")
                depth = self._get_zero_depth()
                depth_fetch_failed = True
            clean_depth = np.asarray(depth, dtype=np.float32) if depth_fetch_failed else self._get_latest_clean_depth(depth)
            self.depth_stack.append(depth)
            self.clean_depth_stack.append(clean_depth)
            time.sleep(0.03)
        return self.get_obs()

    def state(self):
        """
        更新并获取当前的辅助状态信息 (inform)。
        (注意：此方法仅返回 inform 向量，不处理图像堆叠，图像堆叠在 step 方法中维护)。
        
        Returns:
            np.array: inform 向量 [相对距离xyz, 高度, 前向速度, z速度, 偏航角速度, 俯仰角, 横滚角, 偏航角, 朝向目标角度]
        """
        drone_pos = self.airgym.drone_pos()
        now = drone_pos[:2]
        altitude = -drone_pos[2]  # NED coordinate system, negative z is altitude
        
        # 获取完整姿态角: [pitch, roll, yaw]
        pitch, roll, yaw = self.airgym.get_ryp()
        
        # 新的状态向量组成
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)  # [x, y]
        self.relative_z_distance = float(self.goal[2] - drone_pos[2])
        forward_speed = self.airgym.get_forward_speed()  # 前向速度
        z_velocity = self.airgym.get_z_velocity()  # z轴速度
        yaw_rate = self.airgym.get_yaw_rate()  # 偏航角速度
        
        # 为了向后兼容，仍保留这些属性
        self.velocity = np.array([forward_speed, z_velocity, yaw_rate])  # 用新的速度信息
        self.speed = forward_speed  # 前向速度作为主要速度指标
        
        # 组合新的状态向量: [相对距离xyz(3), 高度(1), 前向速度(1), z速度(1), 偏航角速度(1), 俯仰角(1), 横滚角(1), 偏航角(1), 朝向目标角度(1)]
        inform = np.concatenate((
            self.relative_position,  # [x_dist, y_dist]
            [self.relative_z_distance],  # [z_dist]
            [altitude],              # [altitude]
            [forward_speed],         # 前向速度
            [z_velocity],           # z轴速度
            [yaw_rate],             # 偏航角速度
            [pitch],                # [pitch] 俯仰角
            [roll],                 # [roll] 横滚角
            [yaw],                  # [yaw] 偏航角
            self.r_yaw              # [relative_angle_to_target]
        ))
        
        return inform

    def normalize_base_state(self, inform):
        """
        将base state归一化到[0, 1]范围
        基于当前环境的实际参数（从 config 和 game_config_handler 获取）
        
        Args:
            inform: 11维状态向量 [rel_x, rel_y, rel_z, altitude, fwd_speed, z_vel, yaw_rate, pitch, roll, yaw, angle_to_goal]
        
        Returns:
            归一化后的11维向量，每个值在[0, 1]范围内
        """
        return inform

    def _compute_lidar_scan_log_penalty(self, scan_distance):
        """
        NavRL-like safety shaping on lidar beams:
        use all beams (not only nearest one), with mean(log(distance)) style.
        Here we anchor to safe distance so penalty <= 0 and equals 0 when all beams >= safe.
        """
        if self.lidar_log_penalty_weight <= 0.0:
            return 0.0
        if scan_distance is None:
            return 0.0

        d = np.asarray(scan_distance, dtype=np.float32)
        if d.size == 0:
            return 0.0
        d = d[np.isfinite(d)]
        if d.size == 0:
            return 0.0

        safe_dist = max(self.lidar_safe_distance_m, self.lidar_penalty_eps)
        # Clamp above safe distance: far beams contribute 0 risk.
        d_clip = np.clip(d, self.lidar_penalty_eps, safe_dist)
        mean_log_gap = float(np.mean(np.log(d_clip) - math.log(safe_dist)))
        penalty = self.lidar_log_penalty_weight * mean_log_gap
        return float(max(penalty, self.lidar_log_penalty_min))

    def _update_lidar_obstacle_distance(self):
        try:
            scan_distance, _ = self.airgym.get_lidar_scan_grid(
                lidar_name=self.reward_lidar_name,
                horizontal_bins=self.lidar_h_bins,
                vertical_bins=self.lidar_v_bins,
                vfov_min_deg=self.lidar_vfov_min_deg,
                vfov_max_deg=self.lidar_vfov_max_deg,
                max_range_m=self.lidar_distance_cap_m,
                max_attempts=self.lidar_query_max_attempts,
                retry_sleep=self.lidar_query_retry_sleep,
            )
        except Exception:
            self.last_lidar_obstacle_penalty = self._compute_lidar_scan_log_penalty(
                self.last_lidar_scan_distance
            )
            return False

        scan_distance = np.asarray(scan_distance, dtype=np.float32)
        if scan_distance.ndim != 2 or scan_distance.size == 0:
            self.last_lidar_obstacle_penalty = self._compute_lidar_scan_log_penalty(
                self.last_lidar_scan_distance
            )
            return False

        self.last_lidar_scan_distance = scan_distance.copy()

        self.last_lidar_obstacle_penalty = self._compute_lidar_scan_log_penalty(scan_distance)
        return True


    def computeReward(self, now, action, velocity_after=None):
        """
        计算每一步的奖励。
        
        奖励函数组成：
        1. reward_vel：NavRL风格速度投影奖励（可为负值）。
        2. distance_penalty：到目标点距离惩罚 (-goal_dist * 0.03)。
        3. smooth_penalty：NavRL风格速度平滑惩罚 ||v_t - v_{t-1}||。
        4. curvature_penalty：轨迹离散曲率平方惩罚 (r_curv = -alpha * kappa^2)。
        5. step_penalty：每步惩罚（沿用你的配置）。
        6. step_count_penalty：步数惩罚，当前步数 × 0.05。
        
        Args:
            now (np.array): 当前位置·
            action: 当前动作
            velocity_after: 动作执行后的速度 (vx, vy, speed)
            
        Returns:
            float: 计算出的奖励值
        """

        # NavRL-style velocity reward (r_vel): projection of velocity on goal direction.
        goal_vec = np.array([self.goal[0] - now[0], self.goal[1] - now[1]], dtype=np.float32)
        goal_dist = float(np.linalg.norm(goal_vec))
        distance_penalty = -goal_dist * 0.02

        if goal_dist > 1e-6:
            goal_dir = goal_vec / goal_dist
        else:
            goal_dir = np.zeros(2, dtype=np.float32)

        if velocity_after is not None:
            vel_xy = np.asarray(velocity_after[:2], dtype=np.float32)
            reward_vel = float(np.dot(vel_xy, goal_dir))
        else:
            now_pos = self.airgym.drone_pos()[:2]
            r_yaw = self.airgym.goal_direction(self.goal, now_pos)
            reward_vel = float(self.speed * math.cos(r_yaw))

        # Match NavRL base term: reward_vel 
        r = 2 * reward_vel 

        # NavRL-style smoothness penalty: ||v_t - v_{t-1}||
        if velocity_after is not None:
            curr_v = np.asarray(velocity_after, dtype=np.float32)
            prev_v = np.asarray(self.prev_velocity, dtype=np.float32)
            smooth_penalty = float(np.linalg.norm(curr_v - prev_v))
        else:
            smooth_penalty = 0.0
        smooth_penalty_weight = 0.05

        # Curvature penalty with speed gating and angle deadzone.
        curvature_penalty = 0.0
        if velocity_after is not None:
             v_xy_before = float(self.prev_velocity[2]) # speed is at index 2
             v_xy_after = float(velocity_after[2])
             
             dot_product = np.dot(velocity_after[:2], self.prev_velocity[:2])
             cos_theta = dot_product / (v_xy_before * v_xy_after + 1e-6)
             cos_theta = np.clip(cos_theta, -1.0, 1.0)
             angle_change = float(np.arccos(cos_theta))

             curvature_weight = 50.0
             curvature_penalty = curvature_weight * (angle_change ** 2) 
             curvature_penalty = float(np.clip(curvature_penalty, 0.0, 1.0))

        step_penalty = self.stepN * 0.005
        # step_penalty = 0.2

        # Stagnation penalty: penalize when total displacement in recent N steps is too small
        stagnation_penalty = 0.0
        if self.use_stagnation_penalty and len(self.displacement_window) >= self.stagnation_window:
            total_displacement = float(sum(self.displacement_window))
            stagnation_penalty = max(0.0, self.stagnation_window_threshold - total_displacement) * self.stagnation_weight

        # Add penalties to reward
        r -= smooth_penalty * smooth_penalty_weight  + step_penalty 

        lidar_penalty = self._compute_lidar_scan_log_penalty(self.last_lidar_scan_distance)
        self.last_lidar_obstacle_penalty = float(lidar_penalty)
        r += 5 * float(lidar_penalty)
        # print(f"Reward components: r_vel={reward_vel:.3f}, distance_penalty={distance_penalty:.3f}, smooth_penalty={smooth_penalty:.3f}, curvature_penalty={curvature_penalty:.3f}, step_penalty={step_penalty:.3f}, stagnation_penalty={stagnation_penalty:.3f}, lidar_penalty={lidar_penalty:.3f}, total_reward={r:.3f}")

         

        return r


    #根据控制飞机飞的方式不同，动作空间和step会有不同
    #根据不同输入，obs space会有不同
    def step(self, action):
        """
        环境步进函数。执行动作，返回观测、奖励、完成标志等。
        
        1. 执行动作 (连续或离散)。
        2. 获取碰撞状态。
        3. 获取新的观测图像和状态信息 (inform)。
        4. 堆叠最新的图像帧，更新观测栈。
        5. 判断是否结束 (done)：
            - 到达目标 (distance < success_distance) -> 奖励 +20, success=True
            - 发生碰撞 (collided) -> 奖励 -20, success=False
            - 否则计算常规奖励 computeReward。
        
        Args:
            action: 动作
            
        Returns:
            tuple: (state, reward, done, info)
        """

        self.stepN += 1
        if self.check_ue4_status():
            state = self.get_obs()
            self.success = False
            info = {
                "has_collided": False,
                "altitude_violation": False,
                "is_success": False,
                "ue4_restarted": True,
                "lidar_obstacle_log_penalty": float(self.last_lidar_obstacle_penalty),
            }
            return state, 0.0, True, False, info
        
        self.airgym.client.simPause(False)
        if (settings.control_mode == "moveByVelocity"):

            collided = self.airgym.take_continious_action(action, duration=self.action_duration)

        elif (settings.control_mode == "Continuous"):
             # 适配连续动作：如果有多余的维度（batch维），去除它
            if np.ndim(action) > 1:
                action = action[0]
            collided = self.airgym.take_continuous_action_3d(action, duration=self.action_duration)

        else:
            collided = self.airgym.take_discrete_action(action)

        self.airgym.client.simPause(True)

        # Update stacks: max 3 attempts in getScreenDepth; restart immediately on failure.
        depth_img = None
        depth_fetch_failed = False
        try:
            depth_img = self.airgym.getScreenDepth(max_attempts=3)
        except Exception as e:
            depth_fetch_failed = True
            print(f"Failed to fetch images after 3 attempts: {e}. Restarting game...")
            if self.game_handler:
                self.game_handler.restart_game()
                time.sleep(10)  # 等待游戏重启
                # 重新初始化客户端
                self.airgym = AirLearningClient(
                    z=self.airgym.z,
                    ip=self.config.airsim_ip if self.config else None,
                    port=self.config.airsim_port if self.config else None
                )
                # 重启后再尝试一次（同样最多3次）
                try:
                    depth_img = self.airgym.getScreenDepth(max_attempts=3)
                    depth_fetch_failed = False
                except Exception as e2:
                    print(f"Still failed after restart: {e2}. Using zero arrays.")
                    depth_img = self._get_zero_depth()
                    depth_fetch_failed = True
            else:
                print("No game handler available. Using zero arrays.")
                depth_img = self._get_zero_depth()
                depth_fetch_failed = True
        
        if depth_img is None or depth_img.shape != self.depth_shape[-2:]:
            depth_img = self._get_zero_depth()
            depth_fetch_failed = True
        clean_depth_img = np.asarray(depth_img, dtype=np.float32) if depth_fetch_failed else self._get_latest_clean_depth(depth_img)
        self.depth_stack.append(depth_img)
        self.clean_depth_stack.append(clean_depth_img)
        self._update_lidar_obstacle_distance()

        # Get observation
        state = self.get_obs()
        now = self.airgym.drone_pos()

        distance = np.sqrt(np.power((self.goal[0] - now[0]), 2)
                           +np.power((self.goal[1] - now[1]), 2)
                           )
        # 转换高度
        current_altitude = -now[2]  # NED坐标系，z负值表示高度
        
        # 检查高度越界，将其作为碰撞处理
        altitude_violation = False
        if current_altitude > self.max_altitude:
            collided = True
            altitude_violation = True

        success_altitude_min = 0.5
        success_altitude_max = 2.0
        success_altitude_ok = success_altitude_min <= current_altitude <= success_altitude_max

        if distance < settings.success_distance_to_goal and success_altitude_ok:
            self.success_count += 1
            done = True
            self.print_msg_of_inspiration()
            self.success = True
            msgs.success = True
            reward = 20.0

        elif collided == True:
            done = True
            reward = -20.0
            self.success = False
            # if altitude_violation:
            #     print(f"[终止] 高度越界导致episode终止")

        elif self.stepN >= self.config.episode_length:
            done = True
            reward = -30.0
            self.success = False
            
        else:
            # 获取当前速度用于奖励计算 (vx, vy, speed)
            velocity_after = self.airgym.drone_velocity()
            # 计算基础奖励
            reward = self.computeReward(
                now,
                action,
                velocity_after=velocity_after,
            )
            done = False
            self.success = False
            
            # 检查高度违规（施加惩罚但不终止）
            if current_altitude < self.min_altitude_penalty:
                reward -= self.altitude_penalty_value  # 低于惩罚高度阈值，给予固定惩罚
                # print(f"[最低惩罚高度违规] 当前高度: {current_altitude:.2f}m，惩罚高度: {self.min_altitude_penalty}m，惩罚: -{self.altitude_penalty_value}")
            if current_altitude > self.max_altitude_penalty:
                reward -= self.altitude_penalty_value  # 高于惩罚高度阈值，给予固定惩罚
                # print(f"[最高惩罚高度违规] 当前高度: {current_altitude:.2f}m，惩罚高度: {self.max_altitude_penalty}m，惩罚: -{self.altitude_penalty_value}")
        
        # Accumulate reward for episode
        self.episode_reward += reward

        self.prev_state = state
        self.prev_action = action.copy() if isinstance(action, np.ndarray) else action
        self.prev_velocity = self.airgym.drone_velocity()

        # Update displacement window for stagnation penalty
        if self.prev_pos_xy is not None:
            displacement = float(np.linalg.norm(np.array([now[0] - self.prev_pos_xy[0], now[1] - self.prev_pos_xy[1]], dtype=np.float32)))
            self.displacement_window.append(displacement)
        self.prev_pos_xy = np.array(now[:2], dtype=np.float32)

        if (done):
            if self.success:
                self.success_deque.append(1)
            else:
                self.success_deque.append(0)
            self.on_episode_end()

        info = {
            "has_collided": bool(collided),
            "altitude_violation": bool(altitude_violation),
            "is_success": bool(self.success),
            "lidar_obstacle_log_penalty": float(self.last_lidar_obstacle_penalty),
        }
        return state, reward, done, False, info


    def on_episode_end(self):
        
        pass


    def on_episode_start(self):
        self.stepN = 0
        self.episodeN += 1
        self.episode_reward = 0  # Reset reward accumulator for new episode
        self.last_lidar_obstacle_penalty = 0.0
        self.last_lidar_scan_distance = np.full(
            (self.lidar_h_bins, self.lidar_v_bins),
            self.lidar_distance_cap_m,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        """
        重置环境。
        
        1. 根据成功率动态调整难度级别 (Level 0 -> 1 -> 2)。
        2. 清除渲染器几何体。
        3. 自行取消暂停仿真以便进行重置操作。
        4. 随机化环境参数 (randomize_env)。
        5. 重置 Unreal 环境 (resetUnreal)。
        6. 重置 AirSim 状态并起飞 (AirSim_reset)。
        7. 停止仿真以获取初始观测。
        
        Args:
            seed (int, optional): 随机种子。
            options (dict, optional): 重置选项。
        
        Returns:
            tuple: (初始状态 [stacked_images, init_inform], info dict)
        """
        # 设置种子
        if seed is not None:
            self.seed(seed)

        self.check_ue4_status()
        
        # 取消暂停，确保重置和起飞命令可以执行
        self.airgym.client.simPause(False)

        # 课程学习等级升级（仅在启用课程学习时）
        if self.use_curriculum and len(self.success_deque)>0:
            succes_rate=sum(self.success_deque) / len(self.success_deque)
            if succes_rate>0.5 and self.level==0 and self.success_count>300:
                self.level=1
                self.game_config_handler=GameConfigHandler(range_dic_name="settings.medium_range_dic")
            elif succes_rate > 0.6 and self.level == 1 and self.success_count>600:
                self.level = 2
                self.game_config_handler = GameConfigHandler(range_dic_name="settings.hard_range_dic")
            # elif succes_rate > 0.7 and self.level == 2 and self.success_count > 900:
            #     self.level = 3
            #     self.game_config_handler = GameConfigHandler(range_dic_name="settings.dynamic_obstacles_dic")
            

        print("--- Resetting Episode ---")
        
        # 1. 环境随机化
        env_changed = self.randomize_env()
        
        # 2. 只有在环境发生变化时才强制重建环境
        if env_changed:
            try:
                start_time = time.time()
                self.airgym.unreal_reset()
                print(f"Unreal environment reset done. Time: {time.time() - start_time:.2f}s")
            except Exception as e:
                print(f"Unreal reset failed: {e}")
            
            # 环境发了改变，必定存在连接断开，强制抛弃重建
            reconnect_start = time.time()
            self.airgym.AirSim_reset()
            print(f"AirSim drone reset done (Hard Reconnect). Time: {time.time() - reconnect_start:.2f}s")
        else:
            # 环境没有改变时，可以采用软重置 (复用客户端以极速完成)
            try:
                soft_start = time.time()
                self.airgym.client.reset()
                self.airgym.client.enableApiControl(True)
                self.airgym.client.armDisarm(True)
                self.airgym.client.moveToZAsync(float(self.airgym.z), 1.0)
                time.sleep(1.0)
                self.airgym.client.hoverAsync().join()
                print(f"AirSim drone soft reset done. Time: {time.time() - soft_start:.2f}s")
            except Exception:
                # 软复位失败兜底保底
                fallback_start = time.time()
                self.airgym.AirSim_reset()
                print(f"AirSim drone fallback hard reset done. Time: {time.time() - fallback_start:.2f}s")

        # 4. 检查起飞结果并自旋重试 (如果失败)
        # Check takeoff status and retry if failed
        def _get_takeoff_status(max_attempts=3):
            for attempt in range(max_attempts):
                try:
                    return self.airgym.drone_pos(), self.airgym.client.simGetCollisionInfo()
                except Exception as e:
                    print(f"Get takeoff status RPC failed (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1:
                        self.check_ue4_status(force_restart=True, reason=f"takeoff_status_rpc_failed_attempt_{attempt + 1}")
                        self.airgym.AirSim_reset()
                    else:
                        raise

        def _ensure_takeoff_ok():
            start_time = time.time()
            now = None
            collision_info = None
            try:
                now, collision_info = _get_takeoff_status()
            except Exception as e:
                # If getting status fails, try to reconnect heavily
                print(f"Status check failed after reset: {e}. Retrying with hard reset...")
                self.check_ue4_status(force_restart=True, reason="takeoff_status_query_failed")
                self.airgym.AirSim_reset()
                now, collision_info = _get_takeoff_status()

            max_takeoff_retries = 5
            retry_count = 0
            while ((-now[2]) < 0.5 or collision_info.has_collided) and retry_count < max_takeoff_retries:
                print(f"Takeoff attempt {retry_count+1} failed! Height: {-now[2]:.2f}, Collided: {collision_info.has_collided}")

                # 多次失败直接重启游戏（最简单兜底）
                if retry_count == max_takeoff_retries - 1:
                    print("Takeoff failed multiple times. Restarting game...")
                    self.check_ue4_status(force_restart=True, reason="takeoff_failed_max_retry")
                    self.airgym.AirSim_reset()
                    now, collision_info = _get_takeoff_status()
                    retry_count += 1
                    break

                # 失败次数尚未达到阈值，先尝试软重置
                self.game_config_handler.sample('Seed', change_counter=-(retry_count + 1), base_seed=self.base_seed)
                self.airgym.unreal_reset()
                self.airgym.AirSim_reset()

                now, collision_info = _get_takeoff_status()
                retry_count += 1

            print(f"Takeoff check completed. Total time: {time.time() - start_time:.2f}s, Retries: {retry_count}")
            return now

        now = _ensure_takeoff_ok()

        if self.enable_takeoff_obstacle_check:
            # 起飞后激光雷达避障检查：若最近障碍距离 < 阈值，重置环境并重新起飞。
            obstacle_retry_count = 0
            while True:
                try:
                    too_close, min_distance, per_direction = self.airgym.is_obstacle_too_close_lidar(
                        threshold_m=self.takeoff_obstacle_threshold_m,
                        lidar_name=self.takeoff_lidar_name,
                        max_attempts=3,
                        retry_sleep=0.1,
                    )
                except Exception as e:
                    print(f"Takeoff obstacle check failed, skip reset-by-obstacle this episode: {e}")
                    break

                if not too_close:
                    break

                print(
                    f"Takeoff obstacle too close: min={min_distance:.2f}m < {self.takeoff_obstacle_threshold_m:.2f}m, "
                    f"per_direction={per_direction}"
                )

                if obstacle_retry_count >= self.takeoff_obstacle_reset_retries:
                    print("Reached max obstacle reset retries after takeoff, continue with current episode.")
                    break

                obstacle_retry_count += 1
                try:
                    # 近障后优先刷新环境，而不是只重置无人机。
                    self.game_config_handler.sample('Seed', change_counter=-(100 + obstacle_retry_count), base_seed=self.base_seed)
                    self.airgym.unreal_reset()
                except Exception as e:
                    print(f"Takeoff obstacle env refresh failed, fallback to drone reset only: {e}")

                self.airgym.AirSim_reset()
                now = _ensure_takeoff_ok()
        else:
            print("Takeoff obstacle check disabled.")

        # 确保处于目标高度
        if abs(now[2] - self.airgym.z) > 0.1:
            self.airgym.client.moveToZAsync(self.airgym.z, 3).join()
            now = self.airgym.drone_pos()

        # 5. 暂停仿真以同步获取初始状态
        self.airgym.client.simPause(True)
        
        start_time = time.time()
        self.on_episode_start()
        state = self.init_state_f()
        self._update_lidar_obstacle_distance()
        print(f"Initial state acquisition time: {time.time() - start_time:.2f}s")
        
        # # 打印当前 episode 的 JSON 配置
        # print("="*60)
        # print(f"Episode {self.episodeN} Configuration (JSON):")
        # import json
        # print(json.dumps(self.game_config_handler.cur_game_config.get(), indent=2))
        # print("="*60)


        self.prev_state = state
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32) if hasattr(self.action_space, 'shape') and self.action_space.shape else 0
        self.prev_velocity = self.airgym.drone_velocity()
        self.prev_pos_xy = None
        self.displacement_window.clear()
        
        # 返回 (obs, info)
        info = {}  # 可以添加额外信息
        return state, info

    def randomize_env(self):
        """
        根据 episode 计数和配置频率，随机化环境参数。
        如果有参数需要随机化，则调用 sampleGameConfig 进行采样，并更新目标点位置。
        Returns:
            bool: 是否进行了随机化 (True/False)
        """
        vars_to_randomize = []
        for k, v in settings.environment_change_frequency.items():
            if (self.episodeN+1) %  v == 0:
                vars_to_randomize.append(k)

        if (len(vars_to_randomize) > 0):
            print(f"Randomizing environment vars: {vars_to_randomize}")
            self.sampleGameConfig(*vars_to_randomize)
            self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
            return True
            
        return False


    def updateJson(self, *args):
        self.game_config_handler.update_json(*args)

    def getItemCurGameConfig(self, key):
        return self.game_config_handler.get_cur_item(key)

    def setRangeGameConfig(self, *args):
        self.game_config_handler.set_range(*args)

    def getRangeGameConfig(self, key):
        return self.game_config_handler.get_range(key)

    def sampleGameConfig(self, *arg):
        # 使用确定性采样，传入当前变化计数器和基础种子
        # 每次调用 sampleGameConfig 表示一次环境变化
        self.change_counter += 1
        self.game_config_handler.sample(*arg, change_counter=self.change_counter, base_seed=self.base_seed)

    def close(self):
        """Close the environment and clean up resources."""
        return None
