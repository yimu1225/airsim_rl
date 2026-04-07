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


class AirSimEnvGradientReward(gym.Env):
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
    
        STATE_DEPTH_H, STATE_DEPTH_W = 128,128

        self.stack_frames = stack_frames
        self.episode_reward = 0

        self.base_dim = 10
        self.depth_shape = (self.stack_frames, STATE_DEPTH_H, STATE_DEPTH_W)

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=np.float32(0), high=np.float32(255), shape=self.depth_shape, dtype=np.float32),
            "base": spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_dim,), dtype=np.float32)
        })

        self.depth_stack = collections.deque(maxlen=self.stack_frames)

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
        algorithm_name = getattr(config, "algorithm_name", "")
        self.use_curriculum = algorithm_name.startswith("CL-")
        self.curriculum_start_level = getattr(config, "curriculum_start_level", 0)
        self.non_curriculum_level = getattr(config, "non_curriculum_level", 3)
        
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
        sample_vars = ["Seed", "ArenaSize", "NumberOfObjects", "End", "Walls1"]
        # Always sample dynamic object count to avoid carrying stale values from previous runs.
        # For non-dynamic levels, the configured range is [0], so this deterministically clears dynamics.
        sample_vars.append("NumberOfDynamicObjects")
        self.game_config_handler.sample(*sample_vars, change_counter=0, base_seed=self.base_seed)
        
        # 现在启动 UE4，它会读取上面写入的 JSON
        disable_game_restart = getattr(config, "disable_game_restart", False) if config is not None else False
        self.game_handler = None if disable_game_restart else GameHandler()
        if self.game_handler is not None:
            self.game_handler.restart_game()

        print("Scene initialization complete")

        # 无人机 API，传入config中的ip和port参数
        client_ip = config.airsim_ip if config is not None else None
        client_port = config.airsim_port if config is not None else None
        self.airgym = AirLearningClient(z=takeoff_height, ip=client_ip, port=client_port)

        # 动作持续时间 / 时钟缩放参数
        
        self.clock_speed_factor = config.clock_speed_factor
        self.raw_action_duration = config.action_duration
        
        self.action_duration = self.raw_action_duration / self.clock_speed_factor

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
        self.ue4_rpc_fail_threshold = getattr(config, "ue4_rpc_fail_threshold", 2) 
        self.ue4_health_check_interval = getattr(config, "ue4_health_check_interval", 1.0) 
        self.ue4_window_check_interval = getattr(config, "ue4_window_check_interval", 5.0) 
        self.ue4_process_check_interval = max(3.0, self.ue4_health_check_interval * 3.0)
        self._last_ue4_health_check_ts = 0.0
        self._last_process_check_ts = 0.0
        self._last_window_check_ts = 0.0
        self._cached_process_alive = True
        self._cached_window_alive = None

        self.takeoff_obstacle_threshold_m = float(getattr(config, "takeoff_obstacle_threshold_m", 3.0))
        self.takeoff_obstacle_reset_retries = max(0, int(getattr(config, "takeoff_obstacle_reset_retries", 3)))
        self.takeoff_lidar_name = str(getattr(config, "takeoff_lidar_name", "LidarSensor1")).strip()
        
        # Gradient-map reward hyperparameters (从config读取)
        # 设计目标：
        # - 不依赖障碍物坐标（仅使用局部深度观测）
        # - 在课程学习中保持尺度一致（距离归一化）
        self.grad_goal_weight = float(config.grad_goal_weight)
        self.grad_heading_weight = float(config.grad_heading_weight)
        self.grad_obstacle_weight = float(config.grad_obstacle_weight)
        self.grad_altitude_weight = float(config.grad_altitude_weight)
        self.grad_progress_weight = float(config.grad_progress_weight)
        self.grad_step_penalty = float(config.grad_step_penalty)
        self.grad_reward_clip = float(config.grad_reward_clip)
        self.grad_cost_clip = float(config.grad_cost_clip)
        self.grad_shaping_gamma = float(config.grad_shaping_gamma)

        # 仅依赖深度图的局部障碍风险建模
        self.grad_safe_depth_m = float(config.grad_safe_depth_m)
        self.grad_depth_floor_m = float(config.grad_depth_floor_m)
        self.grad_depth_max_m = float(config.grad_depth_max_m)
        self.grad_depth_percentile = float(config.grad_depth_percentile)
        self.grad_obstacle_decay_m = float(config.grad_obstacle_decay_m)
        self.grad_obstacle_balance_weight = float(config.grad_obstacle_balance_weight)

        # 高度归一化带宽
        self.grad_altitude_band_m = float(config.grad_altitude_band_m)

        # 跨课程学习尺度归一化（避免地图变大后奖励量级漂移）
        self.grad_distance_scale_min = float(config.grad_distance_scale_min)
        self.grad_distance_arena_ratio = float(config.grad_distance_arena_ratio)

        # 平滑控制
        self.grad_smoothness_weight = float(config.grad_smoothness_weight)
        self.grad_smoothness_deadzone = float(config.grad_smoothness_deadzone)

        self.grad_success_reward = float(config.grad_success_reward)
        self.grad_collision_reward = float(config.grad_collision_reward)
        self.grad_timeout_reward = float(config.grad_timeout_reward)

        # 轨迹与势能缓存（梯度奖励所需）
        self.prev_position_xy = None
        self.prev_potential = None
        self.last_gradient_terms = {}
        self.episode_arena_diag = 1.0
        self.episode_goal_distance0 = 1.0
        self.episode_goal_scale = 1.0

        # 保留历史动作/速度字段，兼容上层可能读取这些属性的代码
        if hasattr(self.action_space, 'shape') and self.action_space.shape:
            self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        else:
            self.prev_action = 0
        self.prev_velocity = np.zeros(3, dtype=np.float32)

    def _reconnect_airsim_client(self, reason=""):
        """
        Reconnect AirSim client with retries after UE recovery.
        Returns True on success, False on failure.
        """
        z = -0.9
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

    def init_state_f(self):
        self.depth_stack.clear()
        for _ in range(self.stack_frames):
            depth = None
            try:
                depth = self.airgym.getScreenDepth(max_attempts=3)
            except Exception as e:
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
                    except Exception as e2:
                        print(f"init_state_f: still failed after restart: {e2}. Using zero depth.")
                        depth = np.zeros((128, 128), dtype=np.float32)
                else:
                    print("init_state_f: no game handler available. Using zero depth.")
                    depth = np.zeros((128, 128), dtype=np.float32)

            if depth is None or depth.shape != (128, 128):
                print("init_state_f: invalid depth shape, using zero depth.")
                depth = np.zeros((128, 128), dtype=np.float32)
            self.depth_stack.append(depth)
            time.sleep(0.03)
        return self.get_obs()

    def state(self):
        """
        更新并获取当前的辅助状态信息 (inform)。
        (注意：此方法仅返回 inform 向量，不处理图像堆叠，图像堆叠在 step 方法中维护)。
        
        Returns:
            np.array: inform 向量 [相对距离xy, 高度, 前向速度, z速度, 偏航角速度, 俯仰角, 横滚角, 偏航角, 朝向目标角度]
        """
        drone_pos = self.airgym.drone_pos()
        now = drone_pos[:2]
        altitude = -drone_pos[2]  # NED coordinate system, negative z is altitude
        
        # 获取完整姿态角: [pitch, roll, yaw]
        pitch, roll, yaw = self.airgym.get_ryp()
        
        # 新的状态向量组成
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)  # [x, y]
        forward_speed = self.airgym.get_forward_speed()  # 前向速度
        z_velocity = self.airgym.get_z_velocity()  # z轴速度
        yaw_rate = self.airgym.get_yaw_rate()  # 偏航角速度
        
        # 为了向后兼容，仍保留这些属性
        self.velocity = np.array([forward_speed, z_velocity, yaw_rate])  # 用新的速度信息
        self.speed = forward_speed  # 前向速度作为主要速度指标
        
        # 组合新的状态向量: [相对距离xy(2), 高度(1), 前向速度(1), z速度(1), 偏航角速度(1), 俯仰角(1), 横滚角(1), 偏航角(1), 朝向目标角度(1)]
        inform = np.concatenate((
            self.relative_position,  # [x_dist, y_dist] 
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
            inform: 10维状态向量 [rel_x, rel_y, altitude, fwd_speed, z_vel, yaw_rate, pitch, roll, yaw, angle_to_goal]
        
        Returns:
            归一化后的10维向量，每个值在[0, 1]范围内
        """
        return inform


    def _extract_depth_frame(self, depth_img):
        """
        统一将深度图提取为 (H, W) 单帧浮点数组。
        """
        if depth_img is None:
            return np.zeros((128, 128), dtype=np.float32)
        frame = np.asarray(depth_img, dtype=np.float32)
        if frame.ndim == 3:
            frame = frame[0]
        return frame

    def _get_arena_diagonal_xy(self):
        """
        从当前关卡配置中估计场地XY对角线长度（米），用于奖励归一化。
        """
       
        arena = self.game_config_handler.get_cur_item("ArenaSize")
    
        arena_x = max(abs(float(arena[0])), 1.0)
        arena_y = max(abs(float(arena[1])), 1.0)
        return max(math.sqrt(arena_x ** 2 + arena_y ** 2), 1.0)
     

    def _refresh_episode_scales(self, now_xy):
        """
        结合初始目标距离与场地尺度，更新本回合归一化尺度。
        """
        goal_vec = np.array([self.goal[0] - now_xy[0], self.goal[1] - now_xy[1]], dtype=np.float32)
        self.episode_goal_distance0 = max(float(np.linalg.norm(goal_vec)), 1e-3)
        self.episode_arena_diag = self._get_arena_diagonal_xy()

        arena_based_min = self.grad_distance_arena_ratio * self.episode_arena_diag
        self.episode_goal_scale = max(
            self.episode_goal_distance0,
            arena_based_min,
            self.grad_distance_scale_min,
            settings.success_distance_to_goal * 2.0,
            1e-3,
        )

    def _estimate_obstacle_profile(self, depth_img):
        """
        仅依赖前向深度图构建障碍风险，不使用障碍物坐标。
        """
        frame = self._extract_depth_frame(depth_img)
        h, w = frame.shape
        y0, y1 = int(h * 0.30), int(h * 0.82)
        x0, x1 = int(w * 0.12), int(w * 0.88)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return {
                "nearest_depth_m": self.grad_depth_max_m,
                "center_depth_m": self.grad_depth_max_m,
                "left_risk": 0.0,
                "right_risk": 0.0,
                "field_risk": 0.0,
                "balance_risk": 0.0,
                "obstacle_risk": 0.0,
            }

        roi = np.clip(roi, 0.0, 255.0)
        sectors = np.array_split(roi, 3, axis=1)
        sector_depth_m = []
        sector_risk = []
        risk_decay = max(self.grad_obstacle_decay_m, 1e-3)

        for sec in sectors:
            if sec.size == 0:
                depth_m = self.grad_depth_max_m
            else:
                q = float(np.percentile(sec, self.grad_depth_percentile))
                depth_m = (q / 255.0) * self.grad_depth_max_m
                depth_m = float(np.clip(depth_m, self.grad_depth_floor_m, self.grad_depth_max_m))
            sector_depth_m.append(depth_m)
            sector_risk.append(float(math.exp(-depth_m / risk_decay)))

        nearest_depth_m = float(min(sector_depth_m))
        center_depth_m = float(sector_depth_m[1]) if len(sector_depth_m) > 1 else nearest_depth_m
        left_risk = float(sector_risk[0])
        right_risk = float(sector_risk[-1])
        field_risk = float(np.mean(sector_risk))
        balance_risk = abs(right_risk - left_risk)

        clearance_risk = max(
            0.0,
            (self.grad_safe_depth_m - nearest_depth_m) / max(self.grad_safe_depth_m, 1e-6),
        )
        obstacle_risk = (
            0.65 * clearance_risk
            + 0.35 * field_risk
            + self.grad_obstacle_balance_weight * balance_risk
        )
        obstacle_risk = float(np.clip(obstacle_risk, 0.0, 1.5))

        return {
            "nearest_depth_m": nearest_depth_m,
            "center_depth_m": center_depth_m,
            "left_risk": left_risk,
            "right_risk": right_risk,
            "field_risk": field_risk,
            "balance_risk": balance_risk,
            "obstacle_risk": obstacle_risk,
        }

    def _compute_gradient_potential(self, now, depth_img):
        """
        构造观测驱动势能 Phi(s)：
        - 目标距离项：按本回合尺度归一化
        - 朝向项：偏航误差归一化
        - 障碍项：仅来自深度图局部统计
        - 高度项：偏离目标高度的归一化惩罚
        """
        goal_vec = np.array([self.goal[0] - now[0], self.goal[1] - now[1]], dtype=np.float32)
        distance_to_goal = float(np.linalg.norm(goal_vec))
        distance_to_goal_norm = min(distance_to_goal / max(self.episode_goal_scale, 1e-6), self.grad_cost_clip)

        r_yaw = getattr(self, "r_yaw", np.array([0.0], dtype=np.float32))
        r_yaw_val = float(np.asarray(r_yaw, dtype=np.float32).reshape(-1)[0]) if np.size(r_yaw) > 0 else 0.0
        heading_error = 0.5 * (1.0 - math.cos(r_yaw_val))
        heading_error = float(np.clip(heading_error, 0.0, 1.0))

        obstacle_profile = self._estimate_obstacle_profile(depth_img)
        obstacle_risk = float(obstacle_profile["obstacle_risk"])

        current_altitude = float(-now[2])  # NED坐标系下高度为 -z
        # 高度范围软惩罚：在[min_altitude_penalty, max_altitude_penalty]范围内无惩罚，超出后按偏离程度惩罚
        low_violation = max(0.0, self.min_altitude_penalty - current_altitude)
        high_violation = max(0.0, current_altitude - self.max_altitude_penalty)
        altitude_error = low_violation + high_violation
        altitude_error_norm = min(altitude_error / max(self.grad_altitude_band_m, 1e-6), self.grad_cost_clip)

        cost = (
            self.grad_goal_weight * distance_to_goal_norm
            + self.grad_heading_weight * heading_error
            + self.grad_obstacle_weight * obstacle_risk
            + self.grad_altitude_weight * altitude_error_norm
        )
        cost = float(np.clip(cost, 0.0, self.grad_cost_clip))
        potential = -cost

        terms = {
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_norm": distance_to_goal_norm,
            "heading_error": heading_error,
            "obstacle_risk": obstacle_risk,
            "altitude_error": altitude_error,
            "altitude_error_norm": altitude_error_norm,
            "cost": cost,
            "potential": potential,
            "goal_scale": self.episode_goal_scale,
            "goal_distance0": self.episode_goal_distance0,
            "arena_diag": self.episode_arena_diag,
            **obstacle_profile,
        }
        return potential, terms

    def _compute_smoothness_penalty(self, action):
        """
        动作变化惩罚：压制高频抖动，避免策略在局部噪声里震荡。
        """
        if not (hasattr(self.action_space, "shape") and self.action_space.shape):
            return 0.0
        if not isinstance(action, np.ndarray):
            return 0.0

        prev_action = self.prev_action
        if not isinstance(prev_action, np.ndarray):
            return 0.0

        action_span = np.asarray(self.action_space.high - self.action_space.low, dtype=np.float32)
        action_span = np.maximum(action_span, 1e-6)
        delta_norm = np.abs(action.astype(np.float32) - prev_action.astype(np.float32)) / action_span
        delta_excess = np.maximum(delta_norm - self.grad_smoothness_deadzone, 0.0)
        penalty = self.grad_smoothness_weight * float(np.mean(np.square(delta_excess)))
        return float(np.clip(penalty, 0.0, 1.0))

    def computeReward(self, now, action, velocity_after=None, depth_img=None, prev_xy=None):
        """
        梯度奖励主体：
        1. 代价下降量（核心）
        2. 时间惩罚（0.01 * stepN）
        3. 平滑惩罚
        """
        del velocity_after

        potential_now, terms = self._compute_gradient_potential(now, depth_img)
        if self.prev_potential is None:
            progress = 0.0
        else:
            progress = float(self.grad_shaping_gamma * potential_now - self.prev_potential)
        # print(f"computeReward: potential_now={potential_now:.4f}, progress={progress:.4f}")

        smoothness_penalty = self._compute_smoothness_penalty(action)

        if prev_xy is None:
            prev_xy = self.prev_position_xy
        move_distance = 0.0
        if prev_xy is not None:
            move_distance = float(np.linalg.norm(np.array([now[0] - prev_xy[0], now[1] - prev_xy[1]], dtype=np.float32)))

        time_penalty = 0.01 * float(self.stepN)

        reward = (
            self.grad_progress_weight * progress
            - time_penalty
            - smoothness_penalty
        )
        reward = float(np.clip(reward, -self.grad_reward_clip, self.grad_reward_clip))

        self.prev_potential = potential_now
        self.last_gradient_terms = {
            **terms,
            "progress": progress,
            "move_distance": move_distance,
            "time_penalty": time_penalty,
            "smoothness_penalty": smoothness_penalty,
            "reward": reward,
        }
        # # 打印奖励函数的各项
        # print("奖励函数各项:")
        # print(f"  目标距离: {terms['distance_to_goal']:.4f} (归一化: {terms['distance_to_goal_norm']:.4f})")
        # print(f"  朝向误差: {terms['heading_error']:.4f}")
        # print(f"  障碍风险: {terms['obstacle_risk']:.4f}")
        # print(f"  高度误差: {terms['altitude_error']:.4f} (归一化: {terms['altitude_error_norm']:.4f})")
        # print(f"  总代价: {terms['cost']:.4f}")
        # print(f"  势能: {terms['potential']:.4f}")
        # print(f"  进度奖励: {progress:.4f}")
        # print(f"  移动距离: {move_distance:.4f}")
        # print(f"  平滑惩罚: {smoothness_penalty:.4f}")
        # print(f"  平滑惩罚: {smoothness_penalty:.4f}")
        # print(f"  每步惩罚: {self.grad_step_penalty:.4f}")
        # # 打印 reward 计算的各项
        # progress_reward = self.grad_progress_weight * progress
        # step_penalty = self.grad_step_penalty
        # smoothness_penalty_total = smoothness_penalty
        # print(f"  奖励计算:")
        # print(f"    进度奖励项: {self.grad_progress_weight:.4f} * {progress:.4f} = {progress_reward:.4f}")
        # print(f"    每步惩罚项: -{step_penalty:.4f}")
        # print(f"    平滑惩罚项: -{smoothness_penalty_total:.4f}")
        # print(f"    奖励总和: {progress_reward:.4f} - {step_penalty:.4f} - {smoothness_penalty_total:.4f} = {reward:.4f}")
        # print()
        return reward


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
        self.last_gradient_terms = {}

        if self.check_ue4_status():
            state = self.get_obs()
            self.success = False
            info = {
                "has_collided": False,
                "altitude_violation": False,
                "is_success": False,
                "ue4_restarted": True
            }
            return state, 0.0, True, False, info
        
        prev_now = self.airgym.drone_pos()

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
        try:
            depth_img = self.airgym.getScreenDepth(max_attempts=3)
        except Exception as e:
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
                except Exception as e2:
                    print(f"Still failed after restart: {e2}. Using zero arrays.")
                    depth_img = np.zeros((128, 128), dtype=np.float32)
            else:
                print("No game handler available. Using zero arrays.")
                depth_img = np.zeros((128, 128), dtype=np.float32)
        
        self.depth_stack.append(depth_img)

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
            # print(f"[最大高度越界] 当前高度: {current_altitude:.2f}m，最大高度: {self.max_altitude}m")

        success_altitude_max = 2.0
        success_altitude_ok = current_altitude <= success_altitude_max

        if distance < settings.success_distance_to_goal and success_altitude_ok:
            self.success_count += 1
            done = True
            self.print_msg_of_inspiration()
            self.success = True
            msgs.success = True
            reward = self.grad_success_reward

        elif collided == True:
            done = True
            reward = self.grad_collision_reward
            self.success = False


        elif self.stepN >= self.config.episode_length:
            done = True
            reward = self.grad_timeout_reward
            self.success = False
            
        else:
            reward = self.computeReward(
                now=now,
                action=action,
                velocity_after=self.airgym.drone_velocity(),
                depth_img=depth_img,
                prev_xy=prev_now[:2]
            )
            done = False
            self.success = False

        # Accumulate reward for episode
        self.episode_reward += reward

        self.prev_state = state
        self.prev_position_xy = np.array(now[:2], dtype=np.float32)
        self.prev_action = action.copy() if isinstance(action, np.ndarray) else action
        self.prev_velocity = self.airgym.drone_velocity()

        if (done):
            if self.success:
                self.success_deque.append(1)
            else:
                self.success_deque.append(0)
            self.on_episode_end()

        info = {
            "has_collided": bool(collided),
            "altitude_violation": bool(altitude_violation),
            "is_success": bool(self.success)
        }
        if self.last_gradient_terms:
            info["gradient_terms"] = self.last_gradient_terms
        return state, reward, done, False, info


    def on_episode_end(self):
        # Print episode summary
        # print("="*60)
        # print(f"Episode {self.episodeN} Summary:")
        # print(f"  Start Position: [0.0, 0.0, 0.0]")
        # print(f"  Goal Position: {self.goal}")
        # print(f"  Total Steps: {self.stepN}")
        # print(f"  Cumulative Reward: {self.episode_reward:.4f}")
        # print(f"  Success: {self.success}")
        # print(f"  Total Successes: {self.success_count}")
        # print("="*60)
        pass


    def on_episode_start(self):
        self.stepN = 0
        self.episodeN += 1
        self.episode_reward = 0  # Reset reward accumulator for new episode
        self.prev_position_xy = None
        self.prev_potential = None
        self.last_gradient_terms = {}
        self.episode_arena_diag = 1.0
        self.episode_goal_distance0 = 1.0
        self.episode_goal_scale = 1.0
  

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
            elif succes_rate > 0.7 and self.level == 2 and self.success_count > 900:
                self.level = 3
                self.game_config_handler = GameConfigHandler(range_dic_name="settings.dynamic_obstacles_dic")

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
        def _get_takeoff_status():
            return self.airgym.drone_pos(), self.airgym.client.simGetCollisionInfo()

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

        # 确保处于目标高度
        if abs(now[2] - self.airgym.z) > 0.1:
            self.airgym.client.moveToZAsync(self.airgym.z, 3).join()

        # 5. 暂停仿真以同步获取初始状态
        self.airgym.client.simPause(True)
        
        start_time = time.time()
        self.on_episode_start()
        state = self.init_state_f()
        print(f"Initial state acquisition time: {time.time() - start_time:.2f}s")
        # 重新读取当前位置，避免沿用起飞检查时的旧坐标
        now = self.airgym.drone_pos()

        self.prev_state = state
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32) if hasattr(self.action_space, 'shape') and self.action_space.shape else 0
        self.prev_velocity = self.airgym.drone_velocity()
        self.prev_position_xy = np.array(now[:2], dtype=np.float32)
        self._refresh_episode_scales(self.prev_position_xy)
        init_depth = self.depth_stack[-1] if len(self.depth_stack) > 0 else np.zeros((128, 128), dtype=np.float32)
        self.prev_potential, self.last_gradient_terms = self._compute_gradient_potential(now, init_depth)
        
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
