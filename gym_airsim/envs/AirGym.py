from settings_folder import settings
import msgs
import pygame
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
    def __init__(self, need_render=False, takeoff_height=-0.9, config=None, stack_frames=4):
        """
        初始化 AirSim 环境。
        
        Args:
            need_render (bool): 是否需要渲染 2D 轨迹图 (debug用)。
            takeoff_height (float): 起飞高度 (NED坐标，负数为高度)。
            config: 配置对象，包含高度限制等参数。
            stack_frames (int): 堆叠帧数 (默认4帧)。如果为1，则不进行堆叠（用于RNN/LSTM）。
        """
        # 如果 need_render 为 True，则可以使用 2D 窗口渲染环境

        self.config = config  # Store config for reward calculations

        # 从config读取高度限制参数，如果没有config则使用默认值
        
        self.max_altitude = config.max_flight_altitude
        self.min_altitude = config.min_flight_altitude
        self.min_altitude_penalty = config.min_altitude_penalty
        self.max_altitude_penalty = config.max_altitude_penalty
        self.altitude_penalty_value = config.altitude_penalty_value
    
        STATE_RGB_H, STATE_RGB_W = 128,128

        self.stack_frames = stack_frames
        self.episode_reward = 0

        self.base_dim = 10
        self.depth_shape = (self.stack_frames, STATE_RGB_H, STATE_RGB_W)
        self.gray_shape = (self.stack_frames, STATE_RGB_H, STATE_RGB_W)

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=np.float32(0), high=np.float32(255), shape=self.depth_shape, dtype=np.float32),
            "gray": spaces.Box(low=np.float32(0), high=np.float32(255), shape=self.gray_shape, dtype=np.float32),
            "base": spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_dim,), dtype=np.float32)
        })

        self.depth_stack = collections.deque(maxlen=self.stack_frames)
        self.gray_stack = collections.deque(maxlen=self.stack_frames)


        # 速度需要大于 2 或者持续时间大于 0.4
        # 否则效果不佳！
        if (settings.control_mode == "moveByVelocity"):
            self.action_space = spaces.Box(np.array([-0.3, -0.3], dtype=np.float32),
                                           np.array([+0.3, +0.3], dtype=np.float32),
                                           dtype=np.float32)
        elif (settings.control_mode == "Continuous_TD3"):
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

        disable_game_restart = getattr(config, "disable_game_restart", False) if config is not None else False
        self.game_handler = None if disable_game_restart else GameHandler()
        if self.game_handler is not None:
            self.game_handler.restart_game()

        # UE4 环境配置
        self.game_config_handler = GameConfigHandler()

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
        self.level=0
        self.success_deque = collections.deque(maxlen=100)
        self.seed()

        # Initialize pygame viewer if needed
        self.need_render = need_render
        self.screen = None
        self.clock = None
        
        if self.need_render:
            try:
                pygame.init()
                # Set SDL to use X11 on Linux if available
                import os
                if os.environ.get('DISPLAY') is None:
                    os.environ['SDL_VIDEODRIVER'] = 'dummy'
                self.screen = pygame.display.set_mode((1000, 1000))
                pygame.display.set_caption("AirSim RL 2D Visualization")
                self.clock = pygame.time.Clock()
                print("Initializing 2D rendering viewer with pygame...")
                # Initial display to show the window
                pygame.display.flip()
            except Exception as e:
                print(f"Failed to initialize pygame rendering: {e}")
                self.need_render = False
                self.screen = None
                self.clock = None

        # Initialize previous action for jerk penalty calculation
        if hasattr(self.action_space, 'shape') and self.action_space.shape:
            self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        else:
            self.prev_action = 0  # For discrete actions
        self.prev_velocity = np.zeros(3, dtype=np.float32)

    def getGoal(self):
        return self.goal

    def get_space(self):
        return self.observation_space,self.action_space

    def check_ue4_status(self):
        """
        Check if UE4 process exists. If not, restart it and reconnect client.
        Returns True if restarted.
        """
        if self.game_handler:
            if self.game_handler.check_and_recover_game():
                print("UE4 process missing, triggered restart. Reinitializing AirSim client...")
                time.sleep(15) # Wait for game to initialize
                try:
                    # Reconnect
                    # Use existing params if self.airgym exists, else defaults
                    z = -0.9
                    ip = None
                    port = None
                    if hasattr(self, 'airgym'):
                        z = self.airgym.z
                    
                    if hasattr(self, 'config') and self.config:
                        ip = self.config.airsim_ip
                        port = self.config.airsim_port
                        
                    self.airgym = AirLearningClient(z=z, ip=ip, port=port)
                    return True
                except Exception as e:
                    print(f"Error reconnecting after forced restart: {e}")
                    # Even if connection fails here, we return True so loop knows we reset
                    return True
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
        gray_stack_np = np.array(self.gray_stack, dtype=np.float32)
        return {
            "depth": depth_stack_np,
            "gray": gray_stack_np,
            "base": inform
        }

    def init_state_f(self):
        self.depth_stack.clear()
        self.gray_stack.clear()
        for i in range(self.stack_frames):
            self.depth_stack.append(self.airgym.getScreenDepth())
            self.gray_stack.append(self.airgym.getScreenGray())
            time.sleep(0.03)
        return self.get_obs()

    def state(self):
        """
        更新并获取当前的辅助状态信息 (inform)。
        (注意：此方法仅返回 inform 向量，不处理图像堆叠，图像堆叠在 step 方法中维护)。
        
        Returns:
            np.array: inform 向量 [相对距离xy, 高度, 前向速度, z速度, 偏航角速度, 姿态角, 朝向目标角度]
        """
        drone_pos = self.airgym.drone_pos()
        now = drone_pos[:2]
        altitude = -drone_pos[2]  # NED coordinate system, negative z is altitude
        
        pry = self.airgym.get_ryp()
        
        # 新的状态向量组成
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)  # [x, y]
        forward_speed = self.airgym.get_forward_speed()  # 前向速度
        z_velocity = self.airgym.get_z_velocity()  # z轴速度
        yaw_rate = self.airgym.get_yaw_rate()  # 偏航角速度
        
        # 为了向后兼容，仍保留这些属性
        self.velocity = np.array([forward_speed, z_velocity, yaw_rate])  # 用新的速度信息
        self.speed = forward_speed  # 前向速度作为主要速度指标
        
        # 组合新的状态向量: [相对距离xy(2), 高度(1), 前向速度(1), z速度(1), 偏航角速度(1), 姿态角(3), 朝向目标角度(1)]
        inform = np.concatenate((
            self.relative_position,  # [x_dist, y_dist] 
            [altitude],              # [altitude]
            [forward_speed],         # 前向速度
            [z_velocity],           # z轴速度
            [yaw_rate],             # 偏航角速度
            pry,                    # [pitch, roll, yaw]
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


    def computeReward(self, now, action, velocity_after=None):
        """
        计算每一步的奖励。
        
        奖励函数组成：
        1. 距离惩罚：距离目标越远，惩罚越大 (-distance * 0.03)。
        2. 朝向奖励：如果朝向目标飞行 (cos(r_yaw) > 0)，给予速度相关的奖励。
        3. jerk_penalty：动作变化惩罚。
        4. curvature_penalty：曲率惩罚，惩罚急转变。
        5. step_penalty：每步小惩罚。
        
        Args:
            now (np.array): 当前位置
            action: 当前动作
            velocity_after: 动作执行后的速度 (vx, vy, speed)
            
        Returns:
            float: 计算出的奖励值
        """

        distance_now = np.sqrt(np.power((self.goal[0] - now[0]), 2)
                               + np.power((self.goal[1] - now[1]), 2)
                               )
        now_pos = self.airgym.drone_pos()[:2]
        r_yaw = self.airgym.goal_direction(self.goal, now_pos)

        r = -distance_now*0.02#0.02

        if math.cos(r_yaw)>=0:
            r += self.speed*math.cos(r_yaw)

        # Calculate jerk penalty for smoothness
        if self.config is not None and hasattr(self.action_space, 'shape') and len(self.action_space.shape) > 0:
            action_diff = action - self.prev_action
            action_range = np.array([
                self.config.max_forward_speed - self.config.min_forward_speed,
                self.config.max_vertical_speed - (-self.config.max_vertical_speed),  # z velocity range
                self.config.max_yaw_rate - (-self.config.max_yaw_rate)  # yaw rate range
            ], dtype=np.float32)
            jerk_penalty = (
                0.05 * abs(action_diff[1]) / action_range[1] +
                0.05 * abs(action_diff[0]) / action_range[0] +
                0.1 * abs(action_diff[2]) / action_range[2]
            )
            jerk_penalty = float(np.clip(jerk_penalty, 0.0, 1.0))
        else:
            jerk_penalty = 0.0

        # Calculate curvature penalty
        curvature_penalty = 0.0
        if velocity_after is not None:
             v_xy_before = self.prev_velocity[2] # speed is at index 2
             v_xy_after = velocity_after[2]
             if v_xy_before > 0.1 and v_xy_after > 0.1:
                 dot_product = np.dot(velocity_after[:2], self.prev_velocity[:2])
                 cos_theta = dot_product / (v_xy_before * v_xy_after + 1e-6)
                 cos_theta = np.clip(cos_theta, -1.0, 1.0)
                 angle_change = np.arccos(cos_theta)
                 curvature_penalty = 20 * (angle_change ** 2) * (v_xy_after / self.config.max_forward_speed)
                 curvature_penalty = float(np.clip(curvature_penalty, 0.0, 1.0))

        # Add penalties to reward
        r -= jerk_penalty + curvature_penalty + self.config.step_penalty

        # Print reward components for debugging
        # distance_penalty = -distance_now * 0.02
        # orientation_reward = self.speed * math.cos(r_yaw) if math.cos(r_yaw) >= 0 else 0.0
        # print(f"距离惩罚: {distance_penalty:.3f}, 朝向奖励: {orientation_reward:.3f}, jerk_penalty: {jerk_penalty:.3f}, curvature_penalty: {curvature_penalty:.3f}, step_penalty: {self.config.step_penalty:.3f}, 总奖励: {r:.3f}")

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
        
        self.airgym.client.simPause(False)
        if (settings.control_mode == "moveByVelocity"):

            collided = self.airgym.take_continious_action(action, duration=self.action_duration)

        elif (settings.control_mode == "Continuous_TD3"):
             # 适配连续动作：如果有多余的维度（batch维），去除它
            if np.ndim(action) > 1:
                action = action[0]
            collided = self.airgym.take_continuous_action_3d(action, duration=self.action_duration)

        else:
            collided = self.airgym.take_discrete_action(action)

        self.airgym.client.simPause(True)

        # Update stacks
        depth_img = None
        gray_img = None
        for attempt in range(3):  # 初始尝试 + 3次重试
            try:
                depth_img = self.airgym.getScreenDepth()
                gray_img = self.airgym.getScreenGray()
                break
            except Exception as e:
                if attempt < 3:
                    print(f"Error fetching images in step (attempt {attempt+1}/4): {e}")
                    time.sleep(0.1)  # 短暂等待后重试
                else:
                    print(f"Failed to fetch images after 4 attempts: {e}. Restarting game...")
                    if self.game_handler:
                        self.game_handler.restart_game()
                        time.sleep(10)  # 等待游戏重启
                        # 重新初始化客户端
                        self.airgym = AirLearningClient(z=self.airgym.z, ip=self.config.airsim_ip if self.config else None, port=self.config.airsim_port if self.config else None)
                        # 最后一次尝试
                        try:
                            depth_img = self.airgym.getScreenDepth()
                            gray_img = self.airgym.getScreenGray()
                        except Exception as e2:
                            print(f"Still failed after restart: {e2}. Using zero arrays.")
                            depth_img = np.zeros((128, 128), dtype=np.float32)
                            gray_img = np.zeros((128, 128), dtype=np.float32)
                    else:
                        print("No game handler available. Using zero arrays.")
                        depth_img = np.zeros((128, 128), dtype=np.float32)
                        gray_img = np.zeros((128, 128), dtype=np.float32)
        
        self.depth_stack.append(depth_img)
        self.gray_stack.append(gray_img)

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
            print(f"[最大高度越界] 当前高度: {current_altitude:.2f}m，最大高度: {self.max_altitude}m")

        if distance < settings.success_distance_to_goal:
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
            if altitude_violation:
                print(f"[终止] 高度越界导致episode终止")

        elif self.stepN >= self.config.episode_length:
            done = True
            reward = -30.0
            self.success = False
            
        else:
            # 获取当前速度用于奖励计算 (vx, vy, speed)
            velocity_after = self.airgym.drone_velocity()
            # 计算基础奖励
            reward = self.computeReward(now, action, velocity_after=velocity_after)
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
        
        # 取消暂停，确保重置和起飞命令可以执行
        self.airgym.client.simPause(False)

        if len(self.success_deque)>0:
            succes_rate=sum(self.success_deque) / len(self.success_deque)
            if succes_rate>0.6 and self.level==0 and self.success_count>300:
                self.level=1
                self.game_config_handler=GameConfigHandler(range_dic_name="settings.medium_range_dic")
            elif succes_rate > 0.7 and self.level == 1 and self.success_count>600:
                self.level = 2
                self.game_config_handler = GameConfigHandler(range_dic_name="settings.hard_range_dic")
            elif succes_rate > 0.8 and self.level == 2 and self.success_count > 900:
                self.level = 3
                self.game_config_handler = GameConfigHandler(range_dic_name="settings.dynamic_obstacles_dic")
            

        # Clear renderer if available
        if self.need_render and self.screen is not None:
            self.screen.fill((255, 255, 255))
            pygame.display.flip()
        
        print("--- Resetting Episode ---")
        
        # 1. 环境随机化
        env_changed = self.randomize_env()
        
        # 2. 如果环境变化了，重置 Unreal (障碍物等)
        if env_changed:
            self.airgym.unreal_reset()
            print("Unreal environment reset done.")
        
        # 3. 重置无人机状态 (不管环境变没变，无人机都得重置)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.airgym.AirSim_reset()
                print("AirSim drone reset done.")
                break
            except Exception as e:
                print(f"AirSim reset failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
                if attempt == max_retries - 1:
                    print("Critical error: Unable to reset AirSim after multiple attempts. Restarting game...")
                    if self.game_handler:
                        self.game_handler.restart_game()
                        time.sleep(10) # Wait for game to reload
                        # Re-initialize client after game restart
                        # Note: airgym should ideally handle client reconnection, 
                        # but we can force a reconnect here if needed, or rely on AirSim_reset to retry.
                        self.airgym = AirLearningClient(z=self.airgym.z, ip=self.config.airsim_ip, port=self.config.airsim_port)

        # 4. 检查起飞结果并自旋重试 (如果失败)
        # Check takeoff status and retry if failed
        now = None
        collision_info = None

        # Robust check for drone position and collision info
        try:
             now = self.airgym.drone_pos()
             collision_info = self.airgym.client.simGetCollisionInfo()
        except Exception as e:
             # If getting status fails, try to reconnect heavily
             print(f"Status check failed after reset: {e}. Retrying with hard reset...")
             if self.game_handler:
                 self.game_handler.restart_game()
                 # Wait for game restart
                 time.sleep(10)
                 # Reconnect
                 self.airgym = AirLearningClient(z=self.airgym.z, ip=self.config.airsim_ip, port=self.config.airsim_port)
                 now = self.airgym.drone_pos()
                 collision_info = self.airgym.client.simGetCollisionInfo()

        retry_count = 0
        while ((-now[2]) < 0.5 or collision_info.has_collided) and retry_count < 3:
             print(f"Takeoff attempt {retry_count+1} failed! Height: {-now[2]:.2f}, Collided: {collision_info.has_collided}")
             
             # 随机更换种子并重新加载环境
             self.sampleGameConfig('Seed')
             self.airgym.unreal_reset()
             self.airgym.AirSim_reset()
             
             now = self.airgym.drone_pos()
             collision_info = self.airgym.client.simGetCollisionInfo()
             retry_count += 1

        # 确保处于目标高度
        if abs(now[2] - self.airgym.z) > 0.1:
            self.airgym.client.moveToZAsync(self.airgym.z, 3).join()

        # 5. 暂停仿真以同步获取初始状态
        self.airgym.client.simPause(True)
        
        self.on_episode_start()
        state = self.init_state_f()
        self.prev_state = state
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32) if hasattr(self.action_space, 'shape') and self.action_space.shape else 0
        self.prev_velocity = self.airgym.drone_velocity()
        
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
        self.game_config_handler.sample(*arg, np_random=self.np_random)

    def render(self, mode='human', close=False):
        """
        渲染环境的 2D debug 视图。
        
        绘制目标点(绿色圆)、无人机(紫色圆)和竞技场边界(线条)。
        
        Args:
            mode (str): 'human' 显示窗口, 'rgb_array' 返回图像数组
            close (bool): 是否关闭窗口
            
        Returns:
            np.array or None
        """
        # Return early if rendering is not available
        if not self.need_render or self.screen is None:
            return None
        
        # Handle pygame events to prevent window from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                self.need_render = False
                return None
        
        # Clear screen with white background
        self.screen.fill((255, 255, 255))
        
        # Draw goal (green circle)
        goal_pos = (int(self.goal[0]*10 + 500), int(self.goal[1]*10 + 500))
        pygame.draw.circle(self.screen, (64, 200, 26), goal_pos, 30)  # Green color
        
        # Draw UAV (purple circle)
        try:
            now = self.airgym.drone_pos()
            uav_pos = (int(now[0]*10 + 500), int(now[1]*10 + 500))
            pygame.draw.circle(self.screen, (200, 95, 170), uav_pos, 10)  # Purple color
        except Exception as e:
            print(f"Error getting drone position for rendering: {e}")
        
        # Draw arena boundaries
        try:
            size = self.game_config_handler.get_cur_item("ArenaSize")
            h = int(size[0] * 10)
            w = int(size[1] * 10)
            
            # Calculate boundary positions
            left = (1000 - h) // 2
            right = (1000 + h) // 2
            top = (1000 - w) // 2
            bottom = (1000 + w) // 2
            
            # Draw boundary lines (black)
            pygame.draw.line(self.screen, (0, 0, 0), (left, top), (right, top), 2)
            pygame.draw.line(self.screen, (0, 0, 0), (left, top), (left, bottom), 2)
            pygame.draw.line(self.screen, (0, 0, 0), (right, top), (right, bottom), 2)
            pygame.draw.line(self.screen, (0, 0, 0), (left, bottom), (right, bottom), 2)
        except Exception as e:
            print(f"Error drawing arena boundaries: {e}")
        
        # Update display
        if mode == 'human':
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(60)  # 60 FPS
        
        # Return RGB array if requested
        if mode == 'rgb_array':
            # Convert pygame surface to RGB array
            try:
                rgb_array = pygame.surfarray.array3d(self.screen)
                # Pygame uses RGB order, transpose to get (height, width, channels)
                rgb_array = rgb_array.transpose([1, 0, 2])
                return rgb_array
            except Exception as e:
                print(f"Error converting screen to RGB array: {e}")
                return None
        
        return None

    def close(self):
        """Close the environment and clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
