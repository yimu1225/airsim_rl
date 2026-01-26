from settings_folder import settings
import msgs
try:
    from gymnasium.envs.classic_control import rendering
except ImportError:
    rendering = None
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
    def __init__(self, need_render=False, takeoff_height=-0.9, config=None):
        """
        初始化 AirSim 环境。
        
        Args:
            need_render (bool): 是否需要渲染 2D 轨迹图 (debug用)。
            takeoff_height (float): 起飞高度 (NED坐标，负数为高度)。
            config: 配置对象，包含高度限制等参数。
        """
        # 如果 need_render 为 True，则可以使用 2D 窗口渲染环境

        # 从config读取高度限制参数，如果没有config则使用默认值
        if config is not None:
            self.max_altitude = config.max_flight_altitude
            self.min_altitude = config.min_flight_altitude
            self.min_altitude_penalty = config.min_altitude_penalty
        else:
            # 使用默认值
            self.max_altitude = 4.5
            self.min_altitude = 0.2
            self.min_altitude_penalty = 5.0

        STATE_RGB_H, STATE_RGB_W = 112,112

        self.stack_frames=4
        self.episode_reward = 0  # Track cumulative reward for each episode

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.stack_frames, STATE_RGB_H, STATE_RGB_W))


        # 速度需要大于 2 或者持续时间大于 0.4
        # 否则效果不佳！
        if (settings.control_mode == "moveByVelocity"):
            self.action_space = spaces.Box(np.array([-0.3, -0.3]),
                                           np.array([+0.3, +0.3]),
                                           dtype=np.float32)
        elif (settings.control_mode == "Continuous_TD3"):
            # Continuous action space: [forward_speed, z_velocity, yaw_rate]
            fwd_min = config.min_forward_speed
            fwd_max = config.max_forward_speed
            z_max = config.max_vertical_speed
            yaw_max = config.max_yaw_rate

            self.action_space = spaces.Box(
                low=np.array([fwd_min, -z_max, -yaw_max]),
                high=np.array([fwd_max, z_max, yaw_max]),
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
        if config is not None:
            self.clock_speed_factor = max(getattr(config, 'clock_speed_factor', 1.0), 1e-6)
            self.raw_action_duration = getattr(config, 'action_duration', 0.5)
        else:
            self.clock_speed_factor = 1.0
            self.raw_action_duration = 0.5
        self.action_duration = self.raw_action_duration / self.clock_speed_factor

        # 重置环境变量
        self.success_count = 0
        self.episodeN = 0
        self.stepN = 0
        self.goal = airsimize_coordinates(self.game_config_handler.get_cur_item("End"))


        self.prev_state = self.init_state_f()
        self.init_state = self.prev_state
        self.success = False
        self.level=0
        self.success_deque = collections.deque(maxlen=100)
        self.seed()

        self.need_render=need_render
        if self.need_render:
            self.viewer = rendering.Viewer(1000, 1000)

    def getGoal(self):
        return self.goal

    def get_space(self):
        return self.observation_space,self.action_space

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

    def init_state_f(self):
        """
        初始化环境状态。
        
        获取当前无人机位置、姿态，并连续采集 self.stack_frames 帧深度图像形成初始状态栈。
        同时计算新的状态向量：[相对距离xy, 前向速度, z速度, 偏航角速度, 姿态角, 朝向目标角度]。
        
        Returns:
            list: [stacked_images, information_vector]
        """
        now = self.airgym.drone_pos()[:2]
        pry = self.airgym.get_ryp()
        d=[]
        for i in range(self.stack_frames):
            d.append(self.airgym.getScreenDepth())
            time.sleep(0.03)
        
        # 新的状态向量组成
        self.r_yaw = self.airgym.goal_direction(self.goal, now)
        self.relative_position = self.airgym.get_distance(self.goal)  # [x, y]
        forward_speed = self.airgym.get_forward_speed()  # 前向速度
        z_velocity = self.airgym.get_z_velocity()  # z轴速度  
        yaw_rate = self.airgym.get_yaw_rate()  # 偏航角速度
        
        # 组合新的状态向量: [相对距离xy(2), 前向速度(1), z速度(1), 偏航角速度(1), 姿态角(3), 朝向目标角度(1)]
        inform = np.concatenate((
            self.relative_position,  # [x_dist, y_dist]
            [forward_speed],         # 前向速度
            [z_velocity],           # z轴速度
            [yaw_rate],             # 偏航角速度
            pry,                    # [pitch, roll, yaw]
            self.r_yaw              # [relative_angle_to_target]
        ))
        
        d=np.stack(d)
        return [d, inform]

    def state(self):
        """
        更新并获取当前的辅助状态信息 (inform)。
        (注意：此方法仅返回 inform 向量，不处理图像堆叠，图像堆叠在 step 方法中维护)。
        
        Returns:
            np.array: inform 向量 [相对距离xy, 前向速度, z速度, 偏航角速度, 姿态角, 朝向目标角度]
        """
        now = self.airgym.drone_pos()[:2]
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
        
        # 组合新的状态向量: [相对距离xy(2), 前向速度(1), z速度(1), 偏航角速度(1), 姿态角(3), 朝向目标角度(1)]
        inform = np.concatenate((
            self.relative_position,  # [x_dist, y_dist] 
            [forward_speed],         # 前向速度
            [z_velocity],           # z轴速度
            [yaw_rate],             # 偏航角速度
            pry,                    # [pitch, roll, yaw]
            self.r_yaw              # [relative_angle_to_target]
        ))
        
        return inform

    def computeReward(self, now):
        """
        计算每一步的奖励。
        
        奖励函数组成：
        1. 距离惩罚：距离目标越远，惩罚越大 (-distance * 0.03)。
        2. 朝向奖励：如果朝向目标飞行 (cos(r_yaw) > 0)，给予速度相关的奖励。
        
        Args:
            now (np.array): 当前位置
            
        Returns:
            float: 计算出的奖励值
        """

        distance_now = np.sqrt(np.power((self.goal[0] - now[0]), 2)
                               + np.power((self.goal[1] - now[1]), 2)
                               )
        now = self.airgym.drone_pos()[:2]
        r_yaw = self.airgym.goal_direction(self.goal, now)

        r = -distance_now*0.03#0.02

        if math.cos(r_yaw)>=0:
            r += self.speed*math.cos(r_yaw)

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

        # 更新状态
        inform = self.state()

        d=self.airgym.getScreenDepth()
        state = []
        for i in range(self.stack_frames):
            if i <(self.stack_frames-1):
                state.append(self.prev_state[0][i+1])
            else:
                state.append(d)
        state=[np.stack(state), inform]
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

        elif self.stepN >= settings.nb_max_episodes_steps:
            done = True
            reward = -20.0
            self.success = False
            
        else:
            # 计算基础奖励
            reward = self.computeReward(now)
            done = False
            self.success = False
            
            # 检查是否低于最小高度（施加惩罚但不终止）
            if current_altitude < self.min_altitude:
                reward -= self.min_altitude_penalty
                print(f"[最小高度惩罚] 当前高度: {current_altitude:.2f}m，最小高度: {self.min_altitude}m，惩罚: -{self.min_altitude_penalty}")
        
        # Accumulate reward for episode
        self.episode_reward += reward

        self.prev_state = state


        if (done):
            if self.success:
                self.success_deque.append(1)
            else:
                self.success_deque.append(0)
            self.on_episode_end()

        return state, reward, done, False, {}


    def on_episode_end(self):
        # Print episode summary
        print("="*60)
        print(f"Episode {self.episodeN} Summary:")
        print(f"  Start Position: [0.0, 0.0, 0.0]")
        print(f"  Goal Position: {self.goal}")
        print(f"  Total Steps: {self.stepN}")
        print(f"  Cumulative Reward: {self.episode_reward:.4f}")
        print(f"  Success: {self.success}")
        print(f"  Total Successes: {self.success_count}")
        print("="*60)


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
            if succes_rate>0.7 and self.level==0 and self.success_count>300:
                self.level=1
                self.game_config_handler=GameConfigHandler(range_dic_name="settings.medium_range_dic")
            elif succes_rate > 0.7 and self.level == 1 and self.success_count>600:
                self.level = 2
                self.game_config_handler = GameConfigHandler(range_dic_name="settings.hard_range_dic")

        if self.need_render:
            self.viewer.geoms.clear()
            self.viewer.onetime_geoms.clear()
        
        print("--- Resetting Episode ---")
        
        # 1. 环境随机化
        env_changed = self.randomize_env()
        
        # 2. 如果环境变化了，重置 Unreal (障碍物等)
        if env_changed:
            self.airgym.unreal_reset()
            print("Unreal environment reset done.")
        
        # 3. 重置无人机状态 (不管环境变没变，无人机都得重置)
        self.airgym.AirSim_reset()
        print("AirSim drone reset done.")

        # 4. 检查起飞结果并自旋重试 (如果失败)
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
        # 画一个直径为 30 的circle
        #self.viewer.geoms.clear()
        #self.viewer.onetime_geoms.clear()

        goal = rendering.make_circle(30)
        goal.set_color(0.25, 0.78, 0.1)
        # 添加一个平移操作
        circle_transform = rendering.Transform(translation=(self.goal[0]*10+500, self.goal[1]*10+500))
        # 让圆添加平移这个属性
        goal.add_attr(circle_transform)
        self.viewer.add_geom(goal)

        uav = rendering.make_circle(10)
        uav.set_color(0.78, 0.37, 0.66)
        now = self.airgym.drone_pos()
        uav_transform = rendering.Transform(translation=(now[0]*10 + 500, now[1]*10 + +500))
        uav.add_attr(uav_transform)
        self.viewer.add_geom(uav)

        size = self.game_config_handler.get_cur_item("ArenaSize")
        h = size[0] * 10
        w = size[1] * 10

        line1 = rendering.Line(((1000 - h) / 2, (1000 - w) / 2), ((h + 1000) / 2, (1000 - w) / 2))
        line2 = rendering.Line(((1000 - h) / 2, (1000 - w) / 2), ((1000 - h) / 2, (w + 1000) / 2))
        line3 = rendering.Line(((h + 1000) / 2, (1000 - w) / 2), ((h + 1000) / 2, (1000 + w) / 2))
        line4 = rendering.Line(((1000 - h) / 2, (1000 + w) / 2), ((h + 1000) / 2, (1000 + w) / 2))
        # 给元素添加颜色
        line1.set_color(0, 0, 0)
        line2.set_color(0, 0, 0)
        line3.set_color(0, 0, 0)
        line4.set_color(0, 0, 0)
        self.viewer.add_geom(line1)
        self.viewer.add_geom(line2)
        self.viewer.add_geom(line3)
        self.viewer.add_geom(line4)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')