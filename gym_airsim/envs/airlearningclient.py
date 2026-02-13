import airsim
import numpy as np
import math
import time
import cv2
from settings_folder import settings
import msgpackrpc
import functools

class CompatImageResponse:
    """
    兼容性图像响应类
    用于包装AirSim返回的图像数据，确保在不同版本的AirSim客户端之间接口一致。
    Attributes:
        width (int): 图像宽度
        height (int): 图像高度
        image_data_uint8 (bytes): 8位无符号整数格式的图像数据
        image_data_float (list): 浮点数格式的图像数据（用于深度图等）
        pixel_as_float (bool): 是否将像素作为浮点数处理
        compress (bool): 图像是否压缩
        message (str): 附加消息
        time_stamp (float): 时间戳
    """
    def __init__(self, d):
        self.width = d.get('width', 0)
        self.height = d.get('height', 0)
        self.image_data_uint8 = d.get('image_data_uint8', b'')
        self.image_data_float = d.get('image_data_float', [])
        self.pixel_as_float = d.get('pixel_as_float', False)
        self.compress = d.get('compress', True)
        self.message = d.get('message', '')
        self.time_stamp = d.get('time_stamp', 0)

class AirLearningClient(object):
    """
    AirLearning客户端类
    用于与AirSim仿真器交互，封装了获取传感器数据（图像、位置、速度等）
    和发送控制指令（移动、偏航等）的方法。
    同时包含对AirSim客户端的兼容性补丁。
    """
    def _apply_client_patches(self):
        """
        [内部方法] 应用客户端兼容性补丁
        
        该方法会对 self.client 的 simGetImages 和 getMultirotorState 方法进行猴子补丁(monkey patch)，
        以解决AirSim服务器版本差异导致的参数不匹配或字段未知的问题。
        1. simGetImages: 强制只使用 requests 和 vehicle_name 两个参数调用RPC。
        2. getMultirotorState: 移除可能导致崩溃的 'trip_stats' 未知字段。
        """
        # Store reference to the original RPC client
        rpc_client = self.client.client
        
        # Patch simGetImages - server expects only 2 arguments (not 3)
        def compat_simGetImages(requests, vehicle_name = '', external = False):
            # Serialize ImageRequest objects to msgpack format (dicts)
            requests_msgpack = [req.to_msgpack() if hasattr(req, 'to_msgpack') else req for req in requests]
            # Call RPC with only 2 arguments (requests and vehicle_name)
            # The 'external' parameter is ignored by this server
            responses = rpc_client.call('simGetImages', requests_msgpack, vehicle_name)
            return [CompatImageResponse(r) for r in responses]
        self.client.simGetImages = compat_simGetImages

        # Patch getMultirotorState to remove unknown 'trip_stats' field that crashes older airsim client
        def compat_getMultirotorState(vehicle_name = ''):
            state = self.client.client.call('getMultirotorState', vehicle_name)
            if 'trip_stats' in state:
                del state['trip_stats']
            return airsim.MultirotorState.from_msgpack(state)
        self.client.getMultirotorState = compat_getMultirotorState

    def __init__(self, z, ip=None, port=None, config=None):
        """
        初始化 AirLearningClient
        1. 初始化图像缓冲区 (last_img, last_grey, last_rgb)。
        2. 连接到AirSim仿真器 (MultirotorClient)。
        3. 应用兼容性补丁。
        4. 确认连接并启用API控制。
        5. 解锁无人机 (Arm)。
        6. 设置默认飞行高度 z。
        
        Args:
            z (float): 默认飞行高度 (NED坐标，负数为高度)
            ip (str): AirSim服务器IP地址，如果为None则使用settings中的配置
            port (int): AirSim服务器端口，如果为None则使用settings中的配置
        """
        
        self.z = z

        self.last_img = np.zeros((1, 112, 112))
        self.last_grey = np.zeros((112, 112))
        self.last_rgb = np.zeros((112, 112, 3))
        self.width, self.height=84,84 ## DeepMind 设定分辨率

        # 使用传入的ip和port，如果为None则使用settings中的默认值
        client_ip = ip if ip is not None else settings.ip
        client_port = port if port is not None else getattr(settings, 'port', 41451)
        
        # connect to the AirSim simulator
        # Set timeout to 10 seconds to prevent hanging if game crashes
        self.client = airsim.MultirotorClient(ip=client_ip, port=client_port, timeout_value=10)
        self._apply_client_patches()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        #self.z=-3
        # self.z = -0.9 # Now initialized in __init__ arguments

    def goal_direction(self, goal, pos):
        """
        计算目标方向与当前航向的相对角度。
        
        Args:
            goal (list/array): 目标位置坐标 [x, y]
            pos (list/array): 当前位置坐标 [x, y]
            
        Returns:
            np.array: 返回一个包含相对角度的数组 [track]，单位为度。
                      这个角度表示无人机需要旋转多少度才能朝向目标。
        """

        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        yaw = math.degrees(yaw)

        pos_angle = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)
        #return ((math.degrees(track) - 180) % 360) - 180

        return np.array([track])

    def getScreenRGB(self):
        """
        获取前方摄像头(ID: "0")的RGB图像。
        
        1. 发送 simGetImages 请求获取 Scene 类型的图像。
        2. 将返回的 uint8 数据转换为 numpy 数组。
        3. 调整 reshape 为 (height, width, channels)。
        4. 如果是4通道(BGRA)，转换为3通道(BGR)。
        5. 如果获取失败，返回上一帧图像 (self.last_rgb)。
        
        Returns:
            np.array: RGB图像数据
        """
        # 3D? 假设使用的是摄像头 "0" 或 "front_center"
        # responses = self.client.simGetImage("3d", airsim.ImageType.Scene) # 旧调用，返回字节而不是列表？不，Python客户端simGetImage返回字节？
        # 实际上 simGetImage 在旧版本中返回原始字节或字节列表？
        # 为了一致性和更安全的 API，我们使用 simGetImages
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)], vehicle_name='SimpleFlight')
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        if ((response.width != 0 or response.height != 0)):
            # img_rgba = img1d.reshape((response.height, response.width, 3)) # AirSim 如果不压缩通常返回3通道？还是4通道BGRA？
            # 然而，如果 compressed=False（上面的默认值），它返回未压缩的 "image_data_uint8"。
            # 通常是 BGRA（4通道）。
            # 如果设置了默认捕获，我们需要验证。
            # 假设 Scene 是3通道？不，通常是4。
            # 让我们安全地检查形状
            if img1d.size == response.width * response.height * 4:
                 img_rgba = img1d.reshape((response.height, response.width, 4))
                 rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
            elif img1d.size == response.width * response.height * 3:
                 rgb = img1d.reshape((response.height, response.width, 3))
            else:
                 # 回退方案
                 rgb = self.last_rgb
            
            self.last_rgb=rgb
        else:
            print("Something bad happened! Restting AirSim!")
            self.AirSim_reset()

            rgb=self.last_rgb
        rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)

        return rgb

    def getScreenDepth(self):
        """
        获取前方摄像头(ID: "0")的深度图像。
        
        1. 发送 simGetImages 请求获取 DepthPerspective 类型的图像（浮点数据）。
        2. 如果请求失败，返回全0数组。
        3. 截断最大深度值 (clip max=20)，并归一化缩放到 0-255 范围。
        4. 将数据 reshape 为 2D 图像。
        5. 为了适配模型输入，统一 resize 到 112x112 分辨率。
        
        Returns:
            list or np.array: 处理后的深度图像。如果有多个相机，返回列表；单个相机则返回单个数组。
                              当前代码逻辑中只请求了一个相机，所以通常返回单个数组。
        """
        # 使用第一个可用车辆和摄像头 "0" (前置中心)
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True,
                                                                  False)
                                              ]
                                             ,vehicle_name='SimpleFlight')
        #responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis,True, False)])

        if (responses == None):
            print("Camera is not returning image!")
            print("Image size:" + str(responses[0].height) + "," + str(responses[0].width))
            img = [np.array([0]) for _ in responses]
        else:
            img = []
            for res in responses:
                img.append(np.array(res.image_data_float, dtype=float))
            img = np.stack(img, axis=0)


        ## 深度图预处理
        img = img.clip(max=10)
        # 归一化到 0-255 范围
        img = (img / 10.0) * 255.0

        img2d=[]
        for i in range(len(responses)):
            if ((responses[i].width != 0 or responses[i].height != 0)):
                img2d.append(np.reshape(img[i], (responses[i].height, responses[i].width)))
            else:
                print("Something bad happened! Restting AirSim!")
                img2d.append(self.last_img[i])

        self.last_img = np.stack(img2d, axis=0)

        # Resize to 128x128
        img2d_resized = []
        for im in img2d:
             if im.shape != (128, 128):
                 im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
             img2d_resized.append(im)
        
        if len(img2d_resized)>1:
            return img2d_resized
        else:
            return img2d_resized[0]

    def getScreenGray(self):
        rgb = self.getScreenRGB()
        if rgb is None:
             return np.zeros((128, 128), dtype=np.float32)
        
        # Resize if needed
        if rgb.shape[0] != 128 or rgb.shape[1] != 128:
             rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32)

    def get_ryp(self):
        """
        获取无人机当前的姿态角（Roll, Yaw, Pitch）。
        
        Returns:
            np.array: [pitch, roll, yaw] 数组。
        """
        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        return np.array([pitch, roll, yaw])

    def drone_pos(self):
        """
        获取无人机当前的绝对位置坐标。
        
        Returns:
            np.array: [x, y, z] 坐标数组。
        """
        pos = self.client.simGetVehiclePose().position
        x = pos.x_val
        y = pos.y_val
        z = pos.z_val

        return np.array([x, y, z])

    def drone_velocity(self):
        """
        获取无人机当前的线速度。
        
        Returns:
            np.array: [vx, vy, speed] 数组。其中 speed 是水平面上的合速度 (sqrt(vx^2 + vy^2))。
        """
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        v_x = vel.x_val
        v_y = vel.y_val
        v_z = vel.z_val
        speed = np.sqrt(v_x ** 2 + v_y ** 2)
        return np.array([v_x, v_y, speed])

    def get_distance(self, goal):
        """
        计算无人机当前位置到目标点的距离向量。
        注意：此处似乎只计算了X和Y轴的差值，没有计算欧几里得距离作为返回值的一部分，
        只是返回了 [diff_x, diff_y]。
        
        Args:
            goal (list/array): 目标点坐标
            
        Returns:
            np.array: [x_distance, y_distance]
        """
        now = self.client.simGetVehiclePose().position
        xdistance = (goal[0] - now.x_val)
        ydistance = (goal[1] - now.y_val)
        #zdistance = (goal[2] - now.z_val)
        euclidean = np.sqrt(np.power(xdistance,2) + np.power(ydistance,2))
        return np.array([xdistance, ydistance])

    def get_velocity(self):
        """
        获取三维速度向量。
        
        Returns:
            np.array: [vx, vy, vz]
        """
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def get_forward_speed(self):
        """
        获取无人机在当前朝向上的前向速度。
        
        Returns:
            float: 前向速度 (m/s)
        """
        # 获取当前速度和朝向
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        
        # 计算前向速度: vx*cos(yaw) + vy*sin(yaw)
        forward_speed = vel.x_val * math.cos(yaw) + vel.y_val * math.sin(yaw)
        return forward_speed
    
    def get_z_velocity(self):
        """
        获取z轴速度。
        
        Returns:
            float: z轴速度 (m/s，NED坐标系)
        """
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return vel.z_val
    
    def get_yaw_rate(self):
        """
        获取偏航角速度。
        
        Returns:
            float: 偏航角速度 (rad/s)
        """
        angular_vel = self.client.getMultirotorState().kinematics_estimated.angular_velocity
        return angular_vel.z_val

    def AirSim_reset(self):
        """
        重置 AirSim 客户端连接和无人机状态。
        """
        # 暂停仿真
        
        # self.client.simPause(True)
       
            
        # Fast path: reuse existing client to avoid expensive reconnect each episode.
        # Fallback to reconnect only when these commands fail.
        try:
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
        except Exception:
            current_ip = self.client._client._ip if hasattr(self.client, '_client') and hasattr(self.client._client, '_ip') else settings.ip
            current_port = self.client._client._port if hasattr(self.client, '_client') and hasattr(self.client._client, '_port') else getattr(settings, 'port', 41451)

            reconnect_ok = False
            last_error = None
            for _ in range(3):
                try:
                    self.client = airsim.MultirotorClient(ip=current_ip, port=current_port, timeout_value=5)
                    self._apply_client_patches()
                    self.client.confirmConnection()
                    self.client.reset()
                    self.client.enableApiControl(True)
                    self.client.armDisarm(True)
                    reconnect_ok = True
                    break
                except Exception as e:
                    last_error = e
                    time.sleep(0.5)

            if not reconnect_ok:
                raise RuntimeError(f"AirSim_reset confirmConnection failed: {last_error}")
        
        # 使用 takeoff（移除超时参数以避免兼容性问题）
        # self.client.takeoffAsync().join()
        
        # 移动到指定高度 (self.z 是 NED 坐标，负数为高度)
        # 例如 self.z = -0.9
        # self.client.moveToZAsync(float(self.z), 1.0).join()
        self.client.moveToZAsync(float(self.z), 1.0)
        time.sleep(1.0)

        self.client.hoverAsync().join()
        
        # 重新启动仿真
        # self.client.simPause(False)
        
        # 打印当前起飞高度
        pos = self.client.simGetVehiclePose().position
        # print(f"[AirSim_reset] Reset & Takeoff sequence finished. Current Altitude (Z): {pos.z_val:.4f} (Target: {self.z})")

    def unreal_reset(self):
        """
        重置 Unreal 虚幻引擎环境。
        通过调用自定义 RPC 'resetUnreal'，触发 UE4 重新读取 JSON 配置文件，
        并在运行时重建环境（如障碍物、竞技场等），而无需重启游戏进程。
        
        Returns:
            bool: 总是 True
        """
        # 调用自定义RPC从JSON重载环境
        self.client.client.call('resetUnreal')
        time.sleep(2.0)  # 给UE引擎时间重建环境 - 1.0s太短会导致连接断开，回调至2.0s
        return True


    def take_continuous_action_3d(self, action, duration=0.1):
        """
        执行3D连续动作控制 [v_forward, v_z, yaw_rate]。
        
        Args:
            action (np.array): [v_forward, v_z, yaw_rate]
            duration (float): 动作持续时间
            
        Returns:
            bool: 碰撞状态
        """
        
        v_forward = float(action[0])
        v_z = float(action[1])
        yaw_rate = float(action[2]) # rad/s 

        # 获取当前偏航角
        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        
        # 将机体坐标系的前向速度分解为世界坐标系的 vx, vy
        vx = math.cos(yaw) * v_forward
        vy = math.sin(yaw) * v_forward
        
        # 使用 moveByVelocityAsync
        # yaw_mode: is_rate=True, yaw_or_rate=yaw_rate (deg/s)
        # AirSim Python API expected degrees for yaw_or_rate
        
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate)) 
        
        # 使用 moveByVelocityAsync 控制 3D 速度 (vx, vy, vz)
        try:
             self.client.moveByVelocityAsync(vx, vy, v_z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode).join()
        except msgpackrpc.error.TimeoutError:
             print("RPC TimeoutError during moveByVelocityAsync, ignoring and proceeding to collision check")

        for attempt in range(3):
            try:
                return self.client.simGetCollisionInfo().has_collided
            except msgpackrpc.error.TimeoutError:
                print(f"RPC TimeoutError during simGetCollisionInfo (Attempt {attempt+1}/3)")
                if attempt < 2:
                    time.sleep(1)
                else:
                    print("Max retries reached. Triggering game restart...")
                    # Lazy import to avoid circular dependencies if any
                    from game_handling.game_handler_class import GameHandler
                    gh = GameHandler()
                    gh.restart_game()
                    
                    # Reconnect client after restart
                    client_ip = settings.ip
                    client_port = getattr(settings, 'port', 41451)
                    
                    print("Reconnecting AirLearningClient...")
                    self.client = airsim.MultirotorClient(ip=client_ip, port=client_port, timeout_value=3600)
                    self._apply_client_patches()
                    self.client.confirmConnection()
                    self.client.enableApiControl(True)
                    self.client.armDisarm(True)
                    
                    # After restart, we are safe, returns False for collision
                    return False

    def take_continious_action(self, action, duration=None):
        """
        执行连续动作控制（基于速度控制）。
        
        Args:
            action (np.array): 动作向量 [delta_x, delta_y]，用于调整当前速度。
                               值会被截断在 [-0.3, 0.3] 之间。
            duration (float): 指令持续时间，默认使用内部参数。
        
        Returns:
            bool: 执行动作后是否发生碰撞 (collided)。
        """

        if(settings.control_mode=="moveByVelocity"):
            action=np.clip(action, -0.3, 0.3)

            detla_x = action[0]
            detla_y = action[1]
            v=self.drone_velocity()
            v_x = v[0] + detla_x
            v_y = v[1] + detla_y

            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
            move_duration = duration if duration is not None else settings.vel_duration
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, move_duration, 1, yaw_mode).join()

        else:
            raise NotImplementedError

        collided = self.client.simGetCollisionInfo().has_collided

        return collided
        #Todo : Stabilize drone


    def straight(self, speed, duration):
        """
        [辅助方法] 沿当前偏航角方向直线飞行。
        """
        pitch, roll, yaw  = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 1, yaw_mode).join()


    def move_right(self, speed, duration):
        """
        [辅助方法] 向右平移飞行。
        """
        pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        vx = math.sin(yaw) * speed
        vy = math.cos(yaw) * speed
        self.client.moveByVelocityZAsync(vx, vy, self.z, duration, 0).join()
        start = time.time()
        return start, duration

    def yaw_right(self, rate, duration):
        """
        [辅助方法] 向右偏航旋转。
        """
        self.client.rotateByYawRateAsync(rate, duration).join()
        start = time.time()
        return start, duration

    def pitch_up(self, duration):
        """
        [辅助方法] 向上倾斜（爬升）。
        """
        self.client.moveByVelocityAsync(0,0,1,duration,1).join()
        start = time.time()
        return start, duration

    def pitch_down(self, duration):
        """
        [辅助方法] 向下倾斜（下降）。
        """
        #yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        self.client.moveByVelocityAsync(0,0,-1,duration,1).join()
        start = time.time()
        return start, duration

    def move_forward_Speed(self, speed_x = 0.5, speed_y = 0.5, duration = 0.5):
        """
        [辅助方法] 以指定的前向和侧向速度飞行。
        速度基于当前偏航角进行分解合成。
        """
        # speedx 是在 FLU (前左上) 坐标系
        #z = self.drone_pos()[2]
        pitch, roll, yaw  = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        vx = math.cos(yaw) * speed_x + math.sin(yaw) * speed_y
        vy = math.sin(yaw) * speed_x - math.cos(yaw) * speed_y

        drivetrain = 1
        yaw_mode = airsim.YawMode(is_rate= False, yaw_or_rate = 0)

        self.client.moveByVelocityZAsync(vx = (vx +vel.x_val)/2 ,
                             vy = (vy +vel.y_val)/2 , # 这样做是为了尝试平滑运动
                             z = self.z,
                             duration = duration,
                             drivetrain = drivetrain,
                             yaw_mode=yaw_mode
                            ).join()
        start = time.time()
        return start, duration

    def take_discrete_action(self, action):
        """
        执行离散动作控制。
        
        根据传入的 action 索引执行预定义的飞行动作。
        Action 0-1: 直行 (不同速度)
        Action 2-5: 曲线移动/转向移动
        Action 6-7: 原地旋转
        
        Args:
            action (int): 动作索引 (0-7)
            
        Returns:
            bool: 执行动作后是否发生碰撞 (collided)
        """

        if action == 0:
            self.straight(settings.mv_fw_spd_2, settings.rot_dur)
        if action == 1:
            self.straight(settings.mv_fw_spd_3, settings.rot_dur)
        if action == 2:
            #self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur/2)
            #self.straight(settings.mv_fw_spd_3, settings.rot_dur/2)
            self.move_forward_Speed(settings.mv_fw_spd_2*math.cos(0.314),
                                    settings.mv_fw_spd_2*math.sin(0.314), settings.rot_dur)

        if action == 3:
            #self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_3 * math.cos(0.314),
                                    settings.mv_fw_spd_3 * math.sin(0.314), settings.rot_dur)

        if action == 4:
            #self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_2 * math.cos(0.314),
                                    -settings.mv_fw_spd_2 * math.sin(0.314), settings.rot_dur)
        if action == 5:
            #self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur / 2)
            #self.straight(settings.mv_fw_spd_4, settings.rot_dur / 2)
            self.move_forward_Speed(settings.mv_fw_spd_3 * math.cos(0.314),
                                    -settings.mv_fw_spd_3 * math.sin(0.314), settings.rot_dur)

        if action == 6:
            self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur )
        if action == 7:
            self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur)
        
        # --- 以下为旧的动作实现或备用方案 ---
        '''
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0)
        if action == 0:
            v = self.drone_velocity()
            v_x = v[0] + 0.25
            v_y = v[1] + 0
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 1:
            v = self.drone_velocity()
            v_x = v[0] - 0.25
            v_y = v[1] + 0
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 2:
            v = self.drone_velocity()
            v_x = v[0] + 0
            v_y = v[1] + 0.25
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()

        if action == 3:
            v = self.drone_velocity()
            v_x = v[0] + 0
            v_y = v[1] - 0.25
            self.client.moveByVelocityZAsync(v_x, v_y, self.z, settings.vel_duration, 1, yaw_mode).join()


        
        if action == 0:
            start, duration = self.straight(settings.mv_fw_spd_4, settings.mv_fw_dur)
        if action == 1:
            start, duration = self.straight(settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 2:
            start, duration = self.straight(settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 3:
            start, duration = self.straight(settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 4:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_3, settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 5:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_2, settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 6:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_1, settings.mv_fw_spd_1, settings.mv_fw_dur)
        if action == 7:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_3, -settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 8:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_2, -settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 9:
            start, duration = self.move_forward_Speed(settings.mv_fw_spd_1, -settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 10:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_4, settings.mv_fw_dur)
        if action == 11:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_3, settings.mv_fw_dur)
        if action == 12:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_2, settings.mv_fw_dur)
        if action == 13:
            start, duration = self.straight(-0.5*settings.mv_fw_spd_1, settings.mv_fw_dur)

        if action == 14:
            start, duration = self.yaw_right(settings.yaw_rate_1_1, settings.rot_dur)
        if action == 15:
            start, duration = self.yaw_right(settings.yaw_rate_1_2, settings.rot_dur)
        if action == 16:
            start, duration = self.yaw_right(settings.yaw_rate_1_4, settings.rot_dur)
        if action == 17:
            start, duration = self.yaw_right(settings.yaw_rate_1_8, settings.rot_dur)

        if action == 18:
            start, duration = self.yaw_right(settings.yaw_rate_2_1, settings.rot_dur)
        if action == 19:
            start, duration = self.yaw_right(settings.yaw_rate_2_2, settings.rot_dur)
        if action == 20:
            start, duration = self.yaw_right(settings.yaw_rate_2_4, settings.rot_dur)
        if action == 21:
            start, duration = self.yaw_right(settings.yaw_rate_2_8, settings.rot_dur)
        '''

        collided = self.client.simGetCollisionInfo().has_collided

        return collided
