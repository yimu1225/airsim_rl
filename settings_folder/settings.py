import os
import settings_folder.machine_dependent_settings as mds

# ---------------------------
# imports 
# ---------------------------
# Augmenting the sys.path with relavant folders
settings_file_path = os.path.realpath(__file__)
settings_dir_path = os.path.dirname(settings_file_path)
proj_root_path = os.path.abspath(settings_dir_path + "/..")
os.sys.path.insert(0, proj_root_path)

# used for game configuration handling
# change it to adapt to your computer
json_file_addr = mds.json_file_addr

# used for start, restart and killing the game
game_file = mds.game_file
unreal_host_shared_dir = mds.unreal_host_shared_dir
unreal_exec = mds.unreal_exe_path

# ---------------------------
# range
# ---------------------------

# easy_range_dic = { "End": ["Mutable"],
#                                       "MinimumDistance": [5,8],
#                                       "EnvType": ["Indoor"],
#                                       "ArenaSize": [[85, 85, 10],[75, 75, 10],[80, 80, 10]],
#                                       "PlayerStart": [[0, 0, 0]],
#                                       "NumberOfDynamicObjects": list(range(0, 1)),
#                                       "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
#                                       "Seed": list(range(0, 10000)),
#                                       "VelocityRange": [[0, 2]],
#                                       "Name": ["Name"], 
#                                       "NumberOfObjects": list(range(5, 15))}
# medium_range_dic = { "End": ["Mutable"],
#                                       "MinimumDistance": [3,6],
#                                       "EnvType": ["Indoor"],
#                                       "ArenaSize": [[85, 85, 10],[75, 75, 10],[80, 80, 10]],
#                                       "PlayerStart": [[0, 0, 0]],
#                                       "NumberOfDynamicObjects": list(range(0, 1)),
#                                       "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
#                                       "Seed": list(range(0, 10000)),
#                                       "VelocityRange": [[0, 4]],
#                                       "Name": ["Name"],
#                                       "NumberOfObjects": list(range(35, 45))}
# hard_range_dic = { "End": ["Mutable"],
#                                       "MinimumDistance": [3,6],
#                                       "EnvType": ["Indoor"],
#                                       "ArenaSize": [[85, 85, 10],[75, 75, 10],[80, 80, 10]],
#                                       "PlayerStart": [[0, 0, 0]],
#                                       "NumberOfDynamicObjects": list(range(0, 1)),
#                                       "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
#                                       "Seed": list(range(0, 10000)),
#                                       "VelocityRange": [[0, 5]],
#                                       "Name": ["Name"],
#                                       "NumberOfObjects": list(range(45, 65))}
# dynamic_obstacles_dic = { "End": ["Mutable"],
#                                       "MinimumDistance": [2,5],
#                                       "EnvType": ["Indoor"],
#                                       "ArenaSize": [[85, 85, 10],[75, 75, 10],[80, 80, 10]],
#                                       "PlayerStart": [[0, 0, 0]],
#                                       "NumberOfDynamicObjects": list(range(5, 10)),
#                                       "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
#                                       "Seed": list(range(0, 10000)),
#                                       "VelocityRange": [[0, 1]],
#                                       "Name": ["Name"],
#                                       "NumberOfObjects": list(range(75, 85))}
easy_range_dic = { "End": ["Mutable"],
                                      "MinimumDistance": [2,3],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[70, 70, 10]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(0, 1)),
                                      "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[0, 2]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(60, 80))}
medium_range_dic = { "End": ["Mutable"],
                                      "MinimumDistance": [2,3],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[70, 70, 10]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(0, 1)),
                                      "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[0, 4]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(90, 110))}
hard_range_dic = { "End": ["Mutable"],
                                      "MinimumDistance": [2,3],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[70, 70, 10]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(0, 1)),
                                      "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[0, 5]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(120, 140))}
dynamic_obstacles_dic = { "End": ["Mutable"],
                                      "MinimumDistance": [2,3],
                                      "EnvType": ["Indoor"],
                                      "ArenaSize": [[70, 70, 10]],
                                      "PlayerStart": [[0, 0, 0]],
                                      "NumberOfDynamicObjects": list(range(5, 10)),
                                      "Walls1": [[200, 13, 99],[255, 255, 10],[0, 10, 10],[10, 100, 100],[126, 11, 90]],
                                      "Seed": list(range(0, 10000)),
                                      "VelocityRange": [[0.3, 1.0]],
                                      "Name": ["Name"],
                                      "NumberOfObjects": list(range(50, 70))}
default_range_dic = easy_range_dic
# ------------------------------------------------------------
#-game related-
# ------------------------------------------------------------
game_proc_pid = ''  # process associa

# ---------------------------
# sampling frequency (environmental randomization)
# ---------------------------
end_randomization_mode = "inclusive"  # whether each level of difficulty should be inclusive (including the previous level) or exclusive

"""
Environment change frequency: Specifies how often each environment parameter is randomized (every N episodes).
This controls domain randomization to improve policy generalization during training.

Each key-value pair indicates the randomization frequency after UE4 restart:

- ArenaSize: 竞技场尺寸每5个episode随机化一次，改变训练环境的物理空间大小
- Seed: 随机种子每5个episode更新一次，影响障碍物的随机位置生成
- NumberOfObjects: 障碍物数量每5个episode随机化，增加环境复杂度变化
- End: 目标点位置每3个episode随机化，使无人机学会适应不同目标
- Walls1: 墙壁配置每3个episode随机化，改变环境中的障碍物设置
- MinimumDistance: 最小距离要求每3个episode随机化，调整任务难度

这种随机化策略可以防止过拟合，提高策略对不同环境的泛化能力。
"""
environment_change_frequency = {
    "ArenaSize": 5,
    "Seed":3,
    "NumberOfObjects": 5,
    "NumberOfDynamicObjects": 5,
    "End": 3,
    "Walls1": 3,
    "MinimumDistance": 3
}

# environment_change_frequency = {
#     "ArenaSize": 5,
#     "Seed": 5,
#     "NumberOfObjects": 5,
#     "NumberOfDynamicObjects": 5,
#     "End": 3,
#     "Walls1": 3,
#     "MinimumDistance": 3
# }

# Whether to restart the UE4 game process when critical geometry params change.
# Set to False to use resetUnreal RPC for in-place environment reloading (much faster).
# Your UE4 project should support the resetUnreal RPC for this to work properly.
restart_game_on_param_change = False

# ------------------------------------------------------------
#                               -Drone related-
## ------------------------------------------------------------
#ip = '10.243.49.243'
ip = '127.0.0.1'
port = 41451  # AirSim API server port

# ---------------------------
# parameters
# ---------------------------
vel_duration=0.4
double_dqn = False
mv_fw_dur = 0.4
mv_fw_spd_1 = 1* 0.5
mv_fw_spd_2 = 2* 0.5
mv_fw_spd_3 = 3* 0.5
mv_fw_spd_4 = 4* 0.5
mv_fw_spd_5 = 5* 0.5
rot_dur = 0.1#0.15
# yaw_rate = (180/180)*math.pi #in degree
yaw_rate_1_1 = 108.  # FOV of front camera
yaw_rate_1_2 = yaw_rate_1_1 * 0.5  # yaw right by this angle
yaw_rate_1_4 = yaw_rate_1_2 * 0.5
yaw_rate_1_8 = yaw_rate_1_4 * 0.5
yaw_rate_1_16 = yaw_rate_1_8 * 0.5
yaw_rate_2_1 = -108.  # -2 time the FOV of front camera
yaw_rate_2_2 = yaw_rate_2_1 * 0.5  # yaw left by this angle
yaw_rate_2_4 = yaw_rate_2_2 * 0.5
yaw_rate_2_8 = yaw_rate_2_4 * 0.5
yaw_rate_2_16 = yaw_rate_2_8 * 0.5

# ---------------------------
# depth image noise (mild)
# ---------------------------
# Apply lightweight sensor noise augmentation on depth images (0-255 range).
enable_depth_noise = False
depth_gaussian_sigma = 10.0     # std of Gaussian noise (pixel intensity)
depth_salt_prob = 0.01        # probability of salt pixels
depth_pepper_prob = 0.01      # probability of pepper pixels


# ---------------------------
# general params
# ---------------------------
nb_max_episodes_steps = 512*3  # pay attention
success_distance_to_goal = 2
slow_down_activation_distance =  2*success_distance_to_goal  # detrmines at which distance we will punish the higher velocities

# ---------------------------
# reseting params
# ---------------------------
connection_count_threshold = 20  # the upper bound to try to connect to multirouter
restart_game_from_scratch_count_threshold = 3  # the upper bound to try to reload unreal from scratch
window_restart_ctr_threshold = 2  # how many times we are allowed to restart the window
# before easying up the randomization

#-------------------------------
#control mode
control_mode="Continuous" # "moveByVelocity" "Discrete" "Continuous_TD3"

#-------------------------------
#algorithm
algo = 'TD3'

#--------------------------------
# Unreal game settings
#--------------------------------
# 游戏窗口分辨率宽度（像素）
game_resX = 940
# 游戏窗口分辨率高度（像素）
game_resY = 700
# 游戏窗口在屏幕上的X位置（左上角坐标，居中显示）
ue4_winX = 400
# 游戏窗口在屏幕上的Y位置（左上角坐标，居中显示）
ue4_winY = 200
