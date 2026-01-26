# AirSim RL PPO 训练项目

本项目专注于使用 PPO (Proximal Policy Optimization) 算法在 AirSim 环境中训练无人机自主寻路。

## 运行 `train_ppo.py` 的必要文件清单

如果您只想运行 PPO 训练，项目中仅以下文件和目录是必需的：

### 核心脚本
- `train_ppo.py`: 训练的主入口程序。
- `config.py`: 训练超参数配置（如学习率、PPO轮次、折扣因子等）。
- `EnvGenConfig.json`: 环境生成的配置文件（包括目标点坐标等）。
- `msgs.py`: 用于存储和传递全局状态信息。

### 核心模块
- `algorithm/`: PPO 算法的具体实现。
  - `ppo.py`: PPO 更新逻辑。
  - `model.py`: Actor-Critic 网络结构。
- `gym_airsim/`: 适配 OpenAI Gym 的 AirSim 环境封装。
  - `envs/AirGym.py`: 环境核心交互逻辑。
  - `envs/airlearningclient.py`: AirSim API 调用及兼容性层。
- `settings_folder/`: 项目全局设置。
  - `settings.py`: 环境控制模式、分辨率等全局配置。
  - `machine_dependent_settings.py`: **重要**，需要在此配置您本机的 Unreal 运行路径。
- `utils/`: 强化学习常用的工具类。
  - `storage.py`: 经验回放缓冲区 (RolloutStorage)。
  - `distributions.py`: 动作选择的概率分布。
  - `util.py`: 学习率调度、网络初始化等辅助函数。
- `common/`: 通用工具辅助。
  - `utils.py`: 坐标转换、进程查找等。
  - `file_handling.py`: JSON 文件读写。
- `environment_randomization/`: 处理 Unreal 环境配置。
  - `game_config_handler_class.py`: 管理环境配置文件。
  - `game_config_class.py`: 封装配置数据结构。
- `game_handling/`: 游戏进程管理。
  - `game_handler_class.py`: 自动启动和重启 Unreal 游戏。

---

## 运行指南

### 1. 配置环境路径
在运行之前，请务必修改 [settings_folder/machine_dependent_settings.py](settings_folder/machine_dependent_settings.py) 中的以下路径，以匹配您的计算机环境：
- `unreal_exe_path`: 指向您的 Unreal 模拟器可执行文件 (.exe)。
- `game_file`: 指向您的 Unreal 游戏项目文件或编译后的包。

### 2. 准备 Unreal 环境
确保您的 Unreal 项目中已安装 [AirSim 插件](https://github.com/microsoft/AirSim)，并且已手动启动或配置好自动启动。

### 3. 开始训练
在终端中运行以下命令：
```bash
python train_ppo.py
```
*(如果遇到 CUDA 报错，可以尝试添加前缀 `CUDA_LAUNCH_BLOCKING=1` 进行调试)*

### 4. 查看训练进度
训练日志将保存在 `results/` 目录下。您可以使用 Tensorboard 查看：
```bash
tensorboard --logdir=results
```

---

## 迁移与清理说明

1.  **移除 Baselines 依赖**：项目已完全移除 `baselines` 源码包。所有必要的并行环境包装器（如 `utils/env_wrappers.py` 和 `mujoco_envs/utils/env_wrappers.py`）均已迁移至使用更现代的 `stable_baselines3` (SB3)。
2.  **代码整理**：
    *   **[object_detection_parts/](object_detection_parts/)**：集中存放了所有与 3D 目标检测（SMOKE）相关的脚本和数据。
    *   **根目录**：保留了所有算法脚本（PPO, SAC, DQN, Rainbow 等），它们现在都已配合 SB3 库运行。
3.  **核心训练**：当前 `train_ppo.py` 使用的是 `algorithm/` 目录下的 PyTorch 实现，验证已完全脱离旧包依赖。
