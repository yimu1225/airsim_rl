# AirSim RL 导航

本项目在 AirSim 模拟器中实现无人机导航的强化学习算法。它支持各种 TD3 变体，具有不同的架构来处理视觉和状态输入。

## 概述

环境使用 AirSim 进行逼真的无人机模拟。智能体学习从起始位置导航到目标位置，同时避开障碍物并保持安全高度。

## 算法

### 状态输入处理

所有算法处理以下观测空间：
- **深度图像**: (stack_frames, 128, 128) - 深度相机输入
- **灰度图像**: (stack_frames, 128, 128) - 灰度相机输入
- **基础状态**: (10,) - 物理状态向量 [dx, dy, altitude, v_xy, v_z, yaw_rate, pitch, roll, yaw, relative_angle]

#### 非循环算法 (stack_frames=4)
- **TD3**: 标准双延迟 DDPG
  - 输入: 通过 CNN + MLP 处理的堆叠帧（4帧）
  - 状态处理: 图像的 CNN，与基础状态连接

- **AETD3**: 自适应集成 TD3
  - 输入: 与 TD3 相同
  - 状态处理: 与 TD3 相同，具有集成评论家

- **PER TD3**: 优先经验回放 TD3
  - 输入: 与 TD3 相同
  - 状态处理: 与 TD3 相同，具有优先缓冲区采样

#### 循环算法 (stack_frames=1, sequence_length=8)
- **GRU TD3**: 门控循环单元 TD3
  - 输入: 8 步序列: depth(1,128,128), gray(1,128,128), base(9)
  - 状态处理: 每个时间步的 CNN，然后在序列上使用 GRU

- **LSTM TD3**: 长短期记忆 TD3
  - 输入: 与 GRU TD3 相同
  - 状态处理: 每个时间步的 CNN，然后在序列上使用 LSTM

- **CFC TD3**: 闭式连续时间 TD3
  - 输入: 与 GRU TD3 相同
  - 状态处理: 每个时间步的 CNN，然后在序列上使用 CFC

#### LGMD 变体（运动检测）
LGMD（Lobula Giant Movement Detector）添加运动敏感处理：
- **LGMD GRU/LSTM/CFC TD3**: 将 LGMD 与循环网络结合
  - 输入: LGMD 处理的额外灰度运动历史
  - 状态处理: LGMD 模块 + 循环网络

### Actor 网络输出

所有算法输出范围内的连续动作：
- **前进速度**: [min_forward_speed, max_forward_speed] (默认: [0.0, 2.0] m/s)
- **偏航率**: [-max_yaw_rate, max_yaw_rate] (默认: [-1.0, 1.0] rad/s)
- **垂直速度**: [-max_vertical_speed, max_vertical_speed] (默认: [-0.3, 0.3] m/s)

Actor 网络输出被裁剪到这些边界范围内的原始值。

## 奖励函数

- **基于距离**: -distance * 0.03（鼓励接近目标）
- **朝向**: 当面向目标时 +speed * cos(yaw_error)
- **成功**: 到达目标时 +20
- **碰撞**: 碰撞时 -20
- **超时**: 情节超过最大步数时 -20
- **步数惩罚**: 每步 -0.01（鼓励效率）
- **急动惩罚**: 惩罚突然的动作变化
- **高度惩罚**: 超出安全高度范围（0.5-3.5m）时 -0.5

## 安装

1. 安装 AirSim: 按照 https://microsoft.github.io/AirSim/build_linux/
2. 安装依赖: `pip install -r requirements.txt`
3. 为您的环境配置 AirSim 设置

## 使用方法

### 训练
```bash
python main_async.py --algorithm_name lstm_td3 --max_timesteps 1000000
```

### 支持的算法
- td3, aetd3, per_td3, per_aetd3
- gru_td3, lstm_td3, gru_aetd3, lstm_aetd3, cfc_td3
- lgmd_gru_td3, lgmd_lstm_td3, lgmd_gru_aetd3, lgmd_lstm_aetd3, lgmd_cfc_td3

### 配置
修改 `config.py` 来调整：
- 算法参数（学习率、缓冲区大小等）
- 环境设置（高度、速度、惩罚）
- 网络架构（隐藏维度、序列长度）

## 环境详情

- **观测空间**: 包含 'depth', 'gray', 'base' 键的字典
- **动作空间**: Box(3,) 连续动作
- **情节长度**: 最大 512 步
- **目标采样**: 在定义范围内随机
- **碰撞检测**: 通过 AirSim 物理引擎

## 主要特性

- 异步训练与批量更新
- 帧堆叠以获取时间信息
- 循环网络用于序列建模
- 使用 LGMD 的运动检测
- 全面的奖励塑造
- 可配置的超参数
