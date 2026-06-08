# AirSim 强化学习无人机导航框架

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![AirSim](https://img.shields.io/badge/AirSim-1.8.1-lightgrey.svg)](https://microsoft.github.io/AirSim/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1-red.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于 Microsoft AirSim 的无人机自主导航深度强化学习框架，实现并统一了 **45+ 种** DRL 算法变体，覆盖 DDPG / TD3 / SAC / PPO 四大算法家族，支持注意力机制视觉编码、序列建模、优先经验回放、课程学习等前沿技术。

## ⚠️ 免责声明

**本仓库为作者的个人学习项目，主要用于强化学习算法研究和技术积累。代码和实现仅供参考学习，不保证生产环境适用性。欢迎交流学习，但请尊重作者劳动成果。**

---

## 🚁 项目概述

- **高保真物理模拟** — 基于 Microsoft AirSim + Unreal Engine 的无人机物理仿真
- **多模态观测** — 融合深度图像、状态向量、距离传感器阵列的 Dict 观测空间
- **四大算法家族** — 统一实现 DDPG / TD3 / SAC / PPO 及其大量变体
- **视觉编码器** — 支持 CNN、Vision Mamba (Vim)、VideoMamba、Dual-Branch Mamba 等架构
- **时序建模** — 支持 LSTM、Temporal Mamba、状态空间模型 (SSM) 等序列处理方法
- **优先经验回放** — 内置 PER (Prioritized) 与 DPER (Demonstration + Prioritized) 机制
- **课程学习** — 通过 `CL-` 前缀一键启用，支持 progress / success 两种模式
- **特权学习 (Privileged Learning)** — `PL_` 前缀算法支持干净深度图作为特权信息
- **环境随机化** — 四档难度等级，动态障碍物，确定性可复现采样
- **健壮的游戏管理** — 自动 UE4 进程重启、健康检测、窗口状态监控

---

## 🧠 算法全景

### 算法命名规则

| 前缀/后缀 | 含义 |
|-----------|------|
| `CL-` | 课程学习 (Curriculum Learning) |
| `PL_` | 特权学习 (Privileged Learning)，Actor 可访问干净深度图 |
| `DPER_` | 示范数据 + 优先经验回放 (Demonstration PER) |
| `PER_` | 优先经验回放 (Prioritized Experience Replay) |
| `VM` | Vision Mamba 视觉编码器 |
| `SVM` | 状态分解 + Vision Mamba (State-Decomposed VM) |
| `SAFE_` | 带安全约束的变体 |
| `MM_` | 多模态变体 |
| `_Beta` | Beta 分布策略变体 |
| `Mamba_` / `MambaCSJA_` | Mamba / CSJA-Mamba 序列编码器 |

> 示例：`CL-DPER_SVMSAC` = 课程学习 + 示范优先回放 + 状态分解 Vision Mamba + SAC

### 算法家族

#### 1. DDPG 家族
| 算法 | 特点 |
|------|------|
| **DDPG** | 深度确定性策略梯度，基准算法 |
| **SDDPG** | 状态分解 DDPG (State-Decomposition)，解耦位置/速度子空间 |

#### 2. TD3 家族
| 算法 | 视觉编码 | 时序处理 | 亮点 |
|------|----------|----------|------|
| **TD3** | CNN | 帧堆叠 | 标准 Twin Delayed DDPG |
| **DPER_TD3** | CNN | 帧堆叠 | TD3 + 示范优先回放 |
| **AETD3** | CNN | 帧堆叠 | 自适应集成 Critic |
| **VMTD3** | Vision Mamba | Temporal Mamba | 纯 Mamba 时空建模 |
| **Vim_TD3** | Vision Mamba | 无 | 纯 Vim 特征提取 |
| **ST_Seq_Vim_TD3** | Vision Mamba | Temporal Mamba | 状态-视觉双流时空 |
| **STV_Seq_Vim_TD3** | Vision Mamba | Temporal Mamba | 视觉-状态-视觉三流融合 |
| **STV_Patch_TD3** | Vision Mamba | Temporal Mamba | Video-style Patch Embedding |
| **ST_DualVim_TD3** | Dual-Branch VM | Temporal Mamba | 双分支视频 Mamba |
| **Mamba_TD3** | CNN | Temporal Mamba | CNN + Mamba 混合 |
| **DPER_VMTD3** | Vision Mamba | Temporal Mamba | VMTD3 + 示范优先回放 |
| **SAFE_VMTD3** | Vision Mamba | Temporal Mamba | 带安全约束的 VMTD3 |

#### 3. SAC 家族（最大）
| 算法 | 视觉编码 | 时序处理 | 亮点 |
|------|----------|----------|------|
| **SAC** | CNN | 帧堆叠 | 标准 Soft Actor-Critic |
| **SAC_Beta** | CNN | 帧堆叠 | Beta 分布替代高斯 |
| **LSTM_SAC** | CNN | LSTM | 循环神经网络时序建模 |
| **VMSAC** | Vision Mamba | Temporal Mamba | Mamba 时空 + SAC |
| **VMSAC_Beta** | Vision Mamba | Temporal Mamba | VMSAC + Beta 分布 |
| **SVMSAC** | Vision Mamba | Temporal Mamba | **状态分解** VMSAC |
| **PER_VMSAC** | Vision Mamba | Temporal Mamba | VMSAC + 优先回放 |
| **DPER_VMSAC** | Vision Mamba | Temporal Mamba | VMSAC + 示范优先回放 |
| **DPER_VMSAC_Beta** | Vision Mamba | Temporal Mamba | DPER_VMSAC + Beta 分布 |
| **DPER_SVMSAC** | Vision Mamba | Temporal Mamba | 状态分解 + 示范优先回放 |
| **MM_VMSAC** | Vision Mamba | Temporal Mamba | 多模态 VMSAC |
| **SAFE_VMSAC** | Vision Mamba | Temporal Mamba | 带安全约束的 VMSAC |
| **Mamba_SAC** | CNN | Temporal Mamba | CNN + Mamba 混合 SAC |
| **PER_Mamba_SAC** | CNN | Temporal Mamba | Mamba_SAC + 优先回放 |
| **MambaCSJA_SAC** | CNN | CSJA-Mamba | Mamba + 通道-空间联合注意力 |
| **DPER_MambaCSJA_SAC** | CNN | CSJA-Mamba | MambaCSJA + 示范优先回放 |
| **Mamba_RSAC** | CNN | Temporal Mamba | Mamba + 循环 SAC |

#### 4. PPO 家族
| 算法 | 视觉编码 | 时序处理 | 亮点 |
|------|----------|----------|------|
| **PPO** | CNN | 帧堆叠 | 标准 Proximal Policy Optimization |
| **VMPPO** | Vision Mamba | Temporal Mamba | PPO + Mamba 时空编码 |
| **PL_VMPPO** | Vision Mamba | Temporal Mamba | 特权学习 VMPPO |

#### 5. 特权学习 (PL) 变体
所有 PL 前缀算法允许 Actor 访问无噪声的"干净"深度图作为特权观测，Critic 仍使用带噪声的常规观测：

`PL_TD3` · `PL_DPER_TD3` · `PL_VMTD3` · `PL_DPER_VMTD3` · `PL_SAC` · `PL_SAC_Beta` · `PL_VMSAC` · `PL_PER_VMSAC` · `PL_DPER_VMSAC` · `PL_DPER_VMSAC_Beta` · `PL_Mamba_RSAC`

---

## 🎯 环境设计

### 观测空间 (Dict)

```python
observation_space = {
    "depth":       (n_frames, H, W),        # 深度图像序列 (带噪声)
    "base":        (11,),                   # 状态向量 (见下方说明)
    "distance_sensor": (108,),              # 3层距离传感器阵列 (每层36个)
    # ↓ 仅 PL_ 前缀算法额外提供
    "clean_depth": (n_frames, H, W),        # 干净深度图 (特权信息)
}
```

**状态向量 (11维)**：
- `[dx, dy, dz]` — 相对目标位置
- `[body_x_velocity, body_y_velocity, z_velocity]` — 机体速度
- `[yaw_rate]` — 偏航角速度
- `[yaw]` — 当前偏航角
- `[relative_angle_to_target]` — 朝向目标的相对角度
- `[altitude]` — 当前高度
- `[collision]` — 碰撞标志

### 动作空间 (Continuous)

```python
action_space = Box(
    low  = [-2.0, -π/3, -0.3],
    high = [2.0,  π/3,  0.3],
)
```

| 维度 | 含义 | 范围 |
|------|------|------|
| `body_x_velocity` | 机体系 x 轴速度 | [-2.0, 2.0] m/s |
| `yaw_rate` | 偏航角速度 | [-π/3, π/3] rad/s |
| `z_velocity` | 垂直速度 | [-0.3, 0.3] m/s |

### 奖励函数

| 奖励项 | 设计 | 目的 |
|--------|------|------|
| **距离奖励** | `-distance × 0.02` | 鼓励接近目标 |
| **朝向奖励** | `speed × cos(yaw_error)` | 鼓励朝向目标飞行 |
| **成功奖励** | `+20` | 到达目标点 |
| **碰撞惩罚** | `-20` | 避免碰撞 |
| **超时惩罚** | `-30` | 惩罚超时未到达 |
| **步数惩罚** | `-0.1` | 鼓励尽快完成 |
| **急动惩罚** | 动作变化量 | 提高飞行平稳性 |
| **曲率惩罚** | 偏航率变化 | 优化轨迹平滑度 |
| **高度惩罚** | 超出 [0, 2.5]m 时 | 保持安全飞行高度 |
| **停滞惩罚** | 滑动窗口位移过小 | 防止悬停不动 |
| **距离传感器惩罚** | 对数距离惩罚 | 近距障碍物避障 |

### 难度等级

| Level | 名称 | 描述 |
|-------|------|------|
| 0 | Easy | 简单环境，稀疏静态障碍物 |
| 1 | Medium | 中等难度，较多静态障碍物 |
| 2 | Hard | 困难环境，密集静态障碍物 |
| 3 | Dynamic | 动态障碍物，最高难度 |

---

## 🚀 快速开始

### 环境要求

- Ubuntu 22.04 / Windows 10+
- Python 3.9+
- CUDA 12.8 (GPU 训练推荐)
- PyTorch 2.7.0
- Unreal Engine 4.27+ (AirSim 依赖)
- AirSim 1.8.1

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/yimu1225/airsim_rl.git
cd airsim_rl

# 2. 创建 Conda 环境
conda create -n AirSim python=3.9 -y
conda activate AirSim

# 3. 安装 CUDA 工具链
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit=12.8 cuda-nvcc=12.8 -y

# 4. 安装 PyTorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 5. 安装项目依赖
pip install -r requirements.txt

# 6. 编译 Mamba 组件 (可选 — 仅 Mamba 系列算法需要)
cd Vim/mamba-1p1p1
pip install -e .
cd ../..
```

> 详细安装指南参见 [INSTALL_GUIDE.md](INSTALL_GUIDE.md) 和 [Ubuntu 22.04 构建方法](Ubuntu%2022.04%20构建方法.md)

### 训练

```bash
# 单算法训练 (推荐使用算法组名)
python main_async.py --algorithm_name VMSAC --max_timesteps 1000000

# 训练带课程学习的版本
python main_async.py --algorithm_name CL-VMSAC --max_timesteps 1000000

# 训练状态分解版本
python main_async.py --algorithm_name SVMSAC --max_timesteps 1000000

# 批量训练 — 使用算法组
python main_async.py --algorithm_name base --max_timesteps 500000   # 基础算法组
python main_async.py --algorithm_name seq  --max_timesteps 500000   # 时序算法组
python main_async.py --algorithm_name all  --max_timesteps 500000   # 全部算法

# 手动指定多个算法
python main_async.py --algorithm_name "VMSAC,SVMSAC,VMTD3" --max_timesteps 500000

# 多种子训练
python main_async.py --algorithm_name VMSAC --seed "1,2,3" --max_timesteps 1000000
```

### 评估

```bash
# 评估训练好的模型
python eval_SAC.py --model_dir results/AirSimEnv-v42/VMSAC/run1/models --algorithm_name VMSAC

# 指定评估回合数
python eval_SAC.py --model_dir path/to/model --algorithm_name VMSAC --eval_episodes 100
```

---

## ⚙️ 核心配置

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algorithm_name` | `CL-DPER_SVMSAC` | 算法名称，支持逗号分隔多算法、算法组名 |
| `--seed` | `1,2,3` | 随机种子，逗号分隔多种子 |
| `--max_timesteps` | `120000` | 总训练步数 |
| `--hidden_dim` | `128` | 隐藏层维度 |
| `--base_feature_dim` | `32` | 状态向量映射维度 |
| `--batch_size` | `256` | 批次大小 |
| `--buffer_size` | `30000` | 经验回放池大小 |
| `--gamma` | `0.95` | 折扣因子 |
| `--tau` | `0.003` | 目标网络软更新系数 |
| `--actor_lr` / `--critic_lr` | `4e-4` | Actor/Critic 学习率 |
| `--n_frames` | `4` | 输入帧数 (堆叠帧数 / 序列长度) |
| `--episode_length` | `300` | 每回合最大步数 |
| `--action_duration` | `0.15` | 动作执行时间 (秒) |

### 算法专属参数

每个算法在 `algorithm/<算法名>/params.yaml` 中定义专属参数，由 `config_loader` 自动加载。例如 `algorithm/VMSAC/params.yaml` 定义 Vision Mamba 编码器的嵌入维度、深度、patch 大小等。

### 课程学习配置

```bash
# progress 模式 — 随训练进度连续增加难度
python main_async.py --algorithm_name CL-VMSAC --curriculum_mode progress --curriculum_progress_max_ratio 0.9

# success 模式 — 按成功率离散切换难度
python main_async.py --algorithm_name CL-VMSAC --curriculum_mode success

# 指定起始难度
python main_async.py --algorithm_name CL-VMSAC --curriculum_start_level 1

# 非课程学习的固定难度
python main_async.py --algorithm_name VMSAC --non_curriculum_level 2
```

---

## 📁 项目结构

```
airsim_rl/
├── algorithm/                     # 算法实现 (45+ 变体)
│   ├── config_loader.py           # 算法参数自动加载
│   ├── DDPG/                      #   DDPG
│   ├── SDDPG/                     #   状态分解 DDPG
│   ├── TD3/                       #   TD3 基准
│   ├── AETD3/                     #   自适应集成 TD3
│   ├── DPER_TD3/                  #   示范优先回放 TD3
│   ├── VMTD3/                     #   Vision Mamba TD3
│   ├── Vim_TD3/                   #   纯 Vim TD3
│   ├── ST_Seq_Vim_TD3/            #   状态-视觉双流 TD3
│   ├── STV_Seq_Vim_TD3/           #   视觉-状态-视觉三流 TD3
│   ├── STV_Patch_TD3/             #   Video Patch TD3
│   ├── ST_DualVim_TD3/            #   双分支 Mamba TD3
│   ├── Mamba_TD3/                 #   CNN + Mamba TD3
│   ├── DPER_VMTD3/                #   示范优先回放 VMTD3
│   ├── SAFEVMTD3/                 #   安全约束 VMTD3
│   ├── SAC/                       #   SAC 基准
│   ├── SAC_Beta/                  #   Beta 分布 SAC
│   ├── LSTM_SAC/                  #   LSTM 时序 SAC
│   ├── VMSAC/ / VMSAC_Beta/       #   Vision Mamba SAC
│   ├── SVMSAC/                    #   状态分解 VMSAC
│   ├── PER_VMSAC/                 #   优先回放 VMSAC
│   ├── DPER_VMSAC/                #   示范优先回放 VMSAC
│   ├── DPER_VMSAC_Beta/           #   示范优先回放 VMSAC Beta
│   ├── DPER_SVMSAC/               #   状态分解 + 示范优先回放
│   ├── MM_VMSAC/                  #   多模态 VMSAC
│   ├── SAFEVMSAC/                 #   安全约束 VMSAC
│   ├── Mamba_SAC/ / PER_Mamba_SAC/ # Mamba SAC 系列
│   ├── Mamba_RSAC/                #   Mamba 循环 SAC
│   ├── MambaCSJA_SAC/             #   Mamba + 通道空间注意力 SAC
│   ├── DPER_MambaCSJA_SAC/        #   示范优先回放 MambaCSJA SAC
│   ├── PPO/                       #   PPO 基准
│   ├── VMPPO/                     #   Vision Mamba PPO
│   ├── PL_TD3/ ... PL_DPER_VMTD3/ # 特权学习 TD3 系列
│   ├── PL_SAC/ ... PL_Mamba_RSAC/ # 特权学习 SAC 系列
│   └── PL_VMPPO/                  #   特权学习 PPO
│
├── gym_airsim/                    # AirSim Gymnasium 环境
│   └── envs/
│       └── AirGym.py              # 环境主实现
│
├── Vim/                           # Vision Mamba 核心库
│   └── mamba-1p1p1/               # Mamba 编译包
├── vmamba/                        # VMamba 实现
├── VideoMamba/                    # VideoMamba 实现
│
├── common/                        # 通用工具
│   ├── utils.py                   # 工具函数
│   └── file_handling.py           # 文件处理
│
├── sb3_algorithms/                # Stable-Baselines3 包装器
│   ├── config_loader.py           # SB3 参数加载
│   ├── params/                    # SB3 算法参数
│   ├── td3_wrappers.py            # TD3/DDPG 包装
│   ├── sac_wrappers.py            # SAC 包装
│   └── ppo_wrappers.py            # PPO 包装
│
├── sb3_extensions/                # SB3 扩展组件
│   ├── buffers/                   # 自定义经验池
│   ├── feature_extractors/        # 特征提取器
│   ├── policies/                  # 自定义策略
│   └── callbacks/                 # 训练回调
│
├── environment_randomization/     # 环境参数随机化
├── game_handling/                 # UE4 游戏进程管理
├── scripts/                       # 辅助脚本
├── models/                        # 预训练模型
├── results/                       # 训练结果 & TensorBoard 日志
│
├── main_async.py                  # 主训练入口 (异步架构)
├── main_sb3.py                    # SB3 风格训练入口
├── main_ppo.py                    # PPO 专用训练入口
├── main_mamba_rsac.py             # Mamba RSAC 训练入口
├── train_lstm_sac.py              # LSTM-SAC 专用训练脚本
├── train_ppo.py                   # PPO 训练脚本
├── eval_SAC.py                    # 模型评估脚本
├── plot_curves.py                 # 训练曲线绘制
│
├── config.py                      # 全局配置 & 命令行参数
├── algo_name_utils.py             # 算法名解析 & 分组管理
├── requirements.txt               # Python 依赖列表
│
├── INSTALL_GUIDE.md               # 详细安装指南
└── Ubuntu 22.04 构建方法.md        # Ubuntu 构建文档
```

---

## 🔧 开发指南

### 添加新算法

1. 在 `algorithm/` 下创建算法目录 (如 `algorithm/MyAlgo/`)
2. 编写 `network.py` — 定义 Actor / Critic 网络
3. 编写 `agent.py` — 实现 `select_action`、`update`、`save` / `load` 等接口
4. 添加 `params.yaml` — 算法专属超参数
5. 在 `algo_name_utils.py` 的 `_CANONICAL_ALGORITHMS` 中注册算法名
6. 在 `main_async.py` 中 import agent 类并注册到 agent 分发表

### 修改环境

1. 编辑 `gym_airsim/envs/AirGym.py` — 调整观测/动作空间、奖励函数
2. 修改 `environment_randomization/` 中的配置 — 调整环境随机化参数
3. 更新 `settings_folder/` — 调整 AirSim settings

---

## 📊 训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./results --port=6007

# 绘制训练曲线
python plot_curves.py --algorithm_name VMSAC
```

---

## 🐛 常见问题

<details>
<summary><b>AirSim 连接失败</b></summary>

```bash
# 检查 AirSim 是否正确启动
# 确认 IP 和端口配置
python main_async.py --airsim_ip 127.0.0.1 --airsim_port 41451

# 尝试禁用游戏重启，仅重连
python main_async.py --disable_game_restart
```
</details>

<details>
<summary><b>CUDA 内存不足 (OOM)</b></summary>

```bash
# 减小 batch size
python main_async.py --batch_size 64

# 减小经验池
python main_async.py --buffer_size 10000
```
</details>

<details>
<summary><b>Mamba 编译错误</b></summary>

```bash
# 确保安装了正确的 CUDA 版本
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit=12.8 cuda-nvcc=12.8 -y
cd Vim/mamba-1p1p1
pip install -e . --verbose
```
</details>

---

## 📄 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

---

## 📚 参考文献

- **TD3**: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- **SAC**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
- **DDPG**: [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)
- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **PER**: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- **DPER**: [Deep RL with a Small Amount of Expert Demonstrations](https://arxiv.org/abs/1910.09457)
- **Mamba**: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- **Vision Mamba**: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional SSM](https://arxiv.org/abs/2401.13666)
- **VideoMamba**: [VideoMamba: State Space Model for Efficient Video Understanding](https://arxiv.org/abs/2403.06977)
- **State Decomposition DDPG**: [A State-Decomposition DDPG Algorithm for UAV Autonomous Navigation](https://ieeexplore.ieee.org/)
- **AirSim**: [AirSim: High-Fidelity Visual and Physical Simulation for UAVs](https://arxiv.org/abs/1705.09530)
