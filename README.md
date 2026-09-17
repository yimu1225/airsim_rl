# AirSim 无人机自主导航强化学习框架

本项目基于 AirSim / Unreal Engine，研究深度视觉观测下的无人机自主导航。当前主要方法为 **VSSM-SAC（视觉状态空间记忆增强 SAC）** 和 **SSVM-SAC（自监督视觉记忆增强 SAC）**，同时保留 SAC、TD3、DDPG、PPO 等基线及对照算法。

## 两个主要方法

| 方法 | 中文名称 | 主要机制 | 实现位置 |
|---|---|---|---|
| VSSM-SAC | 视觉状态空间记忆增强 SAC | Vision Mamba + Temporal Mamba 编码固定窗口观测，结合 SB-PER，通过 SAC 联合训练 | [algorithm/VSSM/VSSM_SAC](algorithm/VSSM/VSSM_SAC/) |
| SSVM-SAC | 自监督视觉记忆增强 SAC | 分阶段学习视觉表示与因果递推记忆，再冻结感知模块训练 SAC | [algorithm/SSVM_SAC](algorithm/SSVM_SAC/) |

**SSVM-SAC 英文全称：Self-Supervised Visual Memory–Enhanced Soft Actor-Critic。** 自监督目标包含深度图重建、未来深度图预测和潜在表示监督；在线控制只使用当前及历史观测。

SSVM-SAC 的目录与配置标识为 `SSVM_SAC`，展示名称及结果标识为 `SSVM-SAC`。训练采用独立分阶段入口。完整说明见 [SSVM-SAC README](algorithm/SSVM_SAC/README.md)。

## VSSM-SAC 与消融算法

| 命令行名称 | 目录 | 视觉与时序表示 | 回放机制 |
|---|---|---|---|
| `VSSM-SAC` | `algorithm/VSSM/VSSM_SAC/` | Vision Mamba + Temporal Mamba | 成功/非成功双池 PER |
| `no-SB-PER` | `algorithm/VSSM/no_SB_PER/` | Vision Mamba + Temporal Mamba | 单池均匀采样 |
| `no-VSSM` | `algorithm/VSSM/no_VSSM/` | 逐帧 CNN 编码后拼接 | 成功/非成功双池 PER |
| `SAC` | `algorithm/SAC/` | 逐帧 CNN 编码后拼接，按当前默认配置 | 单池均匀采样 |

三个 VSSM 目录直接对应新算法名称，不再使用旧的消融别名映射。`CL-` 前缀表示启用课程学习。

当前完整方法的 Actor/Critic MLP 使用 SiLU，另三个对照使用 ReLU，初始化方式也不同。做严格消融时，应记录或统一这些差异；目录重组未修改训练计算逻辑。

## 当前算法与训练入口

当前名称注册表有 20 个算法标识。实际支持范围由各训练入口决定，不能将所有算法都交给 `main_async.py`。

| 入口 | 算法名称 |
|---|---|
| `main_async.py` | `DDPG`、`SDDPG`、`TD3`、`AETD3`、`VSSM-TD3` |
| `main_async.py` | `SAC`、`SAC_FAE`、`VSSM-SAC`、`no-SB-PER`、`no-VSSM` |
| `main_async.py` | `MM-VSSM-SAC`、`SVSSM-SAC`、`SAFE-VSSM-SAC`、`SB-PER-SVSSM-SAC`、`Transformer-SAC`、`SB-PER-MambaCSJA-SAC` |
| `main_ppo.py` | `PPO`、`VSSM-PPO` |
| `train_lstm_sac.py` | `LSTM-SAC` |
| `algorithm/SSVM_SAC/train.py` | SSVM-SAC 分阶段训练 |

算法选择只接受具体名称。多个算法用逗号分隔，按填写顺序执行；已取消算法组。

## 安装与环境

安装依赖前，先配置 AirSim 场景、UE 可执行文件和项目环境路径。视觉状态空间模块还依赖 PyTorch、CUDA、Mamba 和 causal-conv1d 的兼容安装。

- [安装指南](INSTALL_GUIDE.md)
- [Ubuntu 22.04 构建方法](Ubuntu%2022.04%20构建方法.md)
- [Unreal/AirSim 编译指南](README_Compilation_Guide.md)
- [场景修改说明](README_AirLearningArenaMeshes_Modification.md)
- [Python 依赖文件](requirements.txt)

```bash
conda activate AirSim
python main_async.py --help
python algorithm/SSVM_SAC/train.py --help
```

训练和环境评估需要可用的 AirSim 服务；Mamba 的完整执行还需要匹配的运行环境。离线视觉/记忆训练使用已采集的数据集。

## 训练

以下命令从项目根目录执行。

```bash
# 单算法
python main_async.py --algorithm_name VSSM-SAC --max_timesteps 150000

# 四个对照算法，均启用课程学习
python main_async.py \
  --algorithm_name "CL-VSSM-SAC,CL-no-SB-PER,CL-no-VSSM,CL-SAC" \
  --seed 25 --max_timesteps 150000

# 多随机种子
python main_async.py --algorithm_name CL-VSSM-SAC --seed "25,26,27"

# PPO 使用独立入口
python main_ppo.py --algorithm_name PPO

# LSTM-SAC 使用独立入口
python train_lstm_sac.py --algorithm_name LSTM-SAC
```

`config.py` 中当前默认算法列表为：

```python
default='CL-VSSM-SAC, CL-no-SB-PER, CL-no-VSSM, CL-SAC'
```

SSVM-SAC 按 `bootstrap → collect → vision → memory → sac` 五个阶段执行，完整可运行命令见[算法说明](algorithm/SSVM_SAC/README.md#训练命令)。

## 配置与课程学习

通用参数由 [config.py](config.py) 定义。通用训练入口再通过 [config_loader.py](algorithm/config_loader.py) 加载所选算法的 `params.yaml`，其中同名算法参数会覆盖命令行命名空间中的值。

SSVM-SAC 使用自己的 [config.py](algorithm/SSVM_SAC/config.py) 和 [params.yaml](algorithm/SSVM_SAC/params.yaml)，并将未识别的环境参数转交项目全局配置解析器。其模型结构还会从上游 checkpoint 恢复，详见算法 README。

```bash
# 按训练进度调整环境难度
python main_async.py --algorithm_name CL-VSSM-SAC --curriculum_mode progress

# 按成功率调整环境难度
python main_async.py --algorithm_name CL-VSSM-SAC --curriculum_mode success

# 固定难度
python main_async.py --algorithm_name VSSM-SAC --non_curriculum_level 2
```

SSVM-SAC 的独立入口用 `--curriculum` / `--no-curriculum` 控制课程学习；最终 `sac` 阶段默认启用，启用时要求 `--curriculum_mode progress`。

## 环境与观测

环境主体位于 [AirGym.py](gym_airsim/envs/AirGym.py)。观测字典包含深度图 `depth`、基础状态 `base` 和距离传感器信息 `distance_sensor`；算法是否使用某一观测字段由其实现决定。

- 通用视觉序列方法通过 `n_frames` 设置观测窗口。
- SSVM-SAC 使用 [AirGymSSVM.py](gym_airsim/envs/AirGymSSVM.py)，每步输入一张深度图，由记忆缓存维持历史。
- SSVM-SAC 数据采集可用 `--clean-targets` 额外保存干净深度监督。
- 动作范围、奖励项、回合终止和场景随机化以环境代码及配置为准。

## 评估与曲线

```bash
# 通用异步算法评估
python -m eval.eval_async \
  --algorithm_name CL-VSSM-SAC --seed 25 \
  --load_model models/CL-VSSM-SAC/seed25/async_final.pth

# 训练曲线
python plot_curves.py \
  --algorithm_name "VSSM-SAC,no-SB-PER,no-VSSM,SAC" \
  --plot_show_cl_prefix

tensorboard --logdir results --port 6007
```

PPO 与 LSTM-SAC 分别使用 `eval/eval_ppo.py` 和 `eval/eval_lstm_sac.py`。SSVM-SAC 使用自身的 `eval` 子命令。场景评估说明见 [eval/README.md](eval/README.md)。

通用结果和模型按算法、随机种子区分：

```text
results/<algorithm>/seed<seed>/
models/<algorithm>/seed<seed>/
```

SSVM-SAC 自动生成的曲线目录为 `results/CL-SSVM-SAC/seed<seed>/` 或非课程版本 `results/SSVM-SAC/seed<seed>/`。

## 项目结构

```text
algorithm/
├── VSSM/
│   ├── VSSM_SAC/        # VSSM-SAC
│   ├── no_SB_PER/       # 去掉 SB-PER
│   └── no_VSSM/         # 去掉 VSSM
├── SSVM_SAC/            # 自监督视觉记忆增强 SAC
├── SAC/                 # 基础 SAC
├── SAC_FAE/
├── LSTM_SAC/
├── TD3/
├── DDPG/
├── PPO/
└── config_loader.py     # 算法参数加载，其他保留算法见上表

gym_airsim/              # 环境及 AirSim 接口
environment_randomization/ # 场景随机化
game_handling/          # 仿真进程管理
settings_folder/        # 仿真与环境设置
eval/                   # 评估及可视化
explainability_aosa/    # 策略解释分析
tests/                  # 测试与环境诊断
main_async.py           # 通用异步训练入口
main_ppo.py             # PPO 训练入口
train_lstm_sac.py       # LSTM-SAC 训练入口
plot_curves.py          # 曲线绘制
algo_name_utils.py      # 算法名称解析，不再提供分组
```

## 实验记录与验证边界

实验记录应包含代码版本、算法参数、随机种子、课程设置、训练步数、场景条件和 checkpoint。SSVM-SAC 还需记录数据采集策略、是否使用干净深度监督、视觉与记忆预训练 checkpoint。

本轮目录清理已检查 Python 语法、算法导入路径和参数文件加载；这不等于完成 GPU/AirSim 训练验证。论文中的性能结论应来自实际训练与统一条件下的评估。
