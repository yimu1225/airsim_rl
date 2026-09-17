# SSVM-SAC：自监督视觉记忆增强 SAC

**英文全称：Self-Supervised Visual Memory–Enhanced Soft Actor-Critic。**

SSVM-SAC 通过自监督目标学习视觉表示和因果时序记忆，再冻结感知模块训练 SAC，用于深度视觉观测下的无人机自主导航。

## 名称与实现

方法名称为 **SSVM-SAC**，代码目录为 `algorithm/SSVM_SAC/`，配置类为 `SSVMConfig`，智能体类为 `SSVMSACAgent`。训练、评估、可视化和曲线记录统一使用 SSVM 标识。

本文依据当前实现编写。默认参数来自训练入口加载的 `params.yaml`；已有 checkpoint 中保存的配置可能不同。

## 与 VSSM-SAC 的关系

| 对比项 | VSSM-SAC | SSVM-SAC |
|---|---|---|
| 每步视觉输入 | 固定窗口的多帧深度图 | 当前单帧深度图 |
| 历史表示 | 对当前窗口做时空编码 | 因果 Temporal Mamba 持续更新缓存 |
| 表示学习 | 随 SAC 训练更新编码器 | 先进行独立的视觉与记忆自监督训练 |
| 最终策略训练 | 更新编码器和 SAC 网络 | 冻结视觉编码器与记忆，只训练 SAC 网络 |
| 控制输入 | 编码特征与基础状态 | 记忆输出与基础状态 |
| 回放 | 成功/非成功双池 PER | 成功/非成功双池 PER，存储记忆特征 |

SSVM-SAC 采用分阶段训练：以 Vision Mamba 编码深度观测，以 Temporal Mamba 学习因果视觉记忆，再以 SAC 学习控制策略。未来观测只作为离线监督，不作为当前时刻记忆的输入。

## 模型结构

```text
当前深度图 I_t
    ↓ Vision Mamba Encoder
视觉 latent z_t
    ↓ Temporal Mamba(z_t, cache_{t-1})
记忆输出 m_t，更新后的 cache_t
    ├── 与基础状态 base_t 拼接 → SAC Actor / Twin Critic
    └── 线性投影 → 各时间偏移的 latent → 共享 Vision Mamba Decoder
                                              ↓
                                 过去、当前和未来深度图
```

控制状态为 `concat(base_t, m_t)`，不额外拼接 `z_t`。Mamba 的内部循环缓存用于下一步更新，策略接收的是记忆模块输出 `m_t`。回合开始时清空缓存。

默认结构：

| 参数 | `params.yaml` 默认值 | 含义 |
|---|---|---|
| `image_height` / `image_width` | 128 / 128 | 单通道深度图尺寸 |
| `patch_size` | 4 | 视觉 patch 尺寸 |
| `latent_dim` | 128 | 当前帧视觉表示维度 |
| `encoder_depth` | 2 | 视觉编码深度 |
| `memory_dim` | 1024 | 时序记忆输出维度 |
| `memory_depth` | 4 | Temporal Mamba 层数 |
| `d_state` | 16 | 视觉模块状态维度 |
| `memory_d_state` | 256 | 时序记忆状态维度 |
| `reconstruction_latent_dim` | 128 | 每个目标的重建 latent 维度 |
| `decoder_embed_dim` / `decoder_depth` | 64 / 2 | 解码器各尺度 token 宽度与块数 |
| `reconstruction_offsets` | `[-5, 0, 5]` | 过去、当前和未来监督偏移 |

默认投影为 `1024 → 3 × 128 = 384`。三个 latent 共享一套解码器。解码器从粗空间特征开始，通过线性 patch 展开和空间 Mamba 逐级细化；不使用视觉编码器 token 跳接。

## 五阶段训练

| 阶段 | 子命令 | 训练内容 | 冻结内容 | 主要产物 |
|---|---|---|---|---|
| 1 | `bootstrap` | 随机感知特征上的 SAC | 随机初始化的 Encoder、Memory | 采集策略 |
| 2 | `collect` | 运行策略，收集完整回合 | 全部网络 | `episode_*.npz` |
| 3 | `vision` | 单帧 Encoder 与 Decoder | 无 | 视觉自编码器 checkpoint |
| 4 | `memory` | Temporal Mamba 与目标 latent 投影 | 已训练的 Encoder、Decoder | 记忆 checkpoint |
| 5 | `sac` | 默认重新初始化的 Actor、Critic、熵系数 | Encoder、Memory | 最终导航策略 |

最终阶段不会默认继承 bootstrap 的 SAC 权重。`--warm-start-sac` 仅在提供含有 SAC 权重的 checkpoint 时用于显式恢复这些权重。

### 视觉自监督

阶段 3 将当前深度图编码后重建。默认损失为 `mse`，当前实现以像素平方误差求和优化；日志中的平均重建误差另行统计。

采集时启用 `--clean-targets`，数据集保存带噪观测与干净深度监督。未保存干净深度时，使用观测深度作为目标。评估两种设置时应明确记录监督来源。

### 记忆自监督

阶段 4 按完整回合批量训练，对变长回合进行 padding，并通过有效位置掩码排除越界目标。因果记忆在时刻 `t` 只读取到 `t` 为止的视觉序列。

默认目标偏移为 `k ∈ {-5, 0, +5}`：

- 图像损失：目标 latent 经冻结 Decoder 生成深度图，与 `I_{t+k}` 比较。
- 潜在表示损失：目标 latent 与冻结 Encoder 对目标帧生成的 latent 比较。
- 各偏移的有效损失相加。图像项默认使用平均像素 MSE，latent 项使用 MSE，后者权重为 `memory_latent_loss_weight=1.0`。

阶段 4 冻结 Encoder 与 Decoder，只更新 Memory 和投影。未来帧用于监督记忆具有预测信息，不代表推理时访问未来。当前预测不以动作作为条件，也不执行基于模型的规划。

### 最终 SAC 与回放

最终 SAC 使用冻结的记忆表示和当前基础状态。回放中存储当前/下一时刻的记忆特征，SAC 梯度不更新感知模块。

成功/非成功双池回放按完整回合结果分配 transition。默认成功池容量占总容量的 30%，成功样本目标比例随训练进度从 30% 调整到 40%、45%，对应阈值为 25%、70%；池内按优先级采样，池样本不足时调整实际比例。Critic 更新后回写 TD 误差优先级。

## 训练命令

从项目根目录执行。源代码位于 `algorithm/SSVM_SAC/`，示例产物写入 `runs/SSVM_SAC/`。阶段 1、2、5 和环境评估需要 AirSim；阶段 3、4 读取离线数据。

```bash
# 1. Bootstrap：默认随机感知模块冻结
python algorithm/SSVM_SAC/train.py bootstrap \
  --output runs/SSVM_SAC/bootstrap --max-steps 20000 --seed 25 --level 3

# 2. 采集：保存完整回合，并保存干净深度监督
python algorithm/SSVM_SAC/train.py collect \
  --policy-checkpoint runs/SSVM_SAC/bootstrap/bootstrap_latest.pt \
  --dataset datasets/SSVM_SAC --target-frames 30000 \
  --clean-targets --seed 25 --level 3

# 3. 视觉自编码器
python algorithm/SSVM_SAC/train.py vision \
  --dataset datasets/SSVM_SAC --output runs/SSVM_SAC/vision \
  --epochs 50 --seed 25

# 4. 自监督时序记忆
python algorithm/SSVM_SAC/train.py memory \
  --dataset datasets/SSVM_SAC \
  --vision-checkpoint runs/SSVM_SAC/vision/vision_latest.pt \
  --output runs/SSVM_SAC/memory --epochs 100 --seed 25

# 5. 冻结表示，训练最终 SAC；本阶段默认启用课程学习
python algorithm/SSVM_SAC/train.py sac \
  --perception-checkpoint runs/SSVM_SAC/memory/memory_latest.pt \
  --output runs/SSVM_SAC/final --max-steps 150000 --seed 25 \
  --curriculum_mode progress
```

```bash
for seed in 25 26 27; do
  python algorithm/SSVM_SAC/train.py sac \
    --perception-checkpoint runs/SSVM_SAC/memory/memory_latest.pt \
    --output runs/SSVM_SAC/final/seed${seed} \
    --max-steps 150000 --seed "$seed" --curriculum_mode progress
done
```

这个示例共享感知 checkpoint，仅改变最终 SAC 训练种子；若要评估完整流程的随机性，需要分别训练各阶段。

## 参数与 checkpoint

默认加载同目录的 `params.yaml`，可用 `--config` 指定另一个文件。环境参数通过未知参数转交全局配置解析器；独立训练参数采用 `--max-steps`、`--batch-size` 等连字符形式。

| 参数 | 默认值 |
|---|---|
| `vision_lr` | 0.005 |
| `memory_lr` / `memory_aux_lr` | 0.0005 / 0.0003 |
| `vision_batch_size` / `memory_batch_size` | 128 / 16 |
| `vision_lrf` / `memory_lrf` | 0.01 / 0.01 |
| `actor_lr` / `critic_lr` / `alpha_lr` | 0.0004 |
| `hidden_dim` | 256 |
| `batch_size` / `replay_capacity` | 256 / 50000 |
| `gamma` / `tau` | 0.95 / 0.003 |

视觉阶段使用 AdamW，记忆阶段使用 RAdam，两阶段均使用余弦学习率调度。阶段 4 的视觉结构从视觉 checkpoint 恢复，记忆阶段的目标和相关设置由 `_memory_stage_config` 合并；修改 YAML 不意味着所有上游结构都会改变。

当前 checkpoint 模型版本为 6。加载器还接受版本 3、4、5 的视觉阶段 checkpoint；旧版记忆 checkpoint 不直接兼容当前版本，应从兼容的视觉 checkpoint 重新训练 Memory。具体规则见 [checkpoints.py](checkpoints.py)。

## 评估与可视化

```bash
# 固定难度下评估最终策略
python algorithm/SSVM_SAC/train.py eval \
  --checkpoint runs/SSVM_SAC/final/sac_latest.pt \
  --episodes 100 --seed 25 --level 3 --output runs/SSVM_SAC/eval

# 视觉自编码器重建
python algorithm/SSVM_SAC/visualization/visualize_vision_reconstruction.py \
  --dataset datasets/SSVM_SAC \
  --vision-checkpoint runs/SSVM_SAC/vision/vision_latest.pt \
  --output runs/SSVM_SAC/vision/reconstruction.png

# 时序记忆重建
python algorithm/SSVM_SAC/visualization/visualize_memory_reconstruction.py \
  --dataset datasets/SSVM_SAC \
  --vision-checkpoint runs/SSVM_SAC/vision/vision_latest.pt \
  --memory-checkpoint runs/SSVM_SAC/memory/memory_latest.pt \
  --output runs/SSVM_SAC/memory/reconstruction.png
```

`eval` 使用确定性动作，输出 `evaluation.json`。重建可视化按 checkpoint 的目标偏移显示结果，详见[可视化说明](visualization/README.md)。

## 日志与结果

手动指定的 `--output` 保存 checkpoint 等产物。最终阶段训练曲线按算法与随机种子生成，例如：

```text
results/CL-SSVM-SAC/seed25/CL-SSVM-SAC_seed25_log.csv
```

CSV 字段为 `episode,total_timesteps,reward,episode_length,success_rate`。使用统一的算法名称绘图：

```bash
python plot_curves.py --algorithm_name SSVM-SAC --seed 25
```

论文、报告和图例统一使用 SSVM-SAC（自监督视觉记忆增强 SAC）。

## 代码组织

```text
SSVM_SAC/
├── config.py           # 配置定义与校验
├── params.yaml         # 默认实验参数
├── networks.py         # 视觉编码、因果记忆、共享解码器
├── dataset.py          # 回合文件、数据划分、序列与目标组织
├── agent.py            # 冻结表示上的 SAC
├── buffer.py           # 存储记忆特征的双池 PER
├── checkpoints.py      # 保存、恢复及版本兼容检查
├── metrics.py          # 重建指标
├── train.py            # 五阶段训练及评估入口
└── visualization/      # 视觉与记忆重建可视化
```

## 实验报告边界

结构和训练目标说明不等于性能结论。论文应分别报告导航性能、自监督任务质量、在线推理成本和必要消融；记录数据来源、监督类型、感知 checkpoint 与最终训练种子。重建质量较好不能单独证明导航性能改善，也不能代替完整的 AirSim/GPU 训练验证。
