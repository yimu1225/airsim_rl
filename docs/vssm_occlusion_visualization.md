# VSSM-SAC 多方法策略热力图

`eval.visualize_vssm_occlusion` 在静态测试场景中运行确定性
CL-VSSM-SAC 策略，为四帧深度输入生成完整连续策略输出的可解释性热力图。
运行代码、归因算法和内置自检都位于
`eval/visualize_vssm_occlusion.py` 一个文件内，无独立测试脚本。

提供四种已有方法：

- Occlusion Sensitivity（Zeiler & Fergus, ECCV 2014）；
- Integrated Gradients（Sundararajan et al., ICML 2017）；
- Integrated Gradients + SmoothGrad/NoiseTunnel（Smilkov et al., 2017）；
- MambaLRP core-rule adaptation（Jafari et al., NeurIPS 2024）。

默认设置对应当前实验需求：

- 模型：`models/CL-VSSM-SAC/seed25/async_final.pth`
- 环境：静态测试环境（`SceneEvalAirSimEnv`）
- 模型种子：25
- episode 种子：25
- 遮挡区域：32×32，步长 8
- 遮挡填充值：255（15 m 截断后的自由空间）
- IG：32 个 Gauss–Legendre 积分点
- SmoothGrad：8 个噪声样本，深度标准差 5

## 运行

使用项目的 AirSim Conda 环境，并确保静态测试场景和 AirSim 可正常启动：

```bash
/home/yimu/miniconda3/envs/AirSim/bin/python \
  -m eval.visualize_vssm_occlusion
```

当前仓库的 Vision Mamba 使用融合 Mamba/Triton 内核，因此实际模型推理要求
CUDA；无 CUDA 时脚本会在启动 Unreal/AirSim 之前直接给出错误。

只运行部分方法：

```bash
/home/yimu/miniconda3/envs/AirSim/bin/python \
  -m eval.visualize_vssm_occlusion \
  --methods ig ig_smoothgrad mambalrp
```

脚本先跑完一条确定性测试轨迹，再根据当前深度图的障碍物接近程度自动选择
6 个时刻，并尽量保持至少 10 步的时间间隔。接近程度使用当前帧深度第 10
百分位数计算，避免少量椒盐噪声决定采样点。因而默认采样不是人为指定的固定
步数。若复现实验需要固定时刻，仍可显式指定：

```bash
/home/yimu/miniconda3/envs/AirSim/bin/python \
  -m eval.visualize_vssm_occlusion \
  --capture_steps 10 30 50 70 \
  --model_seed 25
```

若显存不足，可减小遮挡批大小；若需要更快但更粗的结果，可增大步长：

```bash
/home/yimu/miniconda3/envs/AirSim/bin/python \
  -m eval.visualize_vssm_occlusion \
  --occlusion_batch_size 16 \
  --stride 16
```

## 输出

默认输出目录为：

```text
results/explainability/occlusion/test_scene/CL-VSSM-SAC/seed25/episode25/run_<UTC时间>/
```

默认每次运行创建独立的 UTC 时间戳目录，防止不同 checkpoint、方法或归因参数
覆盖并混入旧图。只有显式传入 `--output_dir` 时才由使用者负责目录隔离。

其中：

- `current_frame_summary.png`：类似论文排版的当前帧汇总图；
- `step_XXXX_four_frames.png`：某一决策时刻四个历史帧的完整归因图；
- `step_XXXX_attributions.npz`：基础状态、原始深度、动作和全部原始热力图；
- `metadata.json`：checkpoint、环境、色标、遮挡参数和完成的采样步。

每种方法每帧只显示一张总策略热力图，不按动作分图：

- Occlusion 先用各动作的物理范围归一化遮挡前后的绝对动作变化，再对完整
  动作差向量取 L2 范数，再把重叠窗口的这个标量分数按覆盖次数平均。
- IG、IG + SmoothGrad 和 MambaLRP 需要可微的标量目标。脚本先取得原始
  归一化动作向量的单位方向，再把待解释动作投影到该方向，解释“哪些区域
  支持当前完整决策”。若原动作接近零，则使用等权单位方向。

速度与角速度不会以原始量纲直接相加。每种方法在所有采样时刻共享自己的
色标上限，不在每张图上单独拉伸颜色。

CaMeRL 的图同样把多维策略输出呈现为一张 Grad-CAM 图，但论文正文没有说明
它采用的标量聚合公式。因此，本实现复现的是其“总策略影响图”的展示目标，
不是声称逐公式复现其未公开的 Grad-CAM 目标定义。

## MambaLRP core-rule adaptation 的实现范围

这不是普通梯度图改名。脚本在解释期间临时将 actor encoder 中的
空间 Vision Mamba 和时间 Mamba mixer 替换为前向等价的慢速实现，并采用
原论文/官方代码的三项核心传播规则：

- 对 SiLU 使用相关性守恒传播；
- 在选择性 SSM 中停止 `A/B/C` 选择路径的梯度；
- 对 SSM 输出与门控的乘法使用 half-relevance propagation。

最终在 Vision Mamba patch embeddings 上计算
`embedding × modified-gradient`，对通道求和后上采样至 128×128。每次解释
都检查替换前后动作的最大误差不超过 `5e-4`，并把误差写入 metadata；归因
完成后立即恢复原始融合 Mamba。该适配覆盖 MambaLRP 的 Mamba 专用规则，
但没有完整复现官方实现对 patch convolution、normalization、残差和 actor
普通层采用的所有 LRP 规则，因此不宣称端到端 relevance conservation，也不
宣称这是官方完整 MambaLRP 的逐算子复现。图例和 metadata 均标为
`MambaLRP core adaptation`。

脚本只接受与当前 BiMamba-v2 actor 结构完全一致的 checkpoint，并严格加载
`actor_encoder` 和 `actor`，不会加载无关的训练优化器。旧的单向 seed25
checkpoint 缺少反向权重，不能用于该脚本；应使用重新训练得到的双向模型。
脚本的 MambaLRP 适配同时覆盖 BiMamba-v2 的正向与反向扫描。

## 自检

内置自检包括遮挡聚合、自动采样、IG 输出，以及 MambaLRP 前向等价性：

```bash
/home/yimu/miniconda3/envs/AirSim/bin/python \
  -m eval.visualize_vssm_occlusion --self_test
```

静态测试场景的障碍物属于 Unreal 场景几何，不使用训练环境的
`NumberOfObjects` 参数，因此切换到测试环境后不再设置“140 个障碍物”。

## 方法来源

- [Occlusion Sensitivity 原始工作](https://arxiv.org/abs/1311.2901)
- [Integrated Gradients 原始论文](https://arxiv.org/abs/1703.01365)
- [SmoothGrad 原始论文](https://arxiv.org/abs/1706.03825)
- [MambaLRP NeurIPS 2024 论文](https://papers.neurips.cc/paper_files/paper/2024/hash/d6d0e41e0b1ed38c76d13c9e417a8f1f-Abstract-Conference.html)
- [MambaLRP 官方实现](https://github.com/FarnoushRJ/MambaLRP)
