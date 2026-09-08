# MAVM-SAC 设计方案

> 状态：第一版代码已实现；真实 AirSim/GPU 长训练仍需按阶段执行验证。
>
> 工作名：MAVM-SAC（Mamba-based MAVRL SAC）。

## 1. 目标

MAVM-SAC 使用全 Mamba 感知框架替换 MAVRL 中的视觉与记忆模块，同时保留 MAVRL 的分阶段训练方法：

- 使用 Vision Mamba 进行单帧深度图空间编码；
- 使用因果 Temporal Mamba 形成跨时间记忆；
- 使用 Vision Mamba Decoder 重建过去两帧并预测未来一帧；
- 最终使用冻结的视觉编码器和记忆模块训练 SAC；
- 在线推理时每一步只输入当前深度帧，并通过 Mamba cache 保留历史；
- 最终 SAC 不直接使用当前视觉 latent，而只使用 Mamba 记忆和当前基础状态。

本方案对应关系如下：

| MAVRL | MAVM-SAC |
| --- | --- |
| VAE Encoder | Vision Mamba Encoder |
| LSTM | Causal Temporal Mamba |
| 反卷积 Decoder | Vision Mamba Decoder |
| PPO | SAC |

“全 Mamba”主要指完整的感知与记忆框架。SAC 的 Actor 和 Critic 仍使用适合连续控制的 MLP head，与 MAVRL 使用 MLP 控制 head 的做法一致。

## 2. 总体结构

在时刻 `t`，网络只接收当前深度帧 `I_t`：

```text
当前深度帧 I_t
      │
      ▼
Vision Mamba Encoder
      │
      ▼
当前视觉 latent z_t
      │
      ▼
Causal Temporal Mamba
m_t = F(z_t, cache_{t-1})
      │
      ▼
Linear: D_m → 3 × D_r
      │
      ├── r_{t-5}  ──┐
      ├── r_{t-10} ──┼── Shared Vision Mamba Decoder
      └── r_{t+5}  ──┘               │
                                        ▼
                           I_hat_{t-5}, I_hat_{t-10}, I_hat_{t+5}

最终控制路径：

[m_t, base_state_t] ──► SAC Actor / Twin Critic ──► action_t
```

定义：

```text
z_t = SpatialEncoder(I_t)
m_t, cache_t = TemporalMemory.step(z_t, cache_{t-1})
x_t = concat(base_state_t, m_t)
action_t ~ SACActor(x_t)
```

最终 SAC 不直接拼接 `z_t`。当前帧信息已经通过 `z_t` 参与了 `m_t` 的更新，因此 `m_t` 同时表示当前视觉信息和历史视觉信息。

## 3. 模块设计

### 3.1 Vision Mamba Encoder

职责：将单张深度图编码为紧凑视觉表示。

```text
I_t: (B, 1, H, W)
  ↓ Patch Embedding
spatial tokens: (B, N, D)
  ↓ Vision Mamba blocks
z_t: (B, D_z)
```

要求：

- 每次只编码一张深度图；
- 空间扫描可以是双向的，因为它不会访问未来时间帧；
- 输出维度固定，例如 `D_z = 64`；
- 图像输入统一归一化到 `[0, 1]`；
- 不在 Encoder 内处理时间维度。

### 3.2 Causal Temporal Mamba

职责：根据当前视觉 latent 和上一时刻 cache 更新记忆。

训练接口：

```python
memory_sequence = temporal_memory.forward_sequence(
    latent_sequence,
    episode_start_mask,
)
```

在线接口：

```python
memory_t, cache_t = temporal_memory.step(latent_t, cache_t_minus_1)
```

要求：

- 时间方向必须严格因果；
- episode 开始时 cache 必须清零；
- 碰撞、成功、截断和 AirSim 重启后必须清零；
- 批量序列执行和逐帧 cache 执行应产生数值接近的结果；
- 记忆模块只接收视觉 latent，不接收目标状态或未来动作，以保持与 MAVRL 相同的感知记忆定义。

### 3.3 Vision Mamba Decoder

Decoder 不使用 CNN 或反卷积。与 MAVRL 一致，先把记忆表示线性映射成三个独立的重建 latent，再将三个 latent 送入同一个共享 Decoder。Decoder 从单个重建 latent 生成空间 patch token，再通过 Spatial Mamba 重建深度图。

对目标偏移 `k ∈ {-5, -10, +5}`：

```text
m_t
  ↓ reconstruction projection: D_m → 3 × D_r
r_{t-5}, r_{t-10}, r_{t+5}
  ↓ 每个分支分别进入同一个共享 Decoder
reconstruction latent r_t^k
  ├── input projection
  ├── learned patch queries
  └── 2D positional embeddings
              │
              ▼
       Vision Mamba Decoder blocks
              │
              ▼
       Linear patch prediction head
              │
              ▼
           Unpatchify
              │
              ▼
          I_hat_{t+k}
```

可以表示为：

```text
R_t = W_r m_t ∈ R^(3D_r)
(r_t^-5, r_t^-10, r_t^+5) = Split(R_t)
X_t^k = Q_patch + P_2D + broadcast(W_d r_t^k)
Y_t^k = VisionMambaDecoder(X_t^k)
I_hat_{t+k} = Unpatchify(W_out Y_t^k)
```

说明：

- `Q_patch` 是可学习的空间 patch query；
- `P_2D` 是二维位置编码；
- 三个独立的重建 latent 分别对应 `t-5`、`t-10` 和 `t+5`；
- 三个时间目标共用同一个 Vision Mamba Decoder，结构与 MAVRL 的共享方式一致；
- 若 `D_r=64`，重建投影层总输出为 `3×64=192`，但这不改变 `memory_dim`；
- Decoder 的空间扫描可以双向；
- `Linear patch prediction head` 只是必要的 token 到像素投影，不属于 CNN；
- Decoder 只在感知预训练阶段使用，最终 SAC 推理不加载或不执行 Decoder。

### 3.4 SAC 控制器

最终 SAC 状态表示为：

```text
x_t = concat(base_state_t, m_t)
```

控制器包括：

- stochastic Actor；
- Twin Critic；
- Target Critic；
- 自动 entropy coefficient；
- bounded continuous action mapping。

Actor 可以使用 Gaussian 或 Beta 分布。该选择不影响全 Mamba 感知框架和分阶段训练流程。

## 4. 分阶段训练

### 阶段 1：训练初始 SAC

目的：得到能够稳定向目标飞行的基础策略，用于采集深度序列。

训练方式与 MAVRL 的初始策略阶段一致：

- Vision Mamba Encoder 随机初始化并冻结；
- Temporal Mamba 随机初始化并冻结；
- 只训练 SAC Actor、Critic 和 entropy coefficient；
- 使用 `level 2` 障碍环境；
- SAC 输入为 `[m_t, base_state_t]`；
- 使用与 `SB_PER_VSSM_SAC` 相同的成功/普通双池 PER；
- 初始控制器主要依靠基础状态完成导航。

建议输出：

```text
checkpoints/bootstrap_sac/
├── best.pth
├── latest.pth
└── config.yaml
```

### 阶段 2：采集连续序列数据

使用阶段 1 的初始 SAC 在不同环境中运行，保存完整 episode。

每条 transition 至少包含：

```text
episode_id
step_id
depth_t
base_state_t
action_t
reward_t
terminated_t
truncated_t
```

建议同时保存：

```text
clean_depth_t
noisy_depth_t
timestamp_t
environment_level
environment_seed
is_success
has_collided
```

数据约束：

- 序列不能跨越 episode reset；
- 碰撞前的深度变化需要保留；
- 训练集、验证集和测试集按 episode 划分；
- 不按单帧随机划分，否则相邻帧会造成数据泄漏；
- 数据应覆盖不同障碍密度、地图、速度和飞行方向；
- 如启用观测噪声，建议使用 noisy depth 作为输入、clean depth 作为重建目标。

### 阶段 3：训练单帧 Vision Mamba Autoencoder

训练路径：

```text
I_t ──► Vision Mamba Encoder ──► z_t
                                  │
                                  ▼
                         Vision Mamba Decoder
                                  │
                                  ▼
                               I_hat_t
```

基本损失：

```text
L_vision = MSE(I_hat_t, I_t)
```

第一版先使用 MSE，以便与 MAVRL 的重建训练保持接近。后续可单独比较 Huber、SSIM、梯度或近障区域加权损失。

本阶段：

- 训练 Vision Mamba Encoder；
- 训练单帧 Vision Mamba Decoder；
- 不训练 Temporal Mamba；
- 不训练 SAC。

建议输出：

```text
checkpoints/vision_pretrain/
├── best.pth
├── latest.pth
├── reconstruction_samples/
└── config.yaml
```

供下一阶段加载的关键权重是 Vision Mamba Encoder。

### 阶段 4：训练 Temporal Mamba 记忆

加载阶段 3 的 Vision Mamba Encoder 并冻结。创建新的 Temporal Mamba 和用于记忆重建的 Vision Mamba Decoder。

训练时输入完整因果序列：

```text
I_0, I_1, ..., I_T
 ↓    ↓          ↓
z_0, z_1, ..., z_T
          ↓
Causal Temporal Mamba
          ↓
m_0, m_1, ..., m_T
```

在每个有效时刻 `t`，使用 `m_t` 重建：

```text
I_{t-5}
I_{t-10}
I_{t+5}
```

阶段 3 和阶段 4 统一使用单一的 Charbonnier Loss：

```text
L_charbonnier = mean(sqrt((I_hat - I)^2 + epsilon^2) - epsilon)
```

阶段 4 对 `t-5`、`t-10`、`t+5` 三个有效目标的 Charbonnier Loss 直接求和，
不除以有效目标数量，也不引入额外的多损失加权项。默认 `epsilon=0.001`。

初始配置建议：

```yaml
reconstruction_offsets: [-5, -10, 5]
reconstruction_loss: charbonnier
charbonnier_epsilon: 0.001
```

有效位规则：

- `t < 5` 时不计算不存在的 `t-5` 过去帧损失；
- `t < 10` 时不计算不存在的 `t-10` 过去帧损失；
- `t + 5 >= episode_length` 时不计算未来帧损失；
- padding 部分不计算任何损失；
- Temporal Mamba 在 episode 起点重置状态；
- 任何输入序列都不能跨 episode。

本阶段严格类比 MAVRL：未来帧预测不额外输入动作。动作条件预测可以作为后续消融，但不属于主方案。

本阶段：

- Vision Mamba Encoder 冻结；
- Temporal Mamba 训练；
- 三分支 reconstruction projection 训练；
- Vision Mamba Decoder 训练；
- SAC 不训练。

建议输出：

```text
checkpoints/memory_pretrain/
├── best.pth
├── latest.pth
├── reconstruction_samples/
└── config.yaml
```

### 阶段 5：冻结感知和记忆，重新训练 SAC

加载训练好的：

- Vision Mamba Encoder；
- Causal Temporal Mamba。

然后：

- 冻结 Vision Mamba Encoder；
- 冻结 Temporal Mamba；
- 不加载或不执行 Vision Mamba Decoder；
- 重新初始化 SAC Actor、Critic、Target Critic 和 entropy coefficient；
- 在完整障碍环境或课程环境中重新训练 SAC；
- 最终控制器输入固定为 `[m_t, base_state_t]`。

因为感知和记忆权重已经冻结，最终 SAC replay buffer 可以直接保存：

```text
base_state_t
memory_t
action_t
reward_t
next_base_state_t
next_memory_t
done_t
```

阶段 1 和阶段 5 都使用与 `SB_PER_VSSM_SAC` 相同的双池优先经验回放。一个
episode 的 transition 会先暂存在内存中；episode 结束后，只要其中任一步
`is_success=True`，整条 episode 就写入成功池，否则写入普通池。两个池内部都按
TD-error 执行 proportional PER，采样时再按训练进度混合：前 25% 的成功样本目标比例
为 0.30，中间阶段为 0.40，训练进度达到 70% 后为 0.45。成功池或普通池样本不足时，
由另一个池补足 batch。

这样 SAC 采样 transition 时不需要重新恢复完整历史，也不会产生由旧模型生成 latent
的 off-policy 不一致问题。若未来解冻 Encoder 或 Temporal Mamba，则必须改为按 episode
采样连续序列并重新计算 memory。

建议输出：

```text
checkpoints/final_sac/
├── best.pth
├── async_<timesteps>.pth
└── config.yaml
```

## 5. 训练与在线推理的区别

### Temporal Mamba 预训练

为了提高 GPU 利用率并进行反向传播，阶段 4 使用完整序列：

```text
(B, T, D_z) ──► Temporal Mamba ──► (B, T, D_m)
```

序列必须是严格因果的。

### 在线推理

在线时不重复输入完整历史：

```python
latent_t = encoder(current_depth)
memory_t, cache = temporal_memory.step(latent_t, cache)
action_t = actor(torch.cat([base_state_t, memory_t], dim=-1))
```

在线推理只输入当前帧，历史保存在 Mamba cache 中。

需要建立一致性测试：

```text
forward_sequence(z_0:T)[t] ≈ repeated_step(z_0), ..., step(z_t)
```

## 6. Episode 和 cache 生命周期

以下情况必须调用 `reset_memory()`：

- 环境正常 reset；
- 到达目标；
- 发生碰撞并结束 episode；
- 达到最大 episode 长度；
- Gymnasium 返回 `terminated=True`；
- Gymnasium 返回 `truncated=True`；
- AirSim/UE4 异常重启或重新连接；
- 开始独立评估 episode。

预期 Agent 接口：

```python
agent.reset_memory()
action = agent.select_action(base_state, current_depth)
agent.observe_transition(...)
agent.end_episode()
```

cache 生命周期应隐藏在 Agent 或感知记忆模块内部，不能让训练入口手工操作底层 Mamba state。

## 7. 推荐目录结构

当前实现结构为：

```text
algorithm/MAVM_SAC/
├── README.md
├── __init__.py
├── agent.py
├── networks.py
├── buffer.py
├── dataset.py
├── checkpoints.py
├── config.py
├── metrics.py
├── params.yaml
└── train.py
```

`train.py` 是独立入口，不依赖 `main_async.py`。推荐直接执行文件；这样不会为了启动
MAVM-SAC 而导入其他算法：

```bash
# 阶段 1：随机冻结感知模块上的 bootstrap SAC
python algorithm/MAVM_SAC/train.py bootstrap \
  --output runs/MAVM_SAC/bootstrap --max-steps 20000 --level 2

# 阶段 2：用 bootstrap 策略按完整 episode 采集数据
python algorithm/MAVM_SAC/train.py collect \
  --policy-checkpoint runs/MAVM_SAC/bootstrap/bootstrap_latest.pt \
  --dataset datasets/MAVM_SAC --episodes 300 --clean-targets --level 2

# 阶段 3：单帧 Vision Mamba 自编码预训练
python algorithm/MAVM_SAC/train.py vision \
  --dataset datasets/MAVM_SAC --output runs/MAVM_SAC/vision --epochs 50

# 阶段 4：冻结 Encoder，训练因果 Temporal Mamba 与 Vision Mamba Decoder
python algorithm/MAVM_SAC/train.py memory \
  --dataset datasets/MAVM_SAC \
  --vision-checkpoint runs/MAVM_SAC/vision/vision_latest.pt \
  --output runs/MAVM_SAC/memory --epochs 100

# 可视化阶段 4：上排为真实帧，下排为重建帧（包含当前帧）
python algorithm/MAVM_SAC/visualization/visualize_memory_reconstruction.py \
  --dataset datasets/MAVM_SAC \
  --vision-checkpoint runs/MAVM_SAC/vision/vision_latest.pt \
  --memory-checkpoint runs/MAVM_SAC/memory/memory_latest.pt \
  --output runs/MAVM_SAC/memory/reconstruction_test.png

# 阶段 5：冻结 Encoder/Memory，重新初始化并训练最终 SAC
python algorithm/MAVM_SAC/train.py sac \
  --perception-checkpoint runs/MAVM_SAC/memory/memory_latest.pt \
  --output runs/MAVM_SAC/final --max-steps 150000

# 最终策略的独立成功率评估
python algorithm/MAVM_SAC/train.py eval \
  --checkpoint runs/MAVM_SAC/final/sac_latest.pt \
  --episodes 100 --output runs/MAVM_SAC/eval
```

```bash
python algorithm/MAVM_SAC/train.py bootstrap \
  --output runs/MAVM_SAC/bootstrap \
  --max-steps 30000 \
  --level 2 \
&& python algorithm/MAVM_SAC/train.py collect \
  --policy-checkpoint runs/MAVM_SAC/bootstrap/bootstrap_latest.pt \
  --dataset datasets/MAVM_SAC \
  --episodes 300 \
  --level 2 \
  --clean-targets \
&& python algorithm/MAVM_SAC/train.py vision \
  --dataset datasets/MAVM_SAC \
  --output runs/MAVM_SAC/vision \
  --epochs 100 \
&& python algorithm/MAVM_SAC/train.py memory \
  --dataset datasets/MAVM_SAC \
  --vision-checkpoint runs/MAVM_SAC/vision/vision_latest.pt \
  --output runs/MAVM_SAC/memory \
  --epochs 100
  --batch-size 1 \
&& python algorithm/MAVM_SAC/train.py sac \
  --perception-checkpoint runs/MAVM_SAC/memory/memory_latest.pt \
  --output runs/MAVM_SAC/final \
  --max-steps 150000
  --overwrite-results
```



未被此入口识别的参数会转交给项目原有 `config.py`，因此仍可使用
`--airsim_ip`、`--airsim_port`、`--settings_file` 等环境参数。
Bootstrap、Vision 和 Memory 阶段会把 TensorBoard event 写入各自的
`<output>/tensorboard`。最终 SAC 阶段与 `main_async.py` 保持一致，将 event 和成功率 CSV
一起写入 `results/<algorithm>/seed<seed>`，因此可以直接用 `tensorboard --logdir=results`
与其他正式训练曲线叠加显示。`eval` 只评估最终 SAC checkpoint，并将汇总和逐回合结果
写入 `<output>/evaluation.json`。
Bootstrap 和最终 SAC 默认每收集 100 个环境步执行 50 次梯度更新，`tqdm` 与
`main_async.py` 一样只显示该批训练更新的进度。每个 episode 结束时会按主训练脚本的
格式打印 reward、length、success rate、level、累计 timesteps 和累计成功数。
Vision 和 Temporal Memory 阶段也会在每个 epoch 内按 batch 显示 `tqdm` 进度和当前
loss；epoch 结束后仍会打印原有的 loss 汇总。

阶段 3 和阶段 4 启动时会各自把训练 split 所需的 episode 解压一次并常驻内存，后续
epoch 不再反复打开压缩 NPZ。两个 DataLoader 会复用 worker、预取 batch，并使用 pinned
memory 与异步 CUDA 拷贝。默认 batch 分开配置：`vision_batch_size` 只控制阶段 3，
`memory_batch_size` 只控制阶段 4，`batch_size` 仅控制 SAC；两个离线阶段仍可用命令行
`--batch-size` 临时覆盖。阶段 4 的一个样本是一段完整 episode，所以其 batch 通常应明显
小于阶段 3 的单帧 batch。

只有阶段 5 会额外记录用于绘制正式训练曲线的 CSV，Bootstrap 和表征学习阶段不会
写入该曲线。CSV 的列名、路径和成功率口径与 VSSM-SAC 一致：

```text
results/<algorithm>/seed<seed>/<algorithm>_seed<seed>_log.csv
episode,total_timesteps,reward,episode_length,success_rate
```

阶段 5 默认启用课程学习；只有进行非课程对照实验时才需要显式传入
`--no-curriculum`。使用默认 `seed=25` 时，曲线文件为
`results/CL-MAVM-SAC/seed25/CL-MAVM-SAC_seed25_log.csv`。`success_rate` 使用 AirGym
的最近 256 个 episode 滑动成功率。若同一种子的文件已存在，新训练会拒绝覆盖；
确认要从头开始时加 `--overwrite-results`。

可直接使用项目原有绘图脚本：

```bash
python plot_curves.py --algorithm_name MAVM-SAC --seed 25 --max_timesteps 150000
```

推荐将感知记忆部分设计成独立模块：

```python
class MambaPerceptionMemory:
    def reset(self, batch_size=1): ...
    def step(self, current_depth): ...
    def encode_sequence(self, depth_sequence, episode_start_mask): ...
```

训练入口和 SAC Agent 只依赖该接口，不直接感知 Vision Mamba、Temporal Mamba 和 cache 的内部细节。

## 8. 初始参数建议

以下只是第一版起点，应根据显存和深度图分辨率调整：

```yaml
# Architecture
image_height: 128
image_width: 128
channels: 1
patch_size: 16
latent_dim: 64
encoder_depth: 2
memory_dim: 128
memory_depth: 2
reconstruction_latent_dim: 64
decoder_embed_dim: 64
decoder_depth: 2
d_state: 16
d_conv: 4
expand: 2
drop_rate: 0.0
drop_path_rate: 0.0

# Reconstruction
reconstruction_offsets: [-5, -10, 5]
reconstruction_loss: charbonnier
charbonnier_epsilon: 0.001

# Success-buffer prioritized replay
sb_per_alpha: 0.6
sb_per_beta0: 0.4
sb_per_beta1: 1.0
sb_per_eps: 0.000001
sb_per_success_capacity_ratio: 0.3
sb_per_success_sample_ratio: 0.30
sb_per_mu_low: 0.30
sb_per_mu_mid: 0.40
sb_per_mu_high: 0.45
sb_per_mu_step1: 0.25
sb_per_mu_step2: 0.70

```

如果深度图为 `128 × 128`，`patch_size=16` 会得到 `8 × 8 = 64` 个空间 patch。相比 `patch_size=32`，它更适合深度图重建和障碍物边缘表达，但显存开销更高。

## 9. Checkpoint 约束

每个 checkpoint 除权重外，还应保存：

```text
stage
model_version
image_shape
patch_size
latent_dim
memory_dim
reconstruction_latent_dim
reconstruction_offsets
normalization_config
dataset_version
optimizer_state
training_step_or_epoch
```

加载阶段必须验证结构配置，避免在 patch size、图像尺寸或 latent 维度不同的情况下静默加载部分权重。

最终 SAC checkpoint 至少包含：

```text
vision_encoder
temporal_memory
actor
critic
critic_target
actor_optimizer
critic_optimizer
entropy_state
normalization_config
architecture_config
```

Vision Mamba Decoder 不需要包含在部署模型中。

## 10. 验证与消融

实现完成后至少验证：

1. Encoder 和 Decoder 的张量形状正确；
2. Decoder 输出尺寸和输入深度图完全一致；
3. Temporal Mamba 不访问未来 token；
4. 序列不会跨 episode；
5. padding 和无效 offset 不参与重建损失；
6. `step()` 与 `forward_sequence()` 输出一致；
7. reset 后 cache 不包含上一 episode 信息；
8. 阶段 5 中 Encoder 和 Temporal Mamba 的梯度始终为空；
9. 保存、加载后确定性动作和 memory 输出一致；
10. AirSim 重启后不会继承旧 memory。
11. 成功 episode 的全部 transition 只进入成功池，失败 episode 只进入普通池；
12. PER priority 会由每次 critic 更新后的 TD-error 回写。

建议的核心消融：

```text
VSSM-SAC                 原有端到端固定窗口模型
MAVM-SAC-no-memory       只使用 Vision Mamba Encoder
MAVM-SAC-past-only       重建 t-5、t-10
MAVM-SAC-future-only     只预测 t+5
MAVM-SAC                 重建 t-5、t-10、t+5
MAVM-SAC-z-skip          SAC 额外输入 z_t（非主方案）
```

主要指标：

- 成功率；
- 碰撞率；
- 平均飞行速度；
- 平均到达时间；
- 最小障碍距离；
- 路径长度和路径效率；
- 过去帧和未来帧重建误差；
- 单步在线推理延迟；
- 模型参数量和显存占用。

## 11. 当前确定的设计决策

- 感知框架全部使用 Mamba，不使用 CNN Decoder；
- 当前帧由 Vision Mamba 编码；
- Temporal Mamba 形成持久化因果记忆；
- Vision Mamba Decoder 重建 `t-5`、`t-10` 并预测 `t+5`；
- Temporal Mamba 的记忆通过线性层投影为三个独立重建 latent；
- 三个时间目标共用同一个 Decoder，不使用三套 Decoder 参数；
- 未来帧预测不输入动作，以保持与 MAVRL 相同的训练定义；
- 最终 SAC 只使用 `[m_t, base_state_t]`，不直接使用 `z_t`；
- 最终 SAC 训练期间冻结 Encoder 和 Temporal Mamba；
- 阶段 1 和阶段 5 均使用 `SB_PER_VSSM_SAC` 的成功/普通双池 PER；
- 在线推理每次只输入当前深度帧；
- Temporal Mamba 的完整因果序列只用于离线训练和并行验证。

## 12. 尚未确定的实验选择

以下选择不影响总体架构，可以在实现前或实验阶段确定：

- SAC Actor 使用 Gaussian 分布还是 Beta 分布；
- 输入使用 clean depth，还是 noisy depth 输入、clean depth 监督；
- Vision Mamba Encoder 和 Decoder 的 patch size；
- 重建损失使用 MSE、Huber 或组合损失；
- 最终阶段是否增加低学习率的感知模块微调；
- 是否将动作条件未来预测作为独立消融实验。
