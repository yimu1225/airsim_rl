# AirSim RL 项目 SB3 迁移计划

> 目标：将当前手写训练框架迁移到 Stable-Baselines3 (SB3)，基线算法由 SB3 提供，所有改进模块（Vim/Mamba/PER/Noisy/Asymmetric 等）仍自主实现，CNN 模块统一使用 SB3 默认卷积模块。

---

## 一、当前算法清单与迁移策略

当前 `algorithm/` 目录下共 **29** 个算法实现，按迁移策略分类如下：

### 1.1 基线算法（直接用 SB3，零自定义）

| 算法 | SB3 对应类 | 说明 |
|------|-----------|------|
| `td3` | `stable_baselines3.TD3` | 完全替换 |
| `sac` | `stable_baselines3.SAC` | 完全替换 |
| `ppo` | `stable_baselines3.PPO` | 完全替换 |
| `ddpg` | `stable_baselines3.DDPG` | 完全替换 |

**迁移方式**：直接使用 SB3 原生类，通过 `policy_kwargs` 传入自定义 `FeatureExtractor`。

---

### 1.2 经验回放改进（自定义 ReplayBuffer）

| 算法 | 基线 | 改进点 | 迁移策略 |
|------|------|--------|---------|
| `per_td3` | TD3 | PER 优先回放 | 继承 `TD3`，重写 `_setup_model()` 替换 `ReplayBuffer` → `PrioritizedReplayBuffer`，`train()` 加 IS 权重 |
| `per_aetd3` | TD3 | PER + AutoEncoder | 继承 `TD3`，同时替换 Buffer 和 Feature Extractor（加 AE 重建 loss） |
| `PER_ST_Vim_TD3` | TD3 | PER + ST-Vim | 继承 `TD3`，替换 Buffer + Feature Extractor |
| `PER_ST_Vim_SAC` | SAC | PER + ST-Vim | 继承 `SAC`，替换 Buffer + Feature Extractor |

---

### 1.3 网络结构改进（自定义 Policy / Network）

| 算法 | 基线 | 改进点 | 迁移策略 |
|------|------|--------|---------|
| `noisy_td3` | TD3 | Noisy Network | 自定义 `NoisyTD3Policy` 继承 `TD3Policy`，`make_actor()` 中使用 `NoisyLinear` |
| `noisy_td3_type2` | TD3 | Noisy Network Type2 | 同上，参数化噪声方案不同 |
| `td3_asym` | TD3 | Asymmetric Critic | 自定义 `AsymCritic` + `AsymTD3Policy`，Critic 输入 privileged info |
| `per_td3_asym` | TD3 | PER + Asymmetric | 组合：Asym Critic + PER Buffer |
| `ST_Vim_TD3_asym` | TD3 | ST-Vim + Asymmetric | 组合：ST-Vim Feature Extractor + Asym Critic |
| `aetd3` | TD3 | AutoEncoder | 自定义 Feature Extractor 加重建分支，Actor/Critic 共享编码器 |

---

### 1.4 时序/视觉 backbone 改进（自定义 Feature Extractor）

**核心思路**：将时序模型封装为 `BaseFeaturesExtractor` 子类，SB3 Actor/Critic 看到的是固定维度特征向量。

| 算法 | Backbone 类型 | 迁移策略 |
|------|--------------|---------|
| `LSTM_SAC` | LSTM 时序编码 | 自定义 `LSTMFeatureExtractor`，输入 `(B, T, C, H, W)` → LSTM → feature vector |
| `Vim_TD3` | VisionMamba | 自定义 `VimFeatureExtractor`，逐帧 Vim 编码 |
| `ST_Vim_TD3` | Spatial-Temporal Vim | 自定义 `STVimFeatureExtractor`，帧级 Vim + 时序 Mamba |
| `ST_Seq_Vim_TD3` | ST-Vim 序列 | 自定义 `STSeqVimFeatureExtractor` |
| `STV_Seq_Vim_TD3` | STV-Seq-Vim | 自定义 `STVSeqVimFeatureExtractor` |
| `STV_Patch_TD3` | Patch-level Vim | 自定义 `VimPatchFeatureExtractor` |
| `ST_SVim_TD3` | Spatial Vim | 自定义 `STSVimFeatureExtractor` |
| `ST_3D_Vim_TD3` | 3D Vim | 自定义 `ST3DVimFeatureExtractor`，3D 卷积 + Vim |
| `ST_Vim_SAC` | ST-Vim | 自定义 `STVimFeatureExtractor`，复用 TD3 版本 |
| `ST_Vim_PPO` | ST-Vim | 自定义 `STVimFeatureExtractor`，PPO 使用 Actor-Critic 共享 extractor |
| `cfc_td3` | CFC (Closed-form Continuous-time) | 自定义 `CFCFeatureExtractor` |
| `mamba_td3` | Mamba | 自定义 `MambaFeatureExtractor` |
| `gam_mamba_td3` | GAM + Mamba | 自定义 `GAMMambaFeatureExtractor`，Mamba 后接 GAM 注意力 |
| `gam_td3` | GAM + CNN | 自定义 `GAMCNNFeatureExtractor`，CNN 后接 GAM 注意力 |
| `st_dualvim_td3` | Dual-branch Vim | 自定义 `DualVimFeatureExtractor`，双分支 Vim 编码 |

---

## 二、环境层改造（阶段 1）

### 2.1 Observation Space 保持 Dict 格式

当前环境返回 `dict({'depth': ..., 'base': ...})`，这是最佳实践。SB3 通过 `MultiInputPolicy` + `CombinedExtractor` 原生支持，无需像参考项目那样把状态塞进图像通道。

```python
self.observation_space = spaces.Dict({
    "depth": spaces.Box(low=0, high=255, shape=(n_frames, H, W), dtype=np.uint8),
    "base": spaces.Box(low=-np.inf, high=np.inf, shape=(base_dim,), dtype=np.float32),
})
```

### 2.2 统一 Feature Extractor 架构

所有算法共享同一套 `FeatureExtractor` 接口，差异仅在视觉 backbone 的选择。

```
BaseFeaturesExtractor (SB3)
    └── AirSimBaseExtractor
            ├── CNNExtractor          (SB3 NatureCNN / 自定义轻量CNN)
            ├── VimExtractor
            ├── STVimExtractor
            ├── MambaExtractor
            ├── LSTMExtractor
            ├── CFCExtractor
            └── DualVimExtractor
```

**核心设计**：

```python
class AirSimBaseExtractor(BaseFeaturesExtractor):
    """
    统一架构：视觉分支 + Base 状态分支 → 拼接
    视觉分支由子类通过 _build_vision_backbone() 实现
    """
    def __init__(self, observation_space, features_dim=256, 
                 base_feature_dim=32, vision_output_dim=None):
        # vision_output_dim 默认 = features_dim - base_feature_dim
        # 视觉分支：由子类实现
        self.vision_net = self._build_vision_backbone(observation_space["depth"])
        # Base 分支：小 MLP
        self.base_net = nn.Sequential(
            nn.Linear(base_dim, base_feature_dim), nn.ReLU(),
        )
        
    def forward(self, obs: dict) -> th.Tensor:
        vision_feat = self.vision_net(obs["depth"])
        base_feat = self.base_net(obs["base"])
        return th.cat([vision_feat, base_feat], dim=1)
```

### 2.3 CNN 模块统一替换

**当前**：`algorithm/cnn_modules.py` 使用 `MobileNetV3-Small`。

**目标**：基线算法统一使用 SB3 默认 `NatureCNN`；改进算法在 `FeatureExtractor` 内部使用 `NatureCNN` 或自定义轻量 CNN。

```python
# 基线用 NatureCNN（SB3 默认）
from stable_baselines3.common.torch_layers import NatureCNN

# 如果 NatureCNN 对深度图分辨率过大/过小，使用自定义轻量 CNN
class DepthCNN(nn.Module):
    def __init__(self, n_input_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
```

---

## 三、算法实现层（阶段 2~4）

### 3.1 目录结构

```
sb3_extensions/
├── __init__.py
├── feature_extractors/
│   ├── __init__.py
│   ├── base.py                 # AirSimBaseExtractor
│   ├── cnn_extractors.py       # NatureCNNExtractor, DepthCNNExtractor
│   ├── vim_extractors.py       # VimExtractor, STVimExtractor, ...
│   ├── mamba_extractors.py     # MambaExtractor, GAMMambaExtractor
│   ├── lstm_extractors.py      # LSTMExtractor
│   └── cfc_extractors.py       # CFCExtractor
├── policies/
│   ├── __init__.py
│   ├── asym_policies.py        # AsymTD3Policy, AsymSACPolicy
│   └── noisy_policies.py       # NoisyTD3Policy
├── buffers/
│   ├── __init__.py
│   └── prioritized_replay.py   # PrioritizedReplayBuffer
├── networks/
│   ├── __init__.py
│   ├── noisy_linear.py         # NoisyLinear 层
│   └── gam.py                  # GAM 注意力模块
└── callbacks/
    ├── __init__.py
    ├── curriculum.py           # CurriculumCallback
    ├── airsim_health.py        # AirSimHealthCallback (UE4 重启/恢复)
    └── eval_gif.py             # EvalAndSaveGifCallback

sb3_algorithms/
├── __init__.py
├── per_td3.py
├── per_sac.py
├── per_aetd3.py
├── noisy_td3.py
├── noisy_td3_type2.py
├── asym_td3.py
├── per_asym_td3.py
├── aetd3.py
├── Vim_TD3.py
├── ST_Vim_TD3.py
├── ST_Vim_SAC.py
├── ST_Vim_PPO.py
├── ... (其他时序算法)
└── base_mixin.py               # 共享工具方法
```

### 3.2 基线算法调用示例

```python
from stable_baselines3 import TD3, SAC, PPO, DDPG
from sb3_extensions.feature_extractors import AirSimCNNExtractor

policy_kwargs = dict(
    features_extractor_class=AirSimCNNExtractor,
    features_extractor_kwargs=dict(features_dim=256, base_feature_dim=32),
    net_arch=[256, 256],
    activation_fn=nn.ReLU,
)

model = TD3(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    buffer_size=30000,
    batch_size=512,
    gamma=0.98,
    tau=0.003,
    tensorboard_log="./results/",
    verbose=1,
)
model.learn(total_timesteps=1_000_000, callback=callbacks)
```

### 3.3 改进算法继承模式

以 `PER_TD3` 为例：

```python
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy
from sb3_extensions.buffers import PrioritizedReplayBuffer

class PERTD3(TD3):
    """
    继承 SB3 TD3，仅替换 ReplayBuffer 并在 train() 中支持 priority
    """
    def __init__(self, *args, per_alpha=0.6, per_beta=0.4, **kwargs):
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        super().__init__(*args, **kwargs)
    
    def _setup_model(self):
        # 调用父类 setup，但替换 buffer
        super()._setup_model()
        self.replay_buffer = PrioritizedReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            alpha=self.per_alpha,
        )
    
    def train(self, gradient_steps: int, batch_size: int = 100):
        # 采样时获取 priority weights
        replay_data = self.replay_buffer.sample(batch_size, beta=self.per_beta)
        # ... 父类 TD3 的 critic/actor 更新逻辑
        # 更新后计算新 priority
        # self.replay_buffer.update_priorities(indices, new_priorities)
```

---

## 四、训练主流程（阶段 5）

### 4.1 新建 `main_sb3.py`

旧 `main_async.py` **保留不动**，新旧并存直到全部验证通过。

```python
#!/usr/bin/env python3
"""SB3-based training entry point for AirSim RL."""

from stable_baselines3 import TD3, SAC, PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_extensions.callbacks import CurriculumCallback, AirSimHealthCallback
from sb3_extensions.feature_extractors import (
    AirSimCNNExtractor,           # 基线用
    STVimExtractor,               # ST-Vim 用
    LSTMExtractor,                # LSTM 用
    # ... 按需导入
)
from sb3_algorithms import (
    PERTD3, NoisyTD3, AsymTD3, 
    STVimTD3, STVimSAC, LSTMSAC,
    # ...
)

def get_model(algo_name: str, env, args):
    """根据算法名返回对应的 SB3 model 实例。"""
    policy_kwargs = dict(
        features_extractor_class=resolve_extractor(algo_name),
        features_extractor_kwargs=dict(features_dim=args.hidden_dim),
        net_arch=[args.hidden_dim, args.hidden_dim],
    )
    
    algo_map = {
        "TD3": TD3,
        "SAC": SAC,
        "PPO": PPO,
        "DDPG": DDPG,
        "PER_TD3": PERTD3,
        "noisy_td3": NoisyTD3,
        "ST_Vim_TD3": STVimTD3,
        "ST_Vim_SAC": STVimSAC,
        "LSTM_SAC": LSTMSAC,
        # ...
    }
    
    AlgoClass = algo_map[algo_name]
    return AlgoClass("MultiInputPolicy", env, policy_kwargs=policy_kwargs, ...)


def main():
    args = get_config()
    env = create_env_from_name(args, n_frames=args.n_frames)
    
    model = get_model(args.algorithm_name, env, args)
    
    callbacks = [
        CheckpointCallback(save_freq=10000, save_path=f"models/{args.algorithm_name}/"),
        CurriculumCallback(check_freq=args.eval_freq),
        AirSimHealthCallback(),
    ]
    
    model.learn(
        total_timesteps=args.max_timesteps,
        callback=callbacks,
        log_interval=args.log_interval,
        tb_log_name=f"{args.algorithm_name}_seed{args.seed}",
    )
    
    model.save(f"models/{args.algorithm_name}/final")
```

### 4.2 周边逻辑迁移为 SB3 Callback

| 原 `main_async.py` 逻辑 | SB3 Callback 实现 |
|------------------------|------------------|
| TensorBoard 写入 (`SummaryWriter`) | SB3 内置，无需处理 |
| CSV 日志 | `CSVLoggerCallback(BaseCallback)` |
| 课程学习 (`CL-` 前缀，动态调 level) | `CurriculumCallback`：检查 `env.success_deque`，达标后 `env.env_method("increase_level")` |
| UE4 重启 / 异常恢复 | `AirSimHealthCallback`：`on_rollout_end` 检测异常，调用 `env.check_ue4_status()` |
| 定期 Eval & 保存 GIF | `EvalAndGifCallback`：使用 SB3 `EvalCallback` + 自定义 GIF 录制 |
| 可视化窗口 (`render_window`) | `DepthRenderCallback` |
| 模型保存 (`agent.save`) | `CheckpointCallback`（SB3 原生） |

---

## 五、迁移优先级与里程碑

### 里程碑 1：地基（1 周）
- [ ] 环境 `Dict` observation 验证通过
- [ ] `AirSimCNNExtractor`（NatureCNN + Base MLP）实现
- [ ] `main_sb3.py` 跑通 **TD3** 基线（无改进，纯 SB3）
- [ ] `main_sb3.py` 跑通 **SAC**、**PPO**、**DDPG** 基线
- [ ] `CurriculumCallback` + `AirSimHealthCallback` 实现

### 里程碑 2：Buffer & Network 改进（1 周）
- [ ] `PrioritizedReplayBuffer` 实现
- [ ] `per_td3`、`per_sac`、`per_aetd3` 实现
- [ ] `NoisyLinear` 层实现
- [ ] `noisy_td3`、`noisy_td3_type2` 实现

### 里程碑 3：Asymmetric 系列（3~4 天）
- [ ] `AsymCritic` + `AsymTD3Policy` 实现
- [ ] `td3_asym`、`per_td3_asym` 实现
- [ ] `ST_Vim_TD3_asym` 实现

### 里程碑 4：时序 Backbone 系列（2 周）
- [ ] `LSTMExtractor` + `LSTM_SAC` 实现
- [ ] `VimExtractor` + `Vim_TD3` 实现
- [ ] `STVimExtractor` + `ST_Vim_TD3`、`ST_Vim_SAC`、`ST_Vim_PPO` 实现
- [ ] `MambaExtractor` + `mamba_td3` 实现
- [ ] `CFCExtractor` + `cfc_td3` 实现
- [ ] 其余 Vim 变体（Seq / Patch / SVim / 3D / Dual）依次实现

### 里程碑 5：注意力/GAM（3~4 天）
- [ ] `GAM` 模块实现
- [ ] `GAMCNNExtractor` + `gam_td3` 实现
- [ ] `GAMMambaExtractor` + `gam_mamba_td3` 实现

### 里程碑 6：收尾（2~3 天）
- [ ] 全部算法与旧入口 `main_async.py` 结果对比验证
- [ ] 旧 `algorithm/` 代码归档/清理
- [ ] 统一文档更新

---

## 六、关键设计决策

### 6.1 为什么不用参考项目的"状态塞进图像通道"方案？

参考项目将 `state` 嵌入到深度图的第 2 通道 `(H, W, 2)`，这种做法：
- **信息损失**：连续状态被离散到像素网格。
- **不通用**：难以扩展到时序输入或多模态。
- **SB3 反模式**：`MultiInputPolicy` + `CombinedExtractor` 是 SB3 处理多模态的标准方案。

### 6.2 NatureCNN 是否适合深度图？

SB3 `NatureCNN` 默认结构：
```
Conv(8,4) → Conv(4,2) → Conv(3,1) → Flatten → Linear
```
这是为 Atari `(4, 84, 84)` 设计的。对于深度图：
- 如果分辨率接近 `84x84`：可直接使用。
- 如果分辨率较大（如 `128x128`）或较小：建议自定义 `DepthCNN`（stride=2 的 3x3 卷积堆叠），但仍放在 `BaseFeaturesExtractor` 框架内，保持接口统一。

### 6.3 时序输入的维度问题

当前 `main_async.py` 对 recurrent algo 手动处理 `(T, 1, H, W)`。迁移后：
- **方案 A（推荐）**：环境 `observation_space["depth"]` 直接定义为 `(T, H, W)`，Feature Extractor 内部处理时序。
- **方案 B**：环境保持 `(C, H, W)`（C=1 或 stack_frames），Feature Extractor 自行 reshape 为时序。

建议采用 **方案 A**，逻辑更清晰，且 SB3 Buffer 直接存储 `(T, H, W)` 无压力。

### 6.4 新旧模型兼容性

- SB3 `model.save()` 保存的是 `zip` 格式（含 policy 参数和类信息）。
- 旧模型（`.pth`）无法直接加载到 SB3 模型中。
- **过渡策略**：训练新模型用 SB3，旧模型评估仍用 `main_async.py`，直到新模型全面替代。

---

## 七、给 Codex 的实现委托建议

按以下顺序委托 Codex 实现，每次聚焦一个里程碑：

1. **Task 1**：`sb3_extensions/feature_extractors/base.py` + `cnn_extractors.py` + `main_sb3.py`（仅 TD3 基线）
2. **Task 2**：`sb3_extensions/callbacks/`（Curriculum + AirSimHealth）
3. **Task 3**：`sb3_extensions/buffers/prioritized_replay.py` + `sb3_algorithms/per_td3.py`
4. **Task 4**：`sb3_extensions/policies/noisy_policies.py` + `sb3_algorithms/noisy_td3.py`
5. **Task 5**：`sb3_extensions/policies/asym_policies.py` + `sb3_algorithms/asym_td3.py`
6. **Task 6+**：时序 Extractor 系列（每次 2~3 个算法）

---

## 附录：算法 → Feature Extractor → Policy 映射表

| 算法名 | 需要自定义 Feature Extractor | 需要自定义 Policy | 需要自定义 Buffer |
|--------|---------------------------|------------------|-----------------|
| td3 / sac / ppo / ddpg | ❌ (AirSimCNNExtractor) | ❌ | ❌ |
| per_td3 / per_sac / per_aetd3 | ❌ | ❌ | ✅ |
| noisy_td3 / noisy_td3_type2 | ❌ | ✅ (Actor 换 NoisyLinear) | ❌ |
| td3_asym / per_td3_asym | ❌ | ✅ (Critic 加 privileged) | 部分 |
| aetd3 | ✅ (加重建分支) | ❌ | ❌ |
| LSTM_SAC | ✅ (LSTMExtractor) | ❌ | ❌ |
| Vim_TD3 | ✅ (VimExtractor) | ❌ | ❌ |
| ST_Vim_TD3 / ST_Vim_SAC / ST_Vim_PPO | ✅ (STVimExtractor) | ❌ | ❌ |
| ST_Seq_Vim_TD3 / STV_Seq_Vim_TD3 | ✅ (时序 Extractor) | ❌ | ❌ |
| STV_Patch_TD3 | ✅ (PatchExtractor) | ❌ | ❌ |
| ST_SVim_TD3 | ✅ (SVimExtractor) | ❌ | ❌ |
| st_3dvim_td3 | ✅ (3DVimExtractor) | ❌ | ❌ |
| st_dualvim_td3 | ✅ (DualVimExtractor) | ❌ | ❌ |
| cfc_td3 | ✅ (CFCExtractor) | ❌ | ❌ |
| mamba_td3 | ✅ (MambaExtractor) | ❌ | ❌ |
| gam_td3 | ✅ (GAMCNNExtractor) | ❌ | ❌ |
| gam_mamba_td3 | ✅ (GAMMambaExtractor) | ❌ | ❌ |
| ST_Vim_TD3_asym | ✅ (STVimExtractor) | ✅ (Asym Critic) | ❌ |
| PER_ST_Vim_TD3 / PER_ST_Vim_SAC | ✅ (STVimExtractor) | ❌ | ✅ |
