# VMamba TD3 without Cross-Attention

这是VMamba TD3算法的一个变体，移除了交叉注意力机制，使用更简单的融合方式。

## 主要特性

- **无交叉注意力**：使用简单的全局平均池化和拼接融合，替代了复杂的交叉注意力机制
- **更高效**：减少了计算复杂度，训练速度更快
- **保持VMamba优势**：仍然使用VMamba作为视觉特征提取器
- **兼容性**：与原始VMamba TD3具有相同的接口和使用方式

## 架构差异

### 原始VMamba TD3
- 使用CrossAttention融合视觉特征和状态特征
- 复杂的Query-Key-Value机制
- 更高的计算开销

### VMamba TD3 without Cross-Attention
- 使用SimpleFusionEncoder进行特征融合
- 全局平均池化 + 拼接 + MLP
- 更简单高效的特征融合方式

## 网络组件

### SimpleFusionEncoder
```python
class SimpleFusionEncoder(nn.Module):
    """
    简单的融合编码器：不使用交叉注意力，使用全局平均池化和简单拼接
    """
```

主要步骤：
1. 对视觉特征进行全局平均池化
2. 将视觉特征和状态特征分别投影到hidden_dim
3. 拼接两种特征
4. 通过MLP进行最终融合

## 使用方法

### 1. 导入
```python
from algorithm.vmamba_td3_no_cross.vmamba_td3_no_cross import VMambaTD3NoCrossAgent
from algorithm.vmamba_td3_no_cross.vmamba_td3_no_cross import make_agent
```

### 2. 创建Agent
```python
agent = VMambaTD3NoCrossAgent(
    base_dim=base_dim,
    depth_shape=depth_shape,
    action_space=action_space,
    args=args,
    device=device
)

# 或者使用工厂函数
agent = make_agent(env, initial_obs, args, device)
```

### 3. 参数配置
与原始VMamba TD3相同的参数配置，包括：

```python
# VMamba specific parameters
args.vmamba_patch_size = 4
args.vmamba_hidden_dim = 64
args.vmamba_num_vss_blocks = [2, 2]
args.vmamba_drop_path_rate = 0.1
args.vmamba_layer_scale_init = 1e-6
args.vmamba_ssm_d_state = 16
args.vmamba_ssm_ratio = 2.0
args.vmamba_mlp_ratio = 4.0

# General TD3 parameters
args.feature_dim = 256
args.state_feature_dim = 256
args.hidden_dim = 256
args.lstm_hidden_dim = 256
# ... 其他TD3参数
```

## 性能对比

| 特性 | 原始VMamba TD3 | VMamba TD3 No-Cross |
|------|---------------|-------------------|
| 交叉注意力 | ✓ | ✗ |
| 计算复杂度 | 高 | 低 |
| 训练速度 | 较慢 | 较快 |
| 内存占用 | 较高 | 较低 |
| 特征融合 | 复杂 | 简单 |

## 测试

运行测试脚本验证实现：

```bash
python test_vmamba_td3_no_cross_simple.py
```

## 文件结构

```
algorithm/vmamba_td3_no_cross/
├── __init__.py                    # 模块初始化
├── networks.py                    # 网络组件定义
├── vmamba_td3_no_cross.py         # Agent实现
└── README.md                      # 本文档
```

## 主要类

### VMambaTD3NoCrossAgent
主要的Agent类，实现TD3算法但不使用交叉注意力。

### SimpleFusionEncoder
简单的特征融合编码器，替代原始的交叉注意力机制。

### VMambaVisualEncoder
视觉特征编码器，使用VMamba作为backbone。

### StateMLP
状态特征编码器，使用MLP处理低维状态。

## 注意事项

1. 输入格式与原始版本相同：4帧堆叠的深度图像
2. 支持相同的动作空间和状态空间
3. 模型保存/加载格式兼容
4. 超参数设置与原始版本相同

## 优势

1. **计算效率更高**：移除了复杂的交叉注意力计算
2. **训练速度更快**：简化的融合机制减少了前向传播时间
3. **内存占用更低**：减少了中间变量的存储
4. **实现更简单**：代码更易理解和修改
5. **稳定性更好**：减少了注意力机制可能带来的不稳定性

## 适用场景

- 需要更快训练速度的场景
- 计算资源有限的环境
- 对特征融合复杂度要求不高的任务
- 需要快速原型开发的场景

## 扩展性

该实现保留了良好的扩展性：

1. 可以轻松替换SimpleFusionEncoder为其他融合方式
2. 支持不同的视觉编码器配置
3. 可以添加其他特征处理模块
4. 兼容现有的训练和评估流程
