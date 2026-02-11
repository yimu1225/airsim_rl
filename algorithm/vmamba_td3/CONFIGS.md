# VMamba-TD3 配置对比

## 模型大小对比

### 1. 超轻量级配置 (Ultra-Light)
**适用场景**: 快速原型验证、资源受限环境

```bash
--vmamba_hidden_dim 32 \
--vmamba_num_vss_blocks 2 2 2 2 \
--feature_dim 64 \
--state_feature_dim 64 \
--hidden_dim 128 \
--batch_size 256
```

**预估参数量**: ~0.5M  
**内存占用**: ~2GB (GPU)  
**训练速度**: 最快

---

### 2. 轻量级配置 (Light) - 推荐用于快速训练
**适用场景**: 日常训练、快速迭代

```bash
--vmamba_hidden_dim 64 \
--vmamba_num_vss_blocks 2 2 4 2 \
--feature_dim 128 \
--state_feature_dim 128 \
--hidden_dim 256 \
--batch_size 128
```

**预估参数量**: ~2M  
**内存占用**: ~4GB (GPU)  
**训练速度**: 快

---

### 3. 标准配置 (Standard) - 推荐用于正式实验
**适用场景**: 正式实验、论文结果

```bash
--vmamba_hidden_dim 96 \
--vmamba_num_vss_blocks 2 2 6 2 \
--feature_dim 256 \
--state_feature_dim 256 \
--hidden_dim 512 \
--batch_size 64
```

**预估参数量**: ~8M  
**内存占用**: ~8GB (GPU)  
**训练速度**: 中等

---

### 4. 重量级配置 (Heavy)
**适用场景**: 追求最佳性能、充足计算资源

```bash
--vmamba_hidden_dim 128 \
--vmamba_num_vss_blocks 2 2 9 2 \
--feature_dim 512 \
--state_feature_dim 512 \
--hidden_dim 1024 \
--batch_size 32
```

**预估参数量**: ~20M  
**内存占用**: ~16GB (GPU)  
**训练速度**: 较慢

---

## 原始VMamba模型对比

### VMambaT (Tiny)
```
hidden_dim=96, num_vss_blocks=[2, 2, 9, 2]
参数量: ~22M (用于ImageNet分类)
```

### VMambaS (Small)
```
hidden_dim=96, num_vss_blocks=[2, 2, 27, 2]
参数量: ~44M (用于ImageNet分类)
```

### VMambaB (Base)
```
hidden_dim=128, num_vss_blocks=[2, 2, 27, 2]
参数量: ~75M (用于ImageNet分类)
```

**注意**: 原始模型太大，不适合强化学习在线训练！

---

## 关键参数说明

### vmamba_hidden_dim
- 控制VMamba主干网络的基础通道数
- 每个下采样阶段通道数会翻倍: hidden_dim → 2×hidden_dim → 4×hidden_dim → 8×hidden_dim
- **影响**: 对模型大小和性能影响最大

### vmamba_num_vss_blocks
- 控制4个阶段各自的VSSBlock数量
- 格式: [stage1, stage2, stage3, stage4]
- **建议**: 
  - Stage3是主要的特征提取阶段，可以适当增加
  - Stage1和Stage2可以保持较少的块数
  - Stage4通常保持为2

### feature_dim
- VMamba视觉编码器的输出特征维度
- **建议**: 
  - 与vmamba_hidden_dim成正比
  - 轻量级: 64-128
  - 标准: 128-256
  - 重量级: 256-512

### hidden_dim (Actor/Critic MLP)
- Actor和Critic网络的隐藏层维度
- **建议**: 
  - 应该≥feature_dim
  - 通常设置为feature_dim的1.5-2倍

---

## 性能vs效率权衡

| 配置 | 参数量 | GPU内存 | 训练速度 | 性能 |
|------|--------|---------|----------|------|
| 超轻量 | 0.5M | 2GB | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 轻量 | 2M | 4GB | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 标准 | 8M | 8GB | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 重量 | 20M | 16GB | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 调优建议

### 内存不足？
1. 减小 `vmamba_hidden_dim`
2. 减少 `vmamba_num_vss_blocks` 中的数值
3. 减小 `batch_size`
4. 减小 `buffer_size`

### 训练太慢？
1. 使用轻量级配置
2. 减小 `seq_len` (但可能影响性能)
3. 减小 `vmamba_num_vss_blocks` 中stage3的数值
4. 增大 `policy_freq` (减少Actor更新频率)

### 性能不佳？
1. 增大 `vmamba_hidden_dim`
2. 增加 `vmamba_num_vss_blocks` 中stage3的数值
3. 增大 `feature_dim`
4. 增大 `hidden_dim`
5. 增加 `vmamba_temporal_layers`

### 训练不稳定？
1. 降低学习率 (`actor_lr`, `critic_lr`)
2. 增大 `tau` (软更新系数)
3. 使用梯度裁剪 (`grad_clip`)
4. 减小 `policy_noise`

---

## 完整训练命令示例

### 轻量级快速训练
```bash
python main.py \
  --algorithm_name vmamba_td3 \
  --max_timesteps 200000 \
  --vmamba_hidden_dim 64 \
  --vmamba_num_vss_blocks 2 2 4 2 \
  --feature_dim 128 \
  --hidden_dim 256 \
  --batch_size 128 \
  --seq_len 4
```

### 标准配置训练
```bash
python main.py \
  --algorithm_name vmamba_td3 \
  --max_timesteps 500000 \
  --vmamba_hidden_dim 96 \
  --vmamba_num_vss_blocks 2 2 6 2 \
  --feature_dim 256 \
  --hidden_dim 512 \
  --batch_size 64 \
  --seq_len 4 \
  --actor_lr 3e-4 \
  --critic_lr 5e-4
```
