# SSVM-SAC 视觉与记忆可视化

本目录用于检查**自监督视觉记忆增强 SAC**（Self-Supervised Visual Memory–Enhanced Soft Actor-Critic）的视觉表示与时序记忆。可视化读取离线数据和 checkpoint，不需要启动 AirSim。

## 阶段 3：视觉自编码器

```bash
python algorithm/SSVM_SAC/visualization/visualize_vision_reconstruction.py \
  --dataset datasets/SSVM_SAC \
  --vision-checkpoint runs/SSVM_SAC/vision/vision_latest.pt \
  --output runs/SSVM_SAC/vision/reconstruction.png
```

展示当前观测、监督目标和视觉自编码器输出。默认优先使用数据中保存的干净深度作为目标；传入 `--observed-targets` 可改为观测深度。`--episode-index`、`--time-index` 可指定样本位置。

## 阶段 4：时序记忆

```bash
python algorithm/SSVM_SAC/visualization/visualize_memory_reconstruction.py \
  --dataset datasets/SSVM_SAC \
  --vision-checkpoint runs/SSVM_SAC/vision/vision_latest.pt \
  --memory-checkpoint runs/SSVM_SAC/memory/memory_latest.pt \
  --output runs/SSVM_SAC/memory/reconstruction.png
```

显示哪些目标帧取决于记忆 checkpoint 的 `reconstruction_offsets`。当前训练 YAML 默认为 `[-5, 0, 5]`，对应过去、当前和未来深度图。当前帧重建来自记忆输出，不应与单帧自编码器输出混淆。

记忆计算只读取当前及过去观测。未来帧用于对照和误差计算，不输入当前记忆。各时间目标使用独立 latent，随后共享一套空间 Mamba 解码器。

## checkpoint 与解释范围

模型版本和兼容规则由 [checkpoints.py](../checkpoints.py) 统一检查。默认训练配置见 [params.yaml](../params.yaml)，完整流程见[算法 README](../README.md)。请使用同一训练链的视觉与记忆 checkpoint，确保结构和表示一致。

可视化用于检查感知与记忆学习，不单独证明导航性能。比较不同实验时，应固定数据划分、样本位置、监督类型和时间偏移。
