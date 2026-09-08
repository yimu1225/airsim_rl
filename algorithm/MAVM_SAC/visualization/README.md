# 重建诊断

阶段3独立重建：`python algorithm/MAVM_SAC/visualization/visualize_vision_reconstruction.py`

阶段4重建：`python algorithm/MAVM_SAC/visualization/visualize_memory_reconstruction.py`

固定16张训练图过拟合：

```bash
python algorithm/MAVM_SAC/visualization/overfit_vision.py
```

从随机权重开始，使用当前 params.yaml 的 Encoder/Decoder、学习率和梯度裁剪，
以 SSE 训练1000次更新。固定选择训练集16个 episode 的中间帧，优先使用 clean_depth
监督。仅加载选中 episode，不预加载整个数据集。

输出到 runs/MAVM_SAC/vision_overfit：samples.json、config.json、metrics.csv、
每100次更新的对比图和最终 overfit.pt。grad_norm 是裁剪前梯度范数。
这些指标仅衡量固定训练样本的拟合，不代表验证集泛化质量。

重复实验需通过 --output 指定新的目录，防止覆盖已有结果；可用 --steps 调整更新次数，
--samples 调整样本数。正式阶段3 checkpoint 不受影响。
