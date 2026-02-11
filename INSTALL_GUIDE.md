# 环境安装指南 (AirSim RL Project)

本文档记录了如何在 Linux 环境下从零开始构建基于 CUDA 12.8 和 PyTorch 2.7.0 的运行环境，并成功编译安装 `causal-conv1d`。

## 1. 环境初始化

为了避免旧环境的缓存冲突，建议先彻底清理并创建新环境。

```bash
# 1. 删除旧环境（如果存在）
conda env remove -n AirSim

# 2. 清理 Conda 和 Pip 缓存 (可选，但推荐)
conda clean --all -y
pip cache purge

# 3. 创建新环境 (Python 3.9)
conda create -n AirSim python=3.9 -y

# 4. 激活环境
conda activate AirSim
```

## 2. 安装 CUDA 工具包与构建工具

这是最关键的一步。为了防止自动安装不兼容的 CUDA 13.x 版本，必须指定 `nvidia/label/cuda-12.8.0` 通道。同时安装 `ninja` 以加速后续编译。

```bash
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit=12.8 cuda-nvcc=12.8 ninja packaging -y
```

## 3. 安装 PyTorch

安装适配 CUDA 12.8 的 PyTorch 版本 (2.7.0)。

```bash
# 请根据实际情况调整 index-url，这里假设使用官方源或镜像源
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

## 4. 编译安装 Causal Conv1d

由于 `causal-conv1d` 需要从源码编译，且对环境变量敏感，请严格按照以下步骤操作。

### 4.1 清理干扰环境变量
某些 Conda 环境设置可能会干扰 `nvcc` 编译器。

```bash
unset NVCC_PREPEND_FLAGS
unset CC
unset CXX
```

### 4.2 安装库
使用 `--no-build-isolation` 标志，强制使用当前环境中的 PyTorch 和 Ninja 进行编译，避免版本冲突。

```bash
# 确保 CUDA 架构列表包含你的 GPU 算力 (例如 RTX 30/40 系列通常包含 8.6, 8.9, 9.0)
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"

pip install causal-conv1d>=1.4.0 --no-build-isolation --verbose
```

## 5. 修复 Python 3.9 兼容性问题 (重要)

如果你使用的是 Python 3.9，`causal-conv1d` (v1.6.0) 的源码包含 Python 3.10+ 的语法 (`|` 联合类型)，会导致 `RuntimeError` 或 `TypeError`。安装完成后，**必须**运行以下修复脚本。

创建一个修复脚本 `patch_causal_conv1d.sh`:

```bash
#!/bin/bash
# 获取 site-packages 路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
TARGET_DIR="$SITE_PACKAGES/causal_conv1d"

echo "正在修复文件路径: $TARGET_DIR"

FILES=("cpp_functions.py" "causal_conv1d_interface.py" "causal_conv1d_varlen.py")

for f in "${FILES[@]}"; do
    FILE_PATH="$TARGET_DIR/$f"
    if [ -f "$FILE_PATH" ]; then
        echo "正在修复: $f"
        # 1. 在文件头部添加 future import
        sed -i '1s/^/from __future__ import annotations\n/' "$FILE_PATH"
        # 2. 引入 Optional
        sed -i '2i from typing import Optional' "$FILE_PATH"
        # 3. 替换新语法为旧语法
        sed -i 's/torch.Tensor | None/Optional[torch.Tensor]/g' "$FILE_PATH"
        sed -i 's/int | None/Optional[int]/g' "$FILE_PATH"
    else
        echo "警告: 未找到文件 $FILE_PATH"
    fi
done

echo "修复完成！"
```

运行修复：

```bash
chmod +x patch_causal_conv1d.sh
./patch_causal_conv1d.sh
```

## 6. 安装 Mamba SSM

Mamba SSM 的安装需要特殊的编译配置，特别是在 Conda 环境下，编译器可能找不到 CUDA 头文件。

```bash
# 1. 设置 CUDA 编译器路径和库路径 (解决 fatal error: cuda_runtime_api.h: No such file)
# 注意：Conda 的 cuda-toolkit 12.8 将头文件放在 targets/x86_64-linux/include 下
export CONDA_CUDA_ROOT="$CONDA_PREFIX/targets/x86_64-linux"
export CPATH="$CONDA_CUDA_ROOT/include:$CPATH"
export LIBRARY_PATH="$CONDA_CUDA_ROOT/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_CUDA_ROOT/lib:$LD_LIBRARY_PATH"

# 2. 设置其他必要的环境变量
unset NVCC_PREPEND_FLAGS CC CXX
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0"  # 根据你的显卡调整，RTX 50 系列需要 10.0
export MAMBA_FORCE_BUILD=TRUE  # 强制本地编译，跳过 GitHub 轮子下载（解决网络超时）
export MAX_JOBS=4              # 限制并发数防止 OOM

# 3. 安装
pip install mamba-ssm --no-build-isolation --no-cache-dir --verbose
```

## 7. 验证安装

```bash
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}'); import causal_conv1d; print(f'Causal Conv1d: {causal_conv1d.__version__}'); import mamba_ssm; print(f'Mamba SSM: {mamba_ssm.__version__}')"
```

如果输出类似以下内容，说明安装成功：
- Torch: 2.7.0+cu128
- CUDA: 12.8
- Causal Conv1d: 1.6.0
- Mamba SSM: 2.3.0

## 8. 安装 Selective Scan (VMamba 依赖)

Selective Scan 是 VMamba 模型的核心组件，需要从源码编译。如果遇到 CUDA 版本不匹配或 CUB 库兼容性问题，请按照以下步骤修复。

### 8.1 常见问题与修复

#### CUDA 版本不匹配
如果报错 `RuntimeError: The detected CUDA version (13.0) mismatches the version that was used to compile PyTorch (12.8)`：

修改 `vmamba/kernels/selective_scan/setup.py`，在 `import torch` 后添加：

```python
# Monkey-patch to bypass strictly matching CUDA version check
# System has CUDA 13.0, PyTorch has 12.8
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None
```

#### CUB 库兼容性问题
如果报错 `error: namespace "cub" has no member "LaneId"` 或 `CTA_SYNC`：

修改 `vmamba/kernels/selective_scan/csrc/selective_scan/reverse_scan.cuh`：

1. 将 `lane_id(cub::LaneId())` 替换为 `lane_id(threadIdx.x & 0x1f)`
2. 将 `cub::CTA_SYNC();` 替换为 `__syncthreads();`

### 8.2 安装命令

```bash
# 进入 selective_scan 目录
cd vmamba/kernels/selective_scan

# 使用 --no-build-isolation 强制使用当前环境
pip install . --no-build-isolation
```

### 8.3 验证安装

```bash
 
```
