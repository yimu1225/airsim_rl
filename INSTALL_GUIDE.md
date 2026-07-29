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

## 4. 编译安装 Vim 配套 CUDA 依赖

VSSM 的空间编码器需要仓库内的 BiMamba-v2 定制实现。必须使用
`causal-conv1d 1.4.0` 与 `mamba_ssm 1.1.1` 这一组配套源码；不要用
通用的 `causal-conv1d 1.6.x` 或 `mamba_ssm 2.x` 替换。

Conda CUDA 12.8 的完整 toolkit 根目录位于
`targets/x86_64-linux`。RTX 50 系列（Blackwell）的计算能力为 12.0。

```bash
unset NVCC_PREPEND_FLAGS CC CXX
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"  # 其他显卡请改成实际计算能力
export MAX_JOBS="$(nproc)"

# 使用与 Vim 兼容、且已针对当前 GPU 架构调整过的 v1.4.0 源码。
cd /home/yimu/airsim_rl/Vim/causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  python -m pip install -e . --no-build-isolation

cd /home/yimu/airsim_rl/Vim/mamba-1p1p1
python -m pip install -e . --no-build-isolation
```

当前机器已按要求使用全部 32 核成功编译；若其他机器内存不足，可在安装前将
`MAX_JOBS` 调低为 4。

## 5. 验证安装

```bash
python -c "import inspect, causal_conv1d, mamba_ssm; from mamba_ssm import Mamba; print('causal-conv1d:', causal_conv1d.__version__); print('mamba-ssm:', mamba_ssm.__version__); print(inspect.signature(Mamba.__init__))"
```

正确结果应为 `causal-conv1d 1.4.0`、`mamba-ssm 1.1.1`，且构造签名中
包含 `bimamba_type`、`if_divide_out` 和 `init_layer_scale`。

## 6. 安装 Selective Scan (VMamba 依赖)

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
