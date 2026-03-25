# UE4.18.3 + AirSim Ubuntu 22.04 构建指南

在 Ubuntu 22.04 上直接构建 UE4.18.3 和 AirSim，无需 Docker。

---

## 1. 系统准备

### 1.1 更新系统

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl wget git git-lfs unzip zip rsync dos2unix
```

### 1.2 安装图形相关依赖

```bash
sudo apt install -y \
  libvulkan1 mesa-utils x11-xserver-utils x11-utils \
  libglu1-mesa-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  libxss1 libgtk2.0-0 libgtk-3-0 libnss3 libasound2 libpulse0 \
  libxcomposite1 libxdamage1 libxfixes3 libxtst6 \
  ca-certificates
```

### 1.3 安装 NVIDIA 驱动和 CUDA（如果有 GPU）

```bash
# 安装 NVIDIA 驱动
sudo apt install -y nvidia-driver-470

# 安装 CUDA 工具包（可选，用于 GPU 加速）
# 从 NVIDIA 官网下载 CUDA 10.0 运行文件
# wget https://developer.download.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux.run
# sudo sh cuda_10.0.130_410.48_linux.run --no-opengl-libs
```

### 1.4 安装代理（推荐）

如需访问 GitHub 等外网资源，建议安装代理工具（如 Clash Verge Rev）。

验证代理端口：
```bash
ss -ltnp | grep -E '7897|7898|7899'
```

设置代理：
```bash
export http_proxy=http://127.0.0.1:7899
export https_proxy=http://127.0.0.1:7899
export HTTP_PROXY=http://127.0.0.1:7899
export HTTPS_PROXY=http://127.0.0.1:7899
```

---

## 2. 安装编译工具

### 2.1 安装 CMake 和 Ninja

```bash
sudo apt install -y cmake ninja-build
```



### 2.3 安装 Mono

```bash
sudo apt install -y mono-complete mono-mcs
```

### 2.4 安装 Python

```bash
sudo apt install -y python3 python3-pip python3-dev python3-setuptools
git lfs install
```


## 4. 编译 UE4.18.3

### 4.1 拉取源码

```bash
cd ~/airsim_rl/src
git clone --branch 4.18.3-release https://github.com/EpicGames/UnrealEngine.git
```

> GitHub 账号需绑定 Epic，否则无法访问该仓库。

### 4.2 修复 Commit.gitdeps.xml

从 [UE4.18.3 Release](https://github.com/EpicGames/UnrealEngine/releases/tag/4.18.3-release) 下载 `Commit.gitdeps.xml`，放入 `~/airsim_rl/downloads/`，然后执行：

```bash
cd ~/airsim_rl/src/UnrealEngine
cp Engine/Build/Commit.gitdeps.xml Engine/Build/Commit.gitdeps.xml.bak
cp ~/airsim_rl/downloads/Commit.gitdeps.xml Engine/Build/Commit.gitdeps.xml
```

### 4.3 修复 PhysX 库链接问题

```bash
cd ~/airsim_rl/src/UnrealEngine/Engine/Source/ThirdParty

# 创建软链接（推荐，节省磁盘空间）
ln -s ../PhysX/Lib PhysX3/Lib

# 验证链接成功
ls -la PhysX3/
ls PhysX3/Lib/Linux/x86_64-unknown-linux-gnu/*.a | head -5
```

### 4.4 修复 mono-dmcs 问题

在 `Engine/Build/BatchFiles/Linux/Setup.sh` 中将 `mono-dmcs` 修改为 `mono-mcs`。

### 4.5 编译

```bash
cd ~/airsim_rl/src/UnrealEngine
./Setup.sh
./GenerateProjectFiles.sh
make -j$(nproc)
```

> 如果首次编译，建议先用 `-j4` 或 `-j$(nproc)` 确保稳定。

### 4.6 编译 ShaderCompileWorker（必需）

```bash
make ShaderCompileWorker UnrealLightmass UnrealPak
```

---

## 5. 编译 AirSim

### 5.1 获取源码

将定制版 AirSim 源码放入项目目录，例如：
```
~/airsim_rl/AirSim-1-rpc-base-reset-plus-energy
```

### 5.2 修复 setup.sh 问题

**问题 1：高模车辆资源下载失败**

修改 `setup.sh` 使用 `--no-full-poly-car` 参数。

**问题 2：Eigen 下载地址失效**

将 `setup.sh` 中的 Eigen 下载地址改为 GitLab 可用归档地址：
```
https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
```

### 5.3 编译

```bash
cd ~/airsim_rl/AirSim-1-rpc-base-reset-plus-energy
./setup.sh --no-full-poly-car
./build.sh
```

成功标志：
```text
AirSim plugin is built!
```

---

## 6. 接入 UE4 工程

### 6.1 准备工程目录

将 `airlearning-ue4` 工程放入项目目录：
```
~/airsim_rl/airlearning-ue4
```

### 6.2 复制 AirSim 插件

```bash
cp -r ~/airsim_rl/AirSim-1-rpc-base-reset-plus-energy/Unreal/Plugins ~/airsim_rl/airlearning-ue4/
```

完成后应存在：
```
~/airsim_rl/airlearning-ue4/Plugins/AirSim
```

### 6.3 编译工程

**重要**：该工程的 target 名不是 `AirLearningEditor`，而是 `JsonParsing18VersionEditor`。

```bash
cd ~/airsim_rl/src/UnrealEngine
./Engine/Build/BatchFiles/Linux/Build.sh JsonParsing18VersionEditor Linux Development ~/airsim_rl/airlearning-ue4/AirLearning.uproject -waitmutex
```

检查产物：
```bash
find ~/airsim_rl/airlearning-ue4 -path "*Binaries/Linux/*.so" | sort
find ~/airsim_rl/airlearning-ue4/Plugins/AirSim -path "*Binaries/Linux/*.so" | sort
```

应生成：
- `libUE4Editor-JsonParsing18Version.so`
- `libUE4Editor-AirSim.so`

---

## 7. 运行 UE4Editor

### 7.1 确保图形环境

```bash
echo $DISPLAY
# 应输出 :0 或类似值

xrandr
glxinfo -B
# 应正常返回显示器和 OpenGL 信息
```

### 7.2 启动编辑器

```bash
cd ~/airsim_rl/src/UnrealEngine
export DISPLAY=:0
SDL_VIDEODRIVER=x11 SDL_AUDIODRIVER=dummy \
  ./Engine/Binaries/Linux/UE4Editor ~/airsim_rl/airlearning-ue4/AirLearning.uproject \
  -opengl4 -nosplash -windowed -ResX=1280 -ResY=720 -log
```

游戏模式启动：
```bash
export DISPLAY=:0
SDL_VIDEODRIVER=x11 SDL_AUDIODRIVER=dummy \
  ./Engine/Binaries/Linux/UE4Editor ~/airsim_rl/airlearning-ue4/AirLearning.uproject \
  -game -windowed -ResX=640 -ResY=480 -nosound -noaudio
```

---

## 8. 常见问题

### 问题 1：`Couldn't find target rules file for target 'AirLearningEditor'`

解决：使用正确的 target 名 `JsonParsing18VersionEditor`（见第 6.3 节）。

### 问题 2：AirSim setup.sh 下载失败

**Eigen 404**：替换为 GitLab 归档地址（见第 5.2 节）。

**高模车资源 SSL 失败**：使用 `--no-full-poly-car` 参数（见第 5.2 节）。

### 问题 3：UE4Editor 进程存在但窗口不显示

现象：
- `nvidia-smi` 显示进程占用 GPU
- 日志显示工程已加载
- 但看不到窗口

解决：确保 DISPLAY 环境变量正确设置（见第 7.2 节）。

### 问题 4：模块缺失

现象：
```text
Incompatible or missing module: libUE4Editor-JsonParsing18Version.so
```

解决：执行工程编译命令（见第 6.3 节）。

### 问题 5：PhysX 库链接问题

解决：创建 PhysX3/Lib 到 PhysX/Lib 的软链接（见第 4.3 节）。

### 问题 6：mono-dmcs 命令不存在

解决：修改为使用 mono-mcs（见第 4.4 节）。

---

## 9. 环境变量设置

为方便使用，可以在 `~/.bashrc` 中添加：

```bash
export AIRSIM_RL_HOME=~/airsim_rl
export UE4_ROOT=$AIRSIM_RL_HOME/src/UnrealEngine
export PATH=$UE4_ROOT/Engine/Binaries/Linux:$PATH
```

然后重新加载：
```bash
source ~/.bashrc
```

---

## 10. 性能优化建议

### 10.1 编译优化

- 使用 `make -j$(nproc)` 充分利用多核 CPU
- 首次编译建议使用较低并行度 `-j4` 避免内存不足

### 10.2 运行优化

- 确保有足够的 RAM（建议 16GB+）
- 使用 SSD 存储加速编译
- 游戏模式下可以使用 `-nosound -noaudio` 减少资源占用

### 10.3 GPU 加速

- 确保 NVIDIA 驱动正确安装
- 使用 `-opengl4` 参数启用 OpenGL 4.x

---

## 11. 备份和恢复

### 11.1 备份已编译产物

```bash
# 备份 UE4
cp -r ~/airsim_rl/src/UnrealEngine ~/airsim_rl/backup/

# 备份 AirSim
cp -r ~/airsim_rl/AirSim-1-rpc-base-reset-plus-energy ~/airsim_rl/backup/

# 备份工程
cp -r ~/airsim_rl/airlearning-ue4 ~/airsim_rl/backup/
```

### 11.2 增量编译

如果只修改了代码，可以使用增量编译：
```bash
cd ~/airsim_rl/src/UnrealEngine
make -j$(nproc)  # 只重新编译修改的部分
```</content>
<parameter name="filePath">/home/yimu/airsim_rl/Ubuntu 22.04 构建方法.md