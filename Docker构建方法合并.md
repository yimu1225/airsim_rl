# UE4.18.3 + AirSim Docker 构建指南

在阿里云无影云电脑（Ubuntu 22.04）上使用 Docker + GPU 构建 UE4.18.3 和 AirSim。

---

## 1. 安装 Docker

```bash
# 卸载旧版本
sudo apt remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc || true

# 安装依赖并添加官方源
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

# 安装 Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker

# 将当前用户加入 docker 组
sudo usermod -aG docker $USER
newgrp docker

# 验证
docker run hello-world
```

---

## 2. 配置阿里云镜像加速

在阿里云 ACR 控制台获取镜像加速器地址，然后：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://8gwcir6l.mirror.aliyuncs.com"
  ]
}
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
```

验证：
```bash
docker pull ubuntu:18.04
```

---

## 3. 安装代理（推荐）

如需访问 `nvidia.github.io`、GitHub 等外网资源，建议安装 Clash Verge Rev 等代理工具。

验证代理端口：
```bash
ss -ltnp | grep -E '7897|7898|7899'
```

测试外网访问：
```bash
curl -I -x http://127.0.0.1:7899 https://github.com
curl -I -x http://127.0.0.1:7899 https://nvidia.github.io/libnvidia-container/gpgkey
```

当前 shell 设置代理：
```bash
export http_proxy=http://127.0.0.1:7899
export https_proxy=http://127.0.0.1:7899
export HTTP_PROXY=http://127.0.0.1:7899
export HTTPS_PROXY=http://127.0.0.1:7899
```

---

## 4. 安装 NVIDIA Container Toolkit

```bash
# 添加 NVIDIA 仓库
sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 的 NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证：
```bash
docker info | grep -E "Runtimes|Default Runtime"
# 应包含 nvidia
```

---

## 5. 创建构建容器

### 5.1 准备工作目录

```bash
mkdir -p ~/airsim_rl/workspace/src
mkdir -p ~/airsim_rl/workspace/downloads
mkdir -p ~/airsim_rl/workspace/cache
```

### 5.2 拉取镜像并创建容器

```bash
docker pull ubuntu:18.04

docker run -it --name airsim_rl \
  --gpus all \
  --net host \
  --ipc host \
  --shm-size=16g \
  -v ~/airsim_rl/workspace:/workspace \
  -w /workspace \
  ubuntu:18.04 \
  bash
```

参数说明：
- `--gpus all`：允许容器使用 GPU
- `--net host`：使用宿主机网络
- `--ipc host`：共享 IPC
- `--shm-size=16g`：增大共享内存

### 5.3 验证 GPU

```bash
ls -l /dev/nvidia*
# 应看到 /dev/nvidia0、/dev/nvidiactl、/dev/nvidia-uvm
```

---

## 6. 容器内环境配置

### 6.1 安装基础依赖

```bash
apt update
apt install -y \
  build-essential \
  clang \
  cmake \
  ninja-build \
  git git-lfs \
  curl wget unzip zip rsync dos2unix \
  python3 python3-pip python3-dev python3-setuptools \
  mono-complete \
  libvulkan1 \
  libglu1-mesa-dev \
  libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  libxss1 libgtk2.0-0 libgtk-3-0 libnss3 libasound2 libpulse0 \
  libxcomposite1 libxdamage1 libxfixes3 libxtst6 \
  ca-certificates \
  sudo mono-dmcs

git lfs install
```

### 6.2 创建普通用户

UE4 的 `UnrealHeaderTool` 拒绝以 root 用户运行：

```bash
useradd -m -s /bin/bash yimu
usermod -aG sudo yimu
echo 'yimu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
chown -R yimu:yimu /workspace
su - yimu
cd /workspace
```

### 6.3 切换 clang 到 5.0

```bash
sudo apt install -y clang-5.0 lldb-5.0 lld-5.0
sudo ln -sf /usr/bin/clang-5.0 /usr/bin/clang
sudo ln -sf /usr/bin/clang++-5.0 /usr/bin/clang++
clang --version
```

---

## 7. 编译 UE4.18.3

### 7.1 拉取源码

```bash
cd /workspace/src
git clone --branch 4.18.3-release https://github.com/EpicGames/UnrealEngine.git
```

> GitHub 账号需绑定 Epic，否则无法访问该仓库。

### 7.2 修复 Commit.gitdeps.xml

从 [UE4.18.3 Release](https://github.com/EpicGames/UnrealEngine/releases/tag/4.18.3-release) 下载 `Commit.gitdeps.xml`，放入宿主机 `~/airsim_rl/workspace/downloads/`，然后在容器内执行：

```bash
cd /workspace/src/UnrealEngine
cp Engine/Build/Commit.gitdeps.xml Engine/Build/Commit.gitdeps.xml.bak
cp /workspace/downloads/Commit.gitdeps.xml Engine/Build/Commit.gitdeps.xml
```

### 7.3 修复 PhysX 库链接问题

```bash
cd /workspace/src/UnrealEngine/Engine/Source/ThirdParty

# 创建软链接（推荐，节省磁盘空间）
ln -s ../PhysX/Lib PhysX3/Lib

# 验证链接成功
ls -la PhysX3/
ls PhysX3/Lib/Linux/x86_64-unknown-linux-gnu/*.a | head -5
```

### 7.4 修复 mono-dmcs 问题

在 `Engine/Build/BatchFiles/Linux/Setup.sh` 中将 `mono-dmcs` 修改为 `mono-mcs`。

确保已经安装了 mono-mcs：
```bash
sudo apt-get install mono-mcs
```

### 7.5 编译

```bash
./Setup.sh
./GenerateProjectFiles.sh
make -j4
```

> 如果首次编译，建议先用 `-j1` 或 `-j4` 确保稳定。

### 7.6 编译 ShaderCompileWorker（必需）

```bash
make ShaderCompileWorker UnrealLightmass UnrealPak
```

---

## 8. 编译 AirSim

### 8.1 获取源码

将定制版 AirSim 源码放入容器，例如：
```
/workspace/AirSim-1-rpc-base-reset-plus-energy
```

### 8.2 修复 setup.sh 问题

**问题 1：高模车辆资源下载失败**

```bash
./setup.sh --no-full-poly-car
```

**问题 2：Eigen 下载地址失效**

将 `setup.sh` 中的 Eigen 下载地址改为 GitLab 可用归档地址：
```
https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
```

### 8.3 编译

```bash
./setup.sh --no-full-poly-car
./build.sh
```

成功标志：
```text
AirSim plugin is built!
```

---

## 9. 接入 UE4 工程

### 9.1 准备工程目录

将 `airlearning-ue4` 工程放入容器：
```
/workspace/airlearning-ue4
```

### 9.2 复制 AirSim 插件

```bash
cp -r /workspace/AirSim-1-rpc-base-reset-plus-energy/Unreal/Plugins /workspace/airlearning-ue4/
```

完成后应存在：
```
/workspace/airlearning-ue4/Plugins/AirSim
```

### 9.3 编译工程

**重要**：该工程的 target 名不是 `AirLearningEditor`，而是 `JsonParsing18VersionEditor`。

```bash
cd /workspace/src/UnrealEngine
./Engine/Build/BatchFiles/Linux/Build.sh JsonParsing18VersionEditor Linux Development /workspace/airlearning-ue4/AirLearning.uproject -waitmutex
```

检查产物：
```bash
find /workspace/airlearning-ue4 -path "*Binaries/Linux/*.so" | sort
find /workspace/airlearning-ue4/Plugins/AirSim -path "*Binaries/Linux/*.so" | sort
```

应生成：
- `libUE4Editor-JsonParsing18Version.so`
- `libUE4Editor-AirSim.so`

---

## 10. 运行 UE4Editor

### 10.1 确保图形环境

```bash
sudo apt install -y mesa-utils x11-xserver-utils x11-utils
echo $DISPLAY
# 应输出 :1 或类似值

xrandr
glxinfo -B
# 应正常返回显示器和 NVIDIA OpenGL 信息
```

### 10.2 启动编辑器

```bash
export DISPLAY=:1
SDL_VIDEODRIVER=x11 SDL_AUDIODRIVER=dummy \
  ./Engine/Binaries/Linux/UE4Editor /workspace/airlearning-ue4/AirLearning.uproject \
  -opengl4 -nosplash -windowed -ResX=1280 -ResY=720 -log
```

游戏模式启动：
```bash
export DISPLAY=:1 && SDL_VIDEODRIVER=x11 SDL_AUDIODRIVER=dummy \
  ./UnrealEngine-staging-4.18/Engine/Binaries/Linux/UE4Editor \
  /home/admin/DRL_Project/airlearning-ue4/AirLearning.uproject \
  -game -windowed -ResX=640 -ResY=480 -nosound -noaudio
```

---

## 11. 容器常用操作

```bash
# 退出容器
exit

# 重新进入容器（宿主机执行）
docker start airsim_rl
docker exec -it airsim_rl bash

# 以 yimu 用户进入
docker exec -it -u yimu airsim_rl bash
```

---

## 12. 常见问题

### 问题 1：`docker run hello-world` 拉取失败

现象：`context deadline exceeded`、`EOF`

解决：配置阿里云镜像加速器（见第 2 节）。

### 问题 2：`nvidia.github.io` 无法访问

现象：`Connection reset by peer`、`curl: (35)`

解决：安装代理软件并配置代理（见第 3 节）。

### 问题 3：`Refusing to run with the root privileges`

现象：
```text
Refusing to run with the root privileges.
```

解决：创建普通用户并切换（见第 6.2 节）。如果之前用 root 编译过，需清理残留：

```bash
rm -f Engine/Binaries/Linux/UnrealHeaderTool
rm -f Engine/Binaries/Linux/libUnrealHeaderTool-*.so
rm -rf Engine/Intermediate/Build/Linux
```

### 问题 4：`Couldn't find target rules file for target 'AirLearningEditor'`

现象：工程编译时找不到 target。

解决：该工程实际 target 名为 `JsonParsing18VersionEditor`，不是 `AirLearningEditor`（见第 9.3 节）。

### 问题 5：AirSim setup.sh 下载失败

**Eigen 404**：替换 `setup.sh` 中的 Bitbucket 地址为 GitLab 归档地址。

**高模车资源 SSL 失败**：使用 `--no-full-poly-car` 参数跳过。

### 问题 6：UE4Editor 进程存在但窗口不显示

现象：
- `nvidia-smi` 显示进程占用 GPU
- 日志显示工程已加载
- 但看不到窗口

解决：确保启动 shell 有正确的图形环境变量：

```bash
export DISPLAY=:1
SDL_VIDEODRIVER=x11 SDL_AUDIODRIVER=dummy ./Engine/Binaries/Linux/UE4Editor ...
```

### 问题 7：模块缺失

现象：
```text
Incompatible or missing module: libUE4Editor-JsonParsing18Version.so
```

解决：执行工程编译命令（见第 9.3 节）。

### 问题 8：PhysX 库链接问题

现象：编译时找不到 PhysX3 库文件。

解决：创建 PhysX 库的软链接（见第 7.3 节）。

### 问题 9：mono-dmcs 命令不存在

现象：Setup.sh 执行失败，提示 mono-dmcs 命令不存在。

解决：修改 Setup.sh 中的 mono-dmcs 为 mono-mcs（见第 7.4 节）。

---

## 13. 性能优化建议

### 13.1 编译优化

- 使用 `make -j$(nproc)` 充分利用多核 CPU
- 首次编译建议使用较低并行度 `-j4` 避免内存不足
- 增量编译只重新编译修改的部分

### 13.2 运行优化

- 确保有足够的 RAM（建议 16GB+）
- 使用 SSD 存储加速编译
- 游戏模式下可以使用 `-nosound -noaudio` 减少资源占用

### 13.3 GPU 加速

- 确保 NVIDIA 驱动正确安装
- 使用 `-opengl4` 参数启用 OpenGL 4.x

---

## 14. 备份和恢复

### 14.1 备份已编译产物

```bash
# 备份 UE4
cp -r ~/airsim_rl/workspace/src/UnrealEngine ~/airsim_rl/backup/

# 备份 AirSim
cp -r ~/airsim_rl/workspace/AirSim-1-rpc-base-reset-plus-energy ~/airsim_rl/backup/

# 备份工程
cp -r ~/airsim_rl/workspace/airlearning-ue4 ~/airsim_rl/backup/
```

### 14.2 增量编译

如果只修改了代码，可以使用增量编译：
```bash
cd ~/airsim_rl/workspace/src/UnrealEngine
make -j$(nproc)  # 只重新编译修改的部分
```</content>
<parameter name="filePath">/home/yimu/airsim_rl/Docker构建方法合并.md