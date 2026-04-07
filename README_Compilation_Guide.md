# AirSim 编译与环境搭建修复指南 (高版本 Clang 适配)

本文档记录了在较新的 Linux 系统（如 Ubuntu 22.04）上编译此自定义旧版本 AirSim 的完整修正步骤与踩坑记录。由于您需要适配基于高版本 Clang 成功编译的 UE 4.18，因此必须摒弃原有的低版本环境。

## 1. 核心问题背景

原版 AirSim (针对早期虚幻引擎) 脚本中硬编码强行依赖了 `clang-5.0` 和 `clang++-5.0`。而在当前的 Ubuntu 发行版中，通常内置的高版本编译器（如 Clang 14）。
如果不做修改直接运行，会导致以下两次致命失败：
1. **`setup.sh` / `build.sh` 找不到编译器**：系统只有 `clang` 而没有 `clang-5.0`。
2. **`cannot find -lstdc++` (C++ 测试编译失败)**：单纯将名称改为 `clang` 后，即使 Clang 没问题，由于缺少对应版本的 GCC C++ 后端基本库开发包（`libstdc++-dev` 系列），CMake 在第一步试编译阶段依然会阻断。

---

## 2. 编译准备工作

### 2.1 安装基础编译工具与依赖
为了解决 `cannot find -lstdc++` 链接器报错，需要对齐正确的 GNU/C++ 库。在终端执行以下命令：
```bash
sudo apt update
sudo apt install -y build-essential clang g++-12 libstdc++-12-dev
sudo apt-get install -y build-essential gcc g++ make cmake
sudo apt-get install -y python python-dev python-pip
sudo apt-get install -y libclang-dev libc++-dev lldb
sudo apt-get install -y libboost-all-dev

```
*(注：如果不安装 `libstdc++-12-dev` 和 `g++-12`，较高版本的系统由于依赖不全，Clang 无法正常工作。具体版本视系统而定，Ubuntu 22.04 常用 `12`。)*

### 2.2 修改 `setup.sh` 中的硬编码
解除 `setup.sh` 中对版本号的限制，使其使用系统默认的 clang。您可以使用 `sed` 批量替换，或手动编辑文件。
```bash
sed -i 's|/usr/local/opt/llvm-5.0/bin/clang-5.0|clang|g' setup.sh
sed -i 's|/usr/local/opt/llvm-5.0/bin/clang++-5.0|clang++|g' setup.sh

sed -i 's|sudo apt-get install -y clang-5.0 clang++-5.0|# sudo apt-get install -y clang-5.0 clang++-5.0|g' setup.sh
```

### 2.3 修改 `build.sh` 中的硬编码
同理，`build.sh` 脚本内部也有几处写死了带有 `5.0` 后缀的编译器。进行以下替换：
```bash
sed -i 's/clang++-5.0/clang++/g' build.sh
sed -i 's/clang-5.0/clang/g' build.sh
# 同样适用 sed 命令将其替换为默认包名。
```

---

## 3. 进行编译

### 3.1 运行依赖配置脚本
在清理硬编码和补齐底层开发库之后，执行配置脚本以下载必要文件（如 libc++, rpclib, eigen）：
```bash
./setup.sh
```
*预期结果：跑完一系列下载编译后，控制台绿字输出 `AirSim setup completed successfully!`，且根目录生成了 `llvm-build` 和 `llvm-source-50` 文件夹。*

### 3.2 运行构建脚本
接着执行主编译脚本，开始编译并链接静态库 AirLib 等组件：
```bash
./build.sh
```
*(如果需要清理编译，只需删除 `build_debug` 目录即可重来)*

---

## 4. 常见报错 QA 排查录

**Q1: CMake Error: The C / CXX compiler identification is unknown**
- **原因**：CMake 去寻找名为 `clang-5.0` 的可执行文件但未找到。
- **解决**：检查 `setup.sh` 或 `build.sh` 中是否还有遗漏的 `clang-5.0` 硬编码没清理干净，将其改成 `clang` / `clang++`。

**Q2: `/usr/bin/ld: cannot find -lstdc++: No such file or directory`**
- **原因**：虽然有高版本 Clang，但是 C++ 底阶动态链接库对应的 `-dev` 包缺失，因此 `clang++` 无法完成基础 C++ 文件的链接。
- **解决**：安装对应您系统的开发库关联：`sudo apt install -y libstdc++-12-dev g++-12`。

**Q3: 原本使用 LLVM 3.9/5.0 的 Unreal 兼容性问题**
- **备注**：由于 UE 4.18 在您的工程机上已确认可以使用 Clang 高版本正常工作，因此上述将默认编译器桥接升级是完全可行的，不需要降级 Clang。