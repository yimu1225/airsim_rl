# AirSim 距离传感器补丁迁移说明

本项目现在已经从激光雷达改为使用 36 个水平面均匀分布的 AirSim 距离传感器，用于障碍物奖励惩罚和起飞障碍检查。

只要某台电脑需要运行 UE4 仿真环境，就必须让那台电脑的 AirSim UE4 插件包含距离传感器补丁。训练端 Python 只负责通过 RPC 读取数据，不负责加载 UE4 插件。

## 需要补丁的内容

UE4 项目的 `Plugins/AirSim` 需要加入这些源码改动：

- 增加 RPC 接口：`getDistanceSensorData(sensor_name, vehicle_name)`；
- 让距离传感器读取 `settings.json` 中的 `X/Y/Z`、`Yaw/Pitch/Roll`、`MinDistance`、`MaxDistance`、`DrawDebugPoints`；
- 当 `DrawDebugPoints` 为 `true` 时，在 UE4 中显示距离传感器射线。

这些改动已经写成自动脚本，不需要手动改 C++ 源码。

## 在新机器上自动打补丁

在这个 RL 项目目录下运行：

```bash
python scripts/patch_airsim_distance_sensor_plugin.py --airsim-plugin /path/to/YourUEProject/Plugins/AirSim
```

也可以直接传 UE4 项目根目录：

```bash
python scripts/patch_airsim_distance_sensor_plugin.py --airsim-plugin /path/to/YourUEProject
```

脚本会自动寻找 `Plugins/AirSim`，并修改需要的源码文件。

脚本会给被修改的文件创建备份，备份后缀是：

```text
.bak_distance_sensor_patch
```

脚本可以重复运行。如果已经打过补丁，它会提示不需要修改。

## Windows UE4 + WSL2 训练

这种情况是：UE4/AirLearning 在 Windows 中运行，训练代码在 WSL2 里运行。

这时只需要给 Windows 里的 UE4 项目插件打补丁。WSL2 里的 Python 不需要 AirSim 插件源码，它只是通过 RPC 连接 Windows UE4。

1. 在 WSL2 中运行补丁脚本：

   ```bash
   python scripts/patch_airsim_distance_sensor_plugin.py \
     --airsim-plugin /mnt/d/Projects/airlearning-ue4-1/Plugins/AirSim
   ```

2. 完全关闭 UE4Editor。

3. 重新编译 Windows UE4 项目插件：

   ```powershell
   Set-Location 'D:\Projects\airlearning-ue4-1'
   & 'D:\SoftWare\Epic Games\Game\UE_4.18\Engine\Binaries\DotNET\UnrealBuildTool.exe' `
     JsonParsing18VersionEditor Win64 Development `
     -project='D:\Projects\airlearning-ue4-1\AirLearning.uproject' `
     -waitmutex -progress
   ```

4. 确认 `settings.json` 放在：

   ```text
   D:\Users\35281\Documents\AirSim\settings.json
   ```

5. 启动 UE4/AirLearning，然后在 WSL2 中运行训练。

注意：这种模式下不需要在 WSL2 里编译 Linux 版 AirSim 插件。

## 原生 Ubuntu UE4 + Ubuntu 训练

这种情况是：UE4 仿真环境本身也在 Ubuntu 中运行。

1. 把这个 RL 项目复制到 Ubuntu 机器上。

2. 给 Ubuntu 上的 UE4 项目插件打补丁：

   ```bash
   python scripts/patch_airsim_distance_sensor_plugin.py \
     --airsim-plugin ~/YourUEProject/Plugins/AirSim
   ```

3. 在 Ubuntu 上使用那台机器自己的 Unreal Engine 重新编译 UE4 项目。

4. 确认 `settings.json` 放在：

   ```text
   ~/Documents/AirSim/settings.json
   ```

5. 启动 Ubuntu 上的 UE4 仿真器，然后运行训练。

重要：Windows 编译出来的 `Plugins/AirSim/Binaries/Win64/*.dll` 不能直接拿到 Ubuntu 用。Ubuntu 必须用源码重新编译，生成 Linux 对应的 `.so` 文件。

## 生成 36 个距离传感器配置

生成带 36 个水平距离传感器的 `settings.json`：

```bash
python scripts/generate_distance_sensor_settings.py --count 36
```

如果想在 UE4 里显示射线：

```bash
python scripts/generate_distance_sensor_settings.py --count 36 --draw_debug_points
```

显示规则：

- 红线：传感器命中障碍物；
- 绿线：没有命中，射到最大距离；
- 黄点：命中点。

正式训练时，如果不需要观察射线，建议关闭 `DrawDebugPoints`，这样更省性能。

## 惩罚距离由 MaxDistance 决定

当前 RL 环境已经删除了额外的距离传感器安全距离参数，不再从命令行单独设置惩罚距离。

距离传感器的惩罚范围直接来自 `settings.json` 中每个传感器的：

```json
"MaxDistance": 1.0
```

如果 `MaxDistance` 设置为 `1.0`，含义就是：

- 传感器只检测 1 米以内的障碍物；
- 奖励函数只对 1 米以内的返回距离进行障碍惩罚；
- 起飞障碍检查也使用这个 1 米范围；
- 返回值等于 `MaxDistance` 时视为没有进入惩罚范围。

所以多台机器同步时，只要保证各自的 `settings.json` 中 `MaxDistance` 一致，惩罚距离就一致。

## 最重要的规则

可以跨机器复制或自动补丁源码，但不要跨系统复制编译产物。

- Windows UE4 使用 Windows 上重新编译出来的 `.dll`；
- Ubuntu UE4 使用 Ubuntu 上重新编译出来的 `.so`；
- WSL2 训练连接 Windows UE4 时，WSL2 不需要 UE4 插件编译产物。

简单说：

```text
源码可以迁移，二进制编译结果不能跨系统迁移。
```
