# AirLearning Arena Meshes 修改说明

本文档记录 `airlearning-ue4-1` 中原 `BP_Arena_Meshes` 的 C++ 原生替代实现，以及静态/动态障碍物生成逻辑。代码位于 Unreal 项目的：

```text
Source/JsonParsing18Version/AirLearningArenaMeshes.cpp
Source/JsonParsing18Version/AirLearningArenaMeshes.h
Source/JsonParsing18Version/AirLearningObstacle.cpp
Source/JsonParsing18Version/AirLearningObstacle.h
```

## 修改目标

运行时从环境 JSON 读取场景参数，自动创建场地、终点、静态障碍物和动态障碍物，不再依赖蓝图 `BP_Arena_Meshes` 完成主要生成工作。

## 场地与环境配置

`GenerateFromJson()` 在 BeginPlay 时执行：

1. 清理上一回合生成的障碍物；
2. 读取环境 JSON；
3. 将米制位置和尺寸转换为 Unreal 厘米单位；
4. 调整地板、四面墙和生成区域；
5. 更新 GameMode 的起点、终点位置，并让 PlayerStart 朝向终点；
6. 按 JSON 中的随机种子生成本回合场景。

墙体使用动态材质，并根据 `WallsRGB` 设置颜色。终点网格从 `/Game/Models` 中查找名称为 `End` 的静态网格。

## 出生朝向随目标点更新（2026-09-15）

已修改 [AirLearningArenaMeshes.cpp:946](/home/yimu/YIMU/airlearning-ue4-1/Source/JsonParsing18Version/AirLearningArenaMeshes.cpp:946) 中的 `UpdateGameModeAndPlayerStart()`，在原来只设置起点位置的基础上，增加起点朝向设置。

每次 UE 调用 `GenerateFromJson()` 根据 JSON 重建环境时：

1. 读取 `PlayerStart` 和 `End`，转换为 `PlayerStartWorldPosition` 和 `EndWorldPosition`；
2. 计算从出生点指向目标点的方向，取其水平偏航角 yaw；
3. 设置第一个 `APlayerStart` 的位置和朝向，正常情况下 pitch、roll 均为 0。

对应实现如下：

```cpp
const FVector ToGoal = EndWorldPosition - PlayerStartWorldPosition;
const FRotator StartRotation = ToGoal.IsNearlyZero()
    ? It->GetActorRotation()
    : FRotator(0.0f, ToGoal.Rotation().Yaw, 0.0f);
It->SetActorLocation(PlayerStartWorldPosition);
It->SetActorRotation(StartRotation);
```

当目标与出生点的三维位置近似重合时，保留原先朝向，避免无意义旋转。此处修改的是 `APlayerStart` 的出生朝向，不是飞行过程中持续对准目标的控制逻辑。

生效范围：该函数在 UE 重建场景时更新起点；仅调用 AirSim `reset()` 的软重置不会重新执行此函数。已经生成的无人机是否采用新朝向，还取决于生成/重置流程是否使用更新后的 `APlayerStart`。

该修改已于 2026-09-15 完成源码更新和 Linux Development 编辑器目标编译，生成 `libUE4Editor-JsonParsing18Version.so`。编译成功不等同于运行时朝向验证；重启 UE 后，可更换不同方向的 `End` 并重建环境，检查出生朝向是否对应更新。

## 静态障碍物

- 从 `/Game/Models` 加载静态网格；
- 使用分层网格采样（stratified sampling）分散候选位置；
- 检查起点、终点和已放置障碍物之间的最小距离；
- 最多尝试 `MaxPlacementAttemptsPerObject` 次放置；
- 根据网格名称匹配材质/纹理，匹配不到时使用默认障碍物材质；
- 每次生成结束时输出实际生成数量，例如 `spawned 8/10 static`。

## 动态障碍物

当 `NumberOfDynamicObjects > 0` 时，动态障碍物会使用独立随机流生成。每个动态障碍物具有：

- 随机初始位置；
- 随机沿 X 或 Y 轴的运动方向；
- 随机正向或负向方向；
- 从 JSON `VelocityRange` 中采样的速度；
- 根据场地边界和障碍物尺寸计算出的可行运动路径。

路径生成前会在路径起点、中点和终点进行有效性检查，避免路径穿过起点、终点或已放置障碍物。运动范围由场地边界和 `DynamicObstacleTravelHalfRangeMeters` 共同限制。

动态障碍物通过 `InitializeDynamicMotion()` 接收方向、速度以及路径最小/最大中心点。具体的逐帧移动和到达边界后的处理由 `AirLearningObstacle` 完成。

如果 JSON 中 `NumberOfDynamicObjects` 为 0，动态障碍物不会生成。

## 重要参数

| 参数 | 含义 |
| --- | --- |
| `NumberOfObjects` | 静态障碍物数量 |
| `NumberOfDynamicObjects` | 动态障碍物数量 |
| `ArenaSize` | 场地长、宽、高，单位为米 |
| `MinimumDistance` | 障碍物之间的最小中心距离，单位为米 |
| `VelocityRange` | 动态障碍物速度范围，单位为米/秒；一个值表示最大速度 |
| `DynamicObstacleTravelHalfRangeMeters` | 动态障碍物路径半范围，单位为米 |
| `Seed` | 场景随机种子 |
| `WallsRGB` | 四面墙的颜色参数 |

## 生命周期与训练影响

每次环境重置时，`ClearSpawnedObstacles()` 会销毁上一回合生成的障碍物并清空位置记录，然后按照新的 JSON/随机种子重新生成。因此训练过程中不同 episode 可以拥有不同的障碍物布局和动态障碍物运动参数。

如果训练脚本在 episode 之间暂停仿真，动态障碍物也会随 Unreal 世界一起暂停；恢复仿真后继续按照其 Actor 的 Tick 逻辑运动。

## 编译

本机 UE 4.18 项目的编译命令：

```bash
/home/yimu/YIMU/UnrealEngine-4.18/Engine/Build/BatchFiles/Linux/Build.sh \
  JsonParsing18VersionEditor Linux Development \
  -project=/home/yimu/YIMU/airlearning-ue4-1/AirLearning.uproject \
  -waitmutex
```

修改 Unreal C++ 源码后，需要重新编译项目，使编辑器和运行时加载新的二进制模块。编译成功后，再启动 AirSim/Unreal 进行验证。建议检查日志中的：

```text
AirLearning obstacle generation OK: spawned ... static, ... dynamic obstacles
```

若显示 `INCOMPLETE`，说明部分障碍物在最大尝试次数内未找到合法位置，应减少障碍物数量/尺寸或放宽最小距离。
