#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关于 Level 3 动态障碍物的说明

问题：为什么 Level 3 打开 UE4 后没有看到动态障碍物？

原因：
NumberOfDynamicObjects 参数只是通过 JSON 配置文件传递给 UE4，但 UE4 端需要有相应的
AirLearning 插件代码来实际生成和控制动态障碍物。

参考原始 AirLearning 项目：
https://github.com/harvard-edge/airlearning

原始项目中，动态障碍物的实现包括：
1. UE4 蓝图/C++ 代码读取 NumberOfDynamicObjects 参数
2. 在场景中生成指定数量的动态障碍物 Actor
3. 为每个动态障碍物设置移动逻辑（随机移动、速度控制等）

当前项目状态：
- Python 端已配置 NumberOfDynamicObjects 参数（在 settings.py 的 dynamic_obstacles_dic 中）
- 该参数会被写入 EnvGenConfig.json 文件
- UE4 端通过 resetUnreal RPC 调用读取 JSON 配置
- 但 UE4 项目中可能没有实现动态障碍物的生成逻辑

解决方案：
1. 检查 UE4 项目是否有 DynamicObstacle 相关的蓝图或 C++ 类
2. 如果没有，需要参考原始 AirLearning 项目实现该功能
3. 或者使用其他方式（如 AirSim 的 simSpawnObject API）动态生成障碍物

临时替代方案：
如果 UE4 端不支持动态障碍物，Level 3 实际上和 Level 2 类似，只是配置参数不同。
您仍然可以使用 Level 0-2 来展示课程学习的效果。
"""

print(__doc__)
