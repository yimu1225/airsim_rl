#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境等级测试脚本
用于启动指定课程学习等级的UE4环境

使用方法：
1. 修改下面的 TEST_LEVEL 变量为需要的等级 (0, 1, 2, 3)
2. 直接运行脚本: python test_env_level.py

课程学习等级说明：
- Level 0: 简单环境 (easy_range_dic) - 小地图(30m), 障碍物少(10-20个)
- Level 1: 中等环境 (medium_range_dic) - 中等地图(40m), 障碍物中等(20-30个)
- Level 2: 困难环境 (hard_range_dic) - 大地图(60m), 障碍物多(50-60个)
- Level 3: 动态障碍物环境 (dynamic_obstacles_dic) - 大地图 + 1-5个动态障碍物
"""

import os
import sys
import time
import shutil
import json
import copy
import numpy as np

# ==================== 用户配置区域 ====================
# 修改这里的数字来切换测试等级: 0, 1, 2, 3
TEST_LEVEL = 2

# ====================================================

# 导入项目相关模块
from settings_folder import settings
from game_handling.game_handler_class import GameHandler
from environment_randomization.game_config_handler_class import GameConfigHandler
from gym_airsim.envs.airlearningclient import AirLearningClient

# 等级到配置字典的映射（使用 eval 解析 settings 中的字典）
LEVEL_CONFIG_MAP = {
    0: "settings.easy_range_dic",
    1: "settings.medium_range_dic",
    2: "settings.hard_range_dic",
    3: "settings.dynamic_obstacles_dic"
}

# 等级名称映射
LEVEL_NAMES = {
    0: "easy",
    1: "medium",
    2: "hard",
    3: "dynamic"
}

# 需要在采样时覆盖的完整参数列表（按 level 重建环境，避免残留旧值）
# 注意：之前缺少 VelocityRange，导致切换 level 后速度范围仍保留旧值
SAMPLE_VARS = [
    "Seed",
    "ArenaSize",
    "NumberOfObjects",
    "NumberOfDynamicObjects",
    "End",
    "Walls1",
    "MinimumDistance",
    "VelocityRange",
    "EnvType",
    "PlayerStart",
    "Name",
]

# Level 3 配置（本地定义，避免 settings.py 中缺失导致报错）
DYNAMIC_OBSTACLES_RANGE = {
    "End": ["Mutable"],
    "MinimumDistance": [2, 5],
    "EnvType": ["Indoor"],
    "ArenaSize": [[60, 60, 10]],
    "PlayerStart": [[0, 0, 0]],
    "NumberOfDynamicObjects": list(range(1, 6)),
    "Walls1": [[200, 13, 99], [255, 255, 10], [0, 10, 10], [10, 100, 100], [126, 11, 90]],
    "Seed": list(range(0, 10000)),
    "VelocityRange": [[0, 5]],
    "Name": ["Name"],
    "NumberOfObjects": list(range(50, 60))
}


def read_json_summary(file_path):
    """读取 JSON 并提取关键字段用于对比显示。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    gs = data.get("GameSetting", {})
    indoor = data.get("Indoor", [{}])[0] if isinstance(data.get("Indoor"), list) and data.get("Indoor") else {}
    return {
        "Seed": gs.get("Seed"),
        "ArenaSize": gs.get("ArenaSize"),
        "MinimumDistance": gs.get("MinimumDistance"),
        "VelocityRange": gs.get("VelocityRange"),
        "NumberOfDynamicObjects": gs.get("NumberOfDynamicObjects"),
        "End": gs.get("End"),
        "Walls1": gs.get("Walls1"),
        "NumberOfObjects": indoor.get("NumberOfObjects"),
        "EnvType": gs.get("EnvType"),
        "PlayerStart": gs.get("PlayerStart"),
        "Name": indoor.get("Name"),
    }


def print_summary(label, summary):
    """打印配置摘要。"""
    print(f"\n{label}")
    print("-" * 50)
    for k in ["Seed", "ArenaSize", "NumberOfObjects", "NumberOfDynamicObjects",
              "MinimumDistance", "VelocityRange", "End", "Walls1", "EnvType", "PlayerStart", "Name"]:
        v = summary.get(k)
        if v is not None:
            print(f"  {k}: {v}")


def save_json_snapshot(level_name, source_json_path):
    """保存本次运行生成的 JSON 快照，避免每次都只看到同一路径文件。"""
    snapshot_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "test_env_level_json"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"EnvGenConfig_{level_name}_{timestamp}_{os.getpid()}.json"
    snapshot_path = os.path.join(snapshot_dir, snapshot_name)

    shutil.copy2(source_json_path, snapshot_path)
    return snapshot_path


def main():
    """主函数"""
    print("=" * 60)
    print("AirSim 环境等级测试工具")
    print("=" * 60)

    # 验证等级
    if TEST_LEVEL not in LEVEL_CONFIG_MAP:
        print(f"错误: 无效的等级 {TEST_LEVEL}，请选择 0, 1, 2, 或 3")
        sys.exit(1)

    level_name = LEVEL_NAMES[TEST_LEVEL]
    config_name = LEVEL_CONFIG_MAP[TEST_LEVEL]

    print(f"测试等级: Level {TEST_LEVEL} ({level_name})")
    print(f"配置: {config_name}")
    print("-" * 60)

    # 记录修改前的 JSON 状态
    json_path = settings.json_file_addr
    print(f"目标 JSON 路径: {json_path}")
    before_summary = read_json_summary(json_path)
    print_summary("[修改前] 当前 JSON 配置:", before_summary)

    # 先生成并写入 JSON，确保 UE4 启动时读取的是本次新配置
    print(f"\n[1/4] 初始化环境配置并生成 JSON (Level {TEST_LEVEL})...")

    # 对 level 3 做特殊处理：如果 settings 中没有 dynamic_obstacles_dic，注入本地定义
    if TEST_LEVEL == 3:
        try:
            eval(config_name)
        except Exception:
            print("[WARN] settings.dynamic_obstacles_dic 不存在，使用脚本内置配置")
            import settings_folder.settings as _settings_module
            _settings_module.dynamic_obstacles_dic = DYNAMIC_OBSTACLES_RANGE
    game_config = GameConfigHandler(range_dic_name=config_name)

    # 随机化环境并写入 JSON
    print("随机化环境参数并写入 JSON...")
    run_seed = (time.time_ns() ^ (os.getpid() << 16)) & 0xFFFFFFFF
    np_random = np.random.RandomState(seed=run_seed)

    game_config.sample(*SAMPLE_VARS, np_random=np_random)

    # 显示修改后的状态
    after_summary = read_json_summary(json_path)
    print_summary("[修改后] 生成 JSON 配置:", after_summary)

    # 高亮显示哪些字段发生了变化
    changed = []
    for k in SAMPLE_VARS:
        b = before_summary.get(k)
        a = after_summary.get(k)
        if b != a:
            changed.append(f"  {k}: {b} -> {a}")
    if changed:
        print("\n[变化字段]")
        print("-" * 50)
        print("\n".join(changed))
    else:
        print("\n[WARN] 没有任何字段发生变化，请检查 JSON 路径或权限")

    snapshot_path = save_json_snapshot(level_name, json_path)
    print(f"\n本次 JSON 快照: {snapshot_path}")
    print("环境配置初始化完成")

    # 初始化游戏处理器（启动UE4）
    print("\n[2/4] 正在启动 UE4 环境...")
    game_handler = GameHandler()
    game_handler.restart_game()
    print("UE4 启动完成")

    # 等待游戏完全加载
    print("等待游戏稳定...")
    time.sleep(5)

    # 初始化 AirSim 客户端
    print("\n[3/4] 连接 AirSim 客户端...")
    client_ip = getattr(settings, 'ip', '127.0.0.1')
    client_port = getattr(settings, 'airsim_port', 41451)
    airgym = AirLearningClient(z=-0.9, ip=client_ip, port=client_port)
    print("AirSim 客户端连接成功")

    print("\n[4/4] 同步环境到 UE4/无人机...")

    # 显示当前环境配置
    print("\n当前环境配置:")
    print(f"  地图大小 (ArenaSize): {game_config.get_cur_item('ArenaSize')}")
    print(f"  障碍物数量 (NumberOfObjects): {game_config.get_cur_item('NumberOfObjects')}")
    print(f"  动态障碍物数量 (NumberOfDynamicObjects): {game_config.get_cur_item('NumberOfDynamicObjects')}")
    print(f"  目标点 (End): {game_config.get_cur_item('End')}")
    print(f"  随机种子 (Seed): {game_config.get_cur_item('Seed')}")

    # 重置 Unreal 环境
    print("\n重置 Unreal 环境...")
    airgym.unreal_reset()
    time.sleep(2)

    # 重置无人机
    print("重置无人机...")
    airgym.AirSim_reset()
    time.sleep(1)

    print("\n" + "=" * 60)
    print(f"Level {TEST_LEVEL} ({level_name}) 环境已准备就绪!")
    print("UE4 正在运行，您可以手动截图")
    print("按 Ctrl+C 结束脚本（不会关闭UE4）")
    print("=" * 60)

    # 保持脚本运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n脚本已退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
