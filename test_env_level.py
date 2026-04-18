#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境等级测试脚本
用于启动指定课程学习等级的UE4环境

使用方法：
1. 修改下面的 TEST_LEVEL 变量为需要的等级 (0, 1, 2, 3)
2. 直接运行脚本: python test_env_level.py

课程学习等级说明：
- Level 0: 简单环境 (easy_range_dic) - 小地图(27-30m), 障碍物少(10-15个)
- Level 1: 中等环境 (medium_range_dic) - 中等地图(40-55m), 障碍物中等(16-24个)
- Level 2: 困难环境 (hard_range_dic) - 大地图(55-65m), 障碍物多(25-35个)
- Level 3: 动态障碍物环境 (dynamic_obstacles_dic) - 大地图 + 1-5个动态障碍物
"""

import os
import sys
import time
import shutil
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

# 等级到配置字典的映射
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
    
    # 先生成并写入 JSON，确保 UE4 启动时读取的是本次新配置
    print(f"\n[1/4] 初始化环境配置并生成 JSON (Level {TEST_LEVEL})...")
    game_config = GameConfigHandler(range_dic_name=config_name)

    # 随机化环境并写入 JSON
    print("随机化环境参数并写入 JSON...")
    run_seed = (time.time_ns() ^ (os.getpid() << 16)) & 0xFFFFFFFF
    np_random = np.random.RandomState(seed=run_seed)

    sample_vars = [
        "Seed",
        "ArenaSize",
        "NumberOfObjects",
        "NumberOfDynamicObjects",
        "End",
        "Walls1",
        "MinimumDistance",
    ]
    game_config.sample(*sample_vars, np_random=np_random)

    snapshot_path = save_json_snapshot(level_name, settings.json_file_addr)
    print(f"本次 JSON 快照: {snapshot_path}")
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
