#!/usr/bin/env python3
"""
支持指定端口和settings文件的启动脚本

使用方法:
python run_with_port.py --airsim_port 41451
python run_with_port.py --airsim_ip 127.0.0.1 --airsim_port 42451 
python run_with_port.py --settings_file ./UAV_Navigation(RLlib)/settings(1).json
"""

import argparse
import os
import sys
import shutil
from main import main
from config import get_config

def copy_settings_file(settings_file):
    """
    将指定的settings文件复制到AirSim默认位置
    """
    if not os.path.exists(settings_file):
        print(f"错误: settings文件不存在: {settings_file}")
        return False
    
    # AirSim默认settings文件位置
    import platform
    if platform.system() == "Windows":
        airsim_settings_dir = os.path.join(os.path.expanduser("~"), "Documents", "AirSim")
    else:
        airsim_settings_dir = os.path.join(os.path.expanduser("~"), "Documents", "AirSim")
    
    os.makedirs(airsim_settings_dir, exist_ok=True)
    airsim_settings_file = os.path.join(airsim_settings_dir, "settings.json")
    
    # 备份原有settings文件
    if os.path.exists(airsim_settings_file):
        backup_file = airsim_settings_file + ".backup"
        shutil.copy2(airsim_settings_file, backup_file)
        print(f"原有settings文件已备份到: {backup_file}")
    
    # 复制新settings文件
    shutil.copy2(settings_file, airsim_settings_file)
    print(f"Settings文件已复制到: {airsim_settings_file}")
    return True

def parse_custom_args():
    """
    解析自定义参数，然后调用主配置
    """
    parser = argparse.ArgumentParser(description='AirSim RL with custom port and settings', add_help=False)
    parser.add_argument('--airsim_ip', type=str, help='AirSim server IP')
    parser.add_argument('--airsim_port', type=int, help='AirSim server port')
    parser.add_argument('--settings_file', type=str, help='Path to AirSim settings.json file')
    
    # 解析已知参数，其余传给主配置
    custom_args, remaining_args = parser.parse_known_args()
    
    # 如果指定了settings文件，先复制它
    if custom_args.settings_file:
        if not copy_settings_file(custom_args.settings_file):
            sys.exit(1)
    
    # 重新构建参数列表
    if custom_args.airsim_ip:
        remaining_args.extend(['--airsim_ip', custom_args.airsim_ip])
    if custom_args.airsim_port:
        remaining_args.extend(['--airsim_port', str(custom_args.airsim_port)])
    if custom_args.settings_file:
        remaining_args.extend(['--settings_file', custom_args.settings_file])
    
    # 修改sys.argv传给主程序
    sys.argv = [sys.argv[0]] + remaining_args
    
    return custom_args

if __name__ == "__main__":
    custom_args = parse_custom_args()
    
    print("=" * 60)
    print("启动 AirSim RL 训练")
    if custom_args.airsim_ip:
        print(f"AirSim IP: {custom_args.airsim_ip}")
    if custom_args.airsim_port:
        print(f"AirSim Port: {custom_args.airsim_port}")
    if custom_args.settings_file:
        print(f"Settings File: {custom_args.settings_file}")
    print("=" * 60)
    
    # 调用主程序
    main()