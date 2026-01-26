
import os
import subprocess
import time
import sys

# 添加父目录到 sys.path 以加载 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from settings_folder import machine_dependent_settings

# ==============================================================================
# 用户配置区域 (自动从 machine_dependent_settings.py 读取)
# ==============================================================================

# 1. UE4Editor.exe 的绝对路径
UE4_EDITOR_PATH = machine_dependent_settings.unreal_exe_path

# 2. .uproject 项目文件的绝对路径
PROJECT_PATH = machine_dependent_settings.game_file

# 分辨率设置 (降低分辨率以节省资源)
RES_X = 640
RES_Y = 480

# ==============================================================================

def launch():
    # 获取根目录 (airsim_rl (1.8.1))
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 自动探测 settings 文件目录
    settings_root = root_dir
    
    args = get_config()
    num_workers = args.n_training_threads 
    
    print(f"============================================================")
    print(f"AirSim 并行实例启动脚本")
    print(f"============================================================")
    print(f"计划启动实例数: {num_workers}")
    print(f"UE4Editor路径: {UE4_EDITOR_PATH}")
    print(f"项目文件路径: {PROJECT_PATH}")
    print(f"配置文件根目录: {settings_root}")
    
    # 检查路径是否存在
    if not os.path.exists(UE4_EDITOR_PATH):
        print(f"\n[错误] 找不到 UE4Editor.exe！")
        print(f"请打开此脚本 ({__file__}) 并修改 'UE4_EDITOR_PATH' 变量为正确的路径。")
        return

    if not os.path.exists(PROJECT_PATH):
        print(f"\n[错误] 找不到 .uproject 项目文件！")
        print(f"当前配置路径: {PROJECT_PATH}")
        print(f"请打开此脚本 ({__file__}) 并修改 'PROJECT_PATH' 变量为正确的路径。")
        return

    procs = []
    
    for i in range(1, num_workers + 1):
        # 构造 settings 文件名
        settings_file_name = f"settings({i}).json"
        settings_file_path = os.path.join(settings_root, settings_file_name)
        
        if not os.path.exists(settings_file_path):
            print(f"[错误] 找不到配置文件: {settings_file_path}")
            print(f"请确保 settings(1).json, settings(2).json 等文件存在于项目根目录。")
            continue
            
        # 构造启动命令
        # 注意：-settings 参数必须是绝对路径
        cmd = [
            UE4_EDITOR_PATH,
            PROJECT_PATH,
            "-game",          # 游戏模式运行
            "-windowed",      # 窗口模式
            f"-settings={os.path.abspath(settings_file_path)}", # 关键：指定特定配置文件
            f"-ResX={RES_X}",
            f"-ResY={RES_Y}",
            "-noforcefeedback",
            "-soundnull"
        ]
        
        print(f"\n正在启动实例 {i} ...")
        print(f"  端口设置文件: {settings_file_name}")
        
        try:
            # 启动子进程，不等待其结束
            proc = subprocess.Popen(cmd)
            procs.append(proc)
            print(f"  PID: {proc.pid} (启动成功)")
        except Exception as e:
            print(f"  启动失败: {e}")

        # 错峰启动，减少资源冲击
        if i < num_workers:
            print("等待 10 秒后启动下一个实例...")
            time.sleep(10) 

    if not procs:
        print("\n[失败] 没有启动任何实例。")
        return

    print("\n" + "="*60)
    print("所有实例启动命令已发送。")
    print("请等待 UE4 窗口出现，并完成 AirSim 插件加载 (可能需要几分钟)。")
    print("一旦所有窗口都显示准备就绪，您就可以运行 train_parallel.py 了。")
    print("="*60)
    print("【保持此终端运行】按 Enter 键可以一键关闭所有实例...")
    print("="*60)
    
    try:
        input()
    except KeyboardInterrupt:
        pass
    finally:
        print("正在关闭所有 UE4 实例...")
        for p in procs:
            if p.poll() is None: # 如果还在运行
                p.terminate()
        print("已关闭。")

if __name__ == "__main__":
    launch()
