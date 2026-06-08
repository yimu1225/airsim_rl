#!/usr/bin/env python3
"""
脚本：启动UE4，起飞无人机，获取4张1280x1280的深度图像并保存。

图像类型：
  1. clean                  — 原始无噪声深度图
  2. gaussian               — 添加高斯噪声
  3. gaussian_salt_pepper   — 高斯噪声 + 椒盐噪声
  4. gaussian_sp_motion_blur — 高斯噪声 + 椒盐噪声 + 运动模糊

用法：
  python scripts/capture_noisy_images.py
  python scripts/capture_noisy_images.py --output_dir ./noisy_images
  python scripts/capture_noisy_images.py --no_launch  # 如果UE4已经运行

深度图像处理流程（参考项目 airlearningclient.py 的 getScreenDepth）：
  1. 请求 DepthPerspective 浮点深度数据
  2. 截断深度值到 [0, max_depth] 范围
  3. 归一化到 0-255 范围
  4. 依次叠加不同噪声
  5. 保存为 uint8 PNG

依赖：
  airsim, numpy, opencv-python (cv2), psutil
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2

# 将项目根目录加入 path，以便导入项目模块
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ_ROOT)

from game_handling.game_handler_class import GameHandler
from settings_folder import settings


# ==============================================================================
#  深度图噪声函数（作用于 0-255 单通道 float32 深度图像）
#  与项目 airlearningclient.py 的 _add_depth_noise 保持一致的逻辑
# ==============================================================================

def add_gaussian_noise(img, sigma=10.0, clip=30.0):
    """
    对 0-255 float32 深度图添加裁剪高斯噪声。

    Args:
        img: (H, W) float32 numpy array, 范围 0-255
        sigma: 高斯噪声标准差
        clip: 噪声裁剪范围 [-clip, clip]
    Returns:
        noisy: (H, W) float32 numpy array, 范围 0-255
    """
    noisy = np.array(img, dtype=np.float32, copy=True)
    if sigma > 0.0:
        gaussian = np.random.normal(0.0, sigma, size=noisy.shape).astype(np.float32)
        if clip > 0.0:
            gaussian = np.clip(gaussian, -clip, clip)
        noisy += gaussian
        noisy = np.clip(noisy, 0.0, 255.0)
    return noisy


def add_salt_pepper_noise(img, prob=0.02):
    """
    对 0-255 float32 深度图添加椒盐噪声。

    Args:
        img: (H, W) float32 numpy array, 范围 0-255
        prob: 椒盐噪声总概率（盐噪声 prob/2，胡椒噪声 prob/2）
    Returns:
        noisy: (H, W) float32 numpy array, 范围 0-255
    """
    noisy = np.array(img, dtype=np.float32, copy=True)
    if prob > 0.0:
        half_prob = 0.5 * prob
        salt_mask = np.random.random(size=noisy.shape) < half_prob
        pepper_mask = np.random.random(size=noisy.shape) < half_prob
        noisy[salt_mask] = 255.0
        noisy[pepper_mask] = 0.0
    return noisy


def add_motion_blur(img, kernel_size=3):
    """
    对 0-255 float32 深度图添加水平运动模糊。

    Args:
        img: (H, W) float32 numpy array, 范围 0-255
        kernel_size: 模糊核大小（必须为奇数，<=1 表示不模糊）
    Returns:
        blurred: (H, W) float32 numpy array, 范围 0-255
    """
    if kernel_size <= 1:
        return img
    if kernel_size % 2 == 0:
        kernel_size += 1
    # 水平运动模糊核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / float(kernel_size)
    return cv2.filter2D(img, -1, kernel)


# ==============================================================================
#  UE4 启动 / AirSim 连接
# ==============================================================================

def launch_ue4():
    """启动 UE4 游戏进程。"""
    print("[*] 启动 UE4 游戏进程 ...")
    gh = GameHandler()
    gh.start_game_in_editor()
    return gh


def _apply_client_patches(client):
    """
    应用与项目 airlearningclient.py 相同的 simGetImages 兼容性补丁。
    将 ImageRequest 对象序列化为 msgpack dict 格式，以匹配 AirSim 服务端接口。
    """
    rpc_client = client.client

    def compat_simGetImages(requests, vehicle_name='', _external=False):
        requests_msgpack = [
            req.to_msgpack() if hasattr(req, 'to_msgpack') else req
            for req in requests
        ]
        from gym_airsim.envs.airlearningclient import CompatImageResponse
        responses = rpc_client.call('simGetImages', requests_msgpack, vehicle_name)
        return [CompatImageResponse(r) for r in responses]

    client.simGetImages = compat_simGetImages


def connect_airsim(ip="127.0.0.1", port=41451, max_retries=30, retry_interval=2.0):
    """
    连接到 AirSim 服务器并返回 MultirotorClient（已应用兼容性补丁）。
    """
    import airsim

    print(f"[*] 正在连接 AirSim ({ip}:{port}) ...")
    for i in range(max_retries):
        try:
            client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=10)
            _apply_client_patches(client)
            client.confirmConnection()
            print("[+] AirSim 连接成功！")
            return client
        except Exception as e:
            print(f"    [{i+1}/{max_retries}] 等待 AirSim 就绪 ... ({e})")
            time.sleep(retry_interval)

    raise RuntimeError(f"[-] 在 {max_retries * retry_interval}s 内无法连接到 AirSim。")


# ==============================================================================
#  深度图捕获（参考 airlearningclient.py 的 getScreenDepth）
# ==============================================================================

def capture_depth_image(client, vehicle_name="SimpleFlight",
                        max_depth=15.0, max_attempts=3, retry_sleep=0.1):
    """
    捕获 DepthPerspective 浮点深度图像，归一化到 0-255 float32 范围。

    处理流程（与项目 getScreenDepth 一致）：
      1. 请求 DepthPerspective 浮点数据
      2. 处理 inf / nan 异常值
      3. 截断到 [0, max_depth]
      4. 归一化到 0-255

    Args:
        client: airsim.MultirotorClient（已打补丁）
        vehicle_name: 无人机名称
        max_depth: 最大深度值（米），超过此值的深度被截断
        max_attempts: 最大重试次数
        retry_sleep: 重试间隔（秒）

    Returns:
        depth_255: (H, W) float32 numpy array, 范围 0-255
    """
    import airsim

    max_attempts = max(1, int(max_attempts))
    for attempt in range(1, max_attempts + 1):
        responses = None
        try:
            # pixels_as_float=True  → 返回浮点深度数据 (image_data_float)
            # compress=False        → 不压缩
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)],
                vehicle_name=vehicle_name,
            )
        except Exception as e:
            if attempt < max_attempts:
                print(f"    RPC error (attempt {attempt}/{max_attempts}): {e}, retrying...")
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(f"深度图捕获失败（{max_attempts} 次尝试后）: {e}") from e

        if responses is None or len(responses) == 0:
            if attempt < max_attempts:
                print(f"    Empty response (attempt {attempt}/{max_attempts}), retrying...")
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(f"深度图捕获失败（{max_attempts} 次尝试后）: 空响应")

        response = responses[0]
        if response.width == 0 or response.height == 0:
            if attempt < max_attempts:
                print(f"    Invalid dims ({response.width}x{response.height}) "
                      f"(attempt {attempt}/{max_attempts}), retrying...")
                time.sleep(retry_sleep)
                continue
            raise RuntimeError("深度图捕获失败: 图像尺寸无效")

        # --- 处理浮点深度数据 ---
        depth = np.array(response.image_data_float, dtype=float)
        depth = depth.reshape((response.height, response.width))

        # 处理异常值（inf / nan），与项目保持一致
        depth = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
        depth = np.clip(depth, 0.0, max_depth)

        # 归一化到 0-255 范围
        depth_255 = (depth / max_depth) * 255.0
        depth_255 = np.clip(depth_255, 0.0, 255.0).astype(np.float32)

        return depth_255

    raise RuntimeError(f"深度图捕获失败（{max_attempts} 次尝试后）")


# ==============================================================================
#  主流程
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="起飞无人机并获取带不同噪声的 1280x1280 深度图像"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./captured_images",
        help="图像输出目录（默认: ./captured_images）"
    )
    parser.add_argument(
        "--no_launch", action="store_true",
        help="不启动 UE4（假设 UE4 已在运行）"
    )
    parser.add_argument(
        "--max_depth", type=float, default=15.0,
        help="深度截断最大值 / 米（默认: 15.0）"
    )
    parser.add_argument(
        "--gaussian_sigma", type=float, default=10.0,
        help="高斯噪声标准差（默认: 10.0，作用在 0-255 像素值上）"
    )
    parser.add_argument(
        "--gaussian_clip", type=float, default=30.0,
        help="高斯噪声裁剪范围（默认: 30.0）"
    )
    parser.add_argument(
        "--salt_pepper_prob", type=float, default=0.02,
        help="椒盐噪声总概率（默认: 0.02，盐/胡椒各半）"
    )
    parser.add_argument(
        "--motion_blur_kernel", type=int, default=5,
        help="运动模糊核大小（默认: 5，<=1 则跳过）"
    )
    parser.add_argument(
        "--takeoff_height", type=float, default=-1.5,
        help="起飞高度（NED坐标系，负数向上，默认: -1.5m）"
    )
    parser.add_argument(
        "--airsim_ip", type=str, default="127.0.0.1",
        help="AirSim 服务器 IP（默认: 127.0.0.1）"
    )
    parser.add_argument(
        "--airsim_port", type=int, default=41451,
        help="AirSim 服务器端口（默认: 41451）"
    )
    args = parser.parse_args()

    # ---------- 创建输出目录 ----------
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[*] 图像将保存到: {os.path.abspath(args.output_dir)}")

    # ---------- 启动 UE4 ----------
    if not args.no_launch:
        launch_ue4()
    else:
        print("[*] 跳过 UE4 启动（--no_launch 已设置）")

    # ---------- 连接 AirSim ----------
    client = connect_airsim(ip=args.airsim_ip, port=args.airsim_port)

    # ---------- 解锁并起飞 ----------
    print("[*] 解锁无人机 ...")
    client.enableApiControl(True, vehicle_name="SimpleFlight")
    client.armDisarm(True, vehicle_name="SimpleFlight")

    print(f"[*] 起飞中（目标高度: {args.takeoff_height}m NED）...")
    client.takeoffAsync(vehicle_name="SimpleFlight").join()
    client.moveByVelocityZAsync(0, 0, args.takeoff_height, 1,
                                vehicle_name="SimpleFlight").join()
    time.sleep(2.0)
    print("[+] 起飞完成，开始捕获深度图 ...")

    # ---------- 捕获原始深度图 ----------
    print("[*] 捕获 DepthPerspective 深度图 ...")
    clean_depth = capture_depth_image(client, max_depth=args.max_depth)
    h, w = clean_depth.shape
    print(f"[+] 深度图尺寸: {w}x{h} (float32, 范围 0-255)")

    # ---------- 生成噪声图像 ----------
    print("[*] 生成噪声深度图 ...")

    # Image 1: clean (no noise)
    img_clean = clean_depth.copy()

    # Image 2: gaussian noise (applied to clean image)
    img_gaussian = add_gaussian_noise(
        clean_depth,
        sigma=args.gaussian_sigma,
        clip=args.gaussian_clip,
    )

    # Image 3: gaussian + salt & pepper (applied on top of gaussian)
    img_gaussian_sp = add_salt_pepper_noise(
        img_gaussian.copy(),
        prob=args.salt_pepper_prob,
    )

    # Image 4: gaussian + salt & pepper + motion blur
    img_gaussian_sp_mb = add_motion_blur(
        img_gaussian_sp.copy(),
        kernel_size=args.motion_blur_kernel,
    )

    # ---------- 保存图像 ----------
    print("[*] 保存深度图 ...")
    save_paths = {}

    images = {
        "01_clean":                img_clean,
        "02_gaussian":             img_gaussian,
        "03_gaussian_salt_pepper": img_gaussian_sp,
        "04_gaussian_sp_motion_blur": img_gaussian_sp_mb,
    }

    for name, img in images.items():
        path = os.path.join(args.output_dir, f"{name}.png")
        # float32 [0, 255] -> uint8 -> 保存为单通道灰度 PNG
        img_uint8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
        cv2.imwrite(path, img_uint8)
        save_paths[name] = path
        print(f"    ✓ {path}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print("深度图捕获完成！汇总：")
    for label, p in save_paths.items():
        print(f"  {label}: {p}")
    print("=" * 60)

    print(f"\n参数：max_depth={args.max_depth}m")
    print(f"  高斯噪声 sigma={args.gaussian_sigma}, clip={args.gaussian_clip}")
    print(f"  椒盐噪声 prob={args.salt_pepper_prob}")
    print(f"  运动模糊 kernel_size={args.motion_blur_kernel}")


if __name__ == "__main__":
    main()
