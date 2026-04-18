#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone env-level test script for AirLearning/AirSim.

Design goals:
1) All definitions live in this script (no project-module imports).
2) Do not modify any existing project code.
3) Support WSL2 + Windows UE editor path handling.

Examples:
  python3 test_env_level_all_in_one.py --level 2 --dry-run
  python3 test_env_level_all_in_one.py --level 1
  python3 test_env_level_all_in_one.py --level 3 --seed 1234 --stay-alive
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# User-editable definitions (all kept inside this script)
# ============================================================

# Candidate paths: script auto-picks the first existing one.
JSON_PATH_CANDIDATES: List[str] = [
    "/mnt/d/Projects/airlearning-ue4-cpp-427/Content/JsonFiles/EnvGenConfig.json",
    "/mnt/d/Projects/airlearning-ue4/Content/JsonFiles/EnvGenConfig.json",
]

UPROJECT_CANDIDATES: List[str] = [
    "/mnt/d/Projects/airlearning-ue4-cpp-427/AirLearning.uproject",
    "/mnt/d/Projects/airlearning-ue4/AirLearning.uproject",
]

UE_EXE_CANDIDATES: List[str] = [
    "/mnt/d/SoftWare/Epic Games/Epic Games/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe",
    "/mnt/d/SoftWare/Epic Games/Game/UE_4.18/Engine/Binaries/Win64/UE4Editor.exe",
]

# Runtime defaults
DEFAULT_IP = "127.0.0.1"
DEFAULT_PORT = 41451
DEFAULT_TAKEOFF_Z = -0.9
DEFAULT_GOAL_HALO = 4.0

# UE window defaults
DEFAULT_RES_X = 940
DEFAULT_RES_Y = 700
DEFAULT_WIN_X = 400
DEFAULT_WIN_Y = 200

# Common wall presets (copied from project settings style)
WALL_PRESETS: List[List[int]] = [
    [200, 13, 99],
    [255, 255, 10],
    [0, 10, 10],
    [10, 100, 100],
    [126, 11, 90],
]

# Value spec formats used below:
# - [v1, v2, ...]               -> choose one element
# - ("int_range", low, high)    -> random.randrange(low, high)  (high exclusive)
# - ("fixed", value)            -> fixed value
LEVEL_DEFS: Dict[int, Dict[str, Any]] = {
    0: {
        "Name": ["Name"],
        "EnvType": ["Indoor"],
        "PlayerStart": [[0, 0, 0]],
        "ArenaSize": [[80, 80, 10]],
        "Walls1": WALL_PRESETS,
        "Seed": ("int_range", 0, 10000),
        "MinimumDistance": ("int_range", 3, 6),
        "VelocityRange": [[0, 2]],
        "NumberOfObjects": ("int_range", 20, 30),
        "NumberOfDynamicObjects": ("fixed", 0),
        "End": "Mutable",
    },
    1: {
        "Name": ["Name"],
        "EnvType": ["Indoor"],
        "PlayerStart": [[0, 0, 0]],
        "ArenaSize": [[80, 80, 10]],
        "Walls1": WALL_PRESETS,
        "Seed": ("int_range", 0, 10000),
        "MinimumDistance": ("int_range", 3, 6),
        "VelocityRange": [[0, 4]],
        "NumberOfObjects": ("int_range", 40, 60),
        "NumberOfDynamicObjects": ("fixed", 0),
        "End": "Mutable",
    },
    2: {
        "Name": ["Name"],
        "EnvType": ["Indoor"],
        "PlayerStart": [[0, 0, 0]],
        "ArenaSize": [[80, 80, 10]],
        "Walls1": WALL_PRESETS,
        "Seed": ("int_range", 0, 10000),
        "MinimumDistance": ("int_range", 2, 3),
        "VelocityRange": [[0, 5]],
        "NumberOfObjects": ("int_range", 60, 80),
        "NumberOfDynamicObjects": ("fixed", 0),
        "End": "Mutable",
    },
    3: {
        "Name": ["Name"],
        "EnvType": ["Indoor"],
        "PlayerStart": [[0, 0, 0]],
        "ArenaSize": [[80, 80, 10]],
        "Walls1": WALL_PRESETS,
        "Seed": ("int_range", 0, 10000),
        "MinimumDistance": ("int_range", 2, 4),
        "VelocityRange": [[0, 5]],
        "NumberOfObjects": ("int_range", 60, 80),
        "NumberOfDynamicObjects": ("int_range", 1, 6),
        "End": "Mutable",
    },
}

LEVEL_NAMES = {
    0: "easy",
    1: "medium",
    2: "hard",
    3: "dynamic",
}


def first_existing(candidates: Sequence[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def to_windows_path(path: str) -> str:
    if not path:
        return path
    # already windows-ish
    if len(path) >= 2 and path[1] == ":":
        return path

    try:
        out = subprocess.check_output(["wslpath", "-w", path], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip() or path
    except Exception:
        return path


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def recursive_set_key(obj: Any, key: str, value: Any) -> int:
    changed = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                obj[k] = value
                changed += 1
            else:
                changed += recursive_set_key(v, key, value)
    elif isinstance(obj, list):
        for item in obj:
            changed += recursive_set_key(item, key, value)
    return changed


def recursive_get_first(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = recursive_get_first(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = recursive_get_first(item, key)
            if found is not None:
                return found
    return None


def sample_spec(spec: Any, rng: random.Random) -> Any:
    if isinstance(spec, tuple):
        if len(spec) >= 2 and spec[0] == "fixed":
            return spec[1]
        if len(spec) >= 3 and spec[0] == "int_range":
            low, high = int(spec[1]), int(spec[2])
            if high <= low:
                return low
            return rng.randrange(low, high)
        raise ValueError(f"Unsupported tuple spec: {spec}")

    if isinstance(spec, list):
        if not spec:
            raise ValueError("List spec cannot be empty")
        return rng.choice(spec)

    # fallback literal
    return spec


def generate_mutable_end(arena_size: Sequence[float], rng: random.Random, goal_halo: float) -> List[float]:
    if len(arena_size) < 2:
        raise ValueError(f"ArenaSize invalid: {arena_size}")

    x_q = max(2.0, (float(arena_size[0]) - goal_halo) / 2.0)
    y_q = max(2.0, (float(arena_size[1]) - goal_halo) / 2.0)

    x = rng.uniform(2.0, x_q) * rng.choice([-1.0, 1.0])
    y = rng.uniform(2.0, y_q) * rng.choice([-1.0, 1.0])
    return [x, y, 0.0]


def apply_level_to_config(cfg: Dict[str, Any], level: int, rng: random.Random, goal_halo: float) -> Tuple[Dict[str, Any], List[str]]:
    if level not in LEVEL_DEFS:
        raise ValueError(f"Unsupported level {level}; choose from {sorted(LEVEL_DEFS.keys())}")

    spec = LEVEL_DEFS[level]
    sampled: Dict[str, Any] = {}

    # sample all keys except End first
    for key, value_spec in spec.items():
        if key == "End":
            continue
        sampled[key] = sample_spec(value_spec, rng)

    # mutable End
    if spec.get("End") == "Mutable":
        arena = sampled.get("ArenaSize")
        if arena is None:
            arena = recursive_get_first(cfg, "ArenaSize")
        if arena is None:
            raise RuntimeError("Cannot determine ArenaSize for mutable End generation")
        sampled["End"] = generate_mutable_end(arena, rng, goal_halo)
    else:
        sampled["End"] = sample_spec(spec.get("End"), rng)

    missing_keys: List[str] = []
    for k, v in sampled.items():
        num = recursive_set_key(cfg, k, v)
        if num == 0:
            missing_keys.append(k)

    return sampled, missing_keys


def kill_ue_processes(best_effort: bool = True) -> None:
    process_names = ["UE4Editor.exe", "UnrealEditor.exe", "CrashReportClient.exe"]

    # WSL/Linux calling Windows taskkill
    if os.name != "nt":
        for p in process_names:
            cmd = ["taskkill.exe", "/f", "/im", p]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        # Optional Linux-native names
        for p in ["UE4Editor", "UnrealEditor", "CrashReportClient"]:
            subprocess.run(["killall", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return

    # Native Windows
    for p in process_names:
        cmd = ["taskkill", "/f", "/im", p]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def launch_ue(ue_exe: str, uproject: str, res_x: int, res_y: int, win_x: int, win_y: int) -> Tuple[subprocess.Popen, List[str]]:
    if not os.path.exists(ue_exe):
        raise FileNotFoundError(f"UE executable not found: {ue_exe}")
    if not os.path.exists(uproject):
        raise FileNotFoundError(f"uproject not found: {uproject}")

    project_arg = uproject
    if os.name != "nt" and ue_exe.lower().endswith(".exe"):
        project_arg = to_windows_path(uproject)

    cmd = [
        ue_exe,
        project_arg,
        "-game",
        f"-ResX={int(res_x)}",
        f"-ResY={int(res_y)}",
        f"-WinX={int(win_x)}",
        f"-WinY={int(win_y)}",
        "-Windowed",
        "-NOPAUSE",
    ]

    proc = subprocess.Popen(cmd)
    return proc, cmd


def connect_and_reset_airsim(ip: str, port: int, takeoff_z: float, do_unreal_reset: bool = True) -> None:
    try:
        import airsim  # type: ignore
    except Exception as e:
        raise RuntimeError(f"airsim import failed: {e}")

    client = airsim.MultirotorClient(ip=ip, port=port)
    client.confirmConnection()

    if do_unreal_reset and hasattr(client, "resetUnreal"):
        try:
            client.resetUnreal()
            time.sleep(2.0)
        except Exception:
            pass

    # Best-effort standard reset sequence
    try:
        client.reset()
    except Exception:
        pass
    try:
        client.enableApiControl(True)
    except Exception:
        pass
    try:
        client.armDisarm(True)
    except Exception:
        pass
    try:
        client.moveToZAsync(float(takeoff_z), 1.0).join()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone env-level test script")
    parser.add_argument("--level", type=int, default=2, choices=sorted(LEVEL_DEFS.keys()), help="Curriculum level to test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: time-based)")

    parser.add_argument("--json-path", type=str, default=first_existing(JSON_PATH_CANDIDATES) or JSON_PATH_CANDIDATES[0], help="EnvGenConfig.json path")
    parser.add_argument("--uproject", type=str, default=first_existing(UPROJECT_CANDIDATES) or UPROJECT_CANDIDATES[0], help="AirLearning.uproject path")
    parser.add_argument("--ue-exe", type=str, default=first_existing(UE_EXE_CANDIDATES) or UE_EXE_CANDIDATES[0], help="UE editor executable path")

    parser.add_argument("--ip", type=str, default=DEFAULT_IP, help="AirSim RPC IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="AirSim RPC port")
    parser.add_argument("--takeoff-z", type=float, default=DEFAULT_TAKEOFF_Z, help="Takeoff Z for reset")

    parser.add_argument("--goal-halo", type=float, default=DEFAULT_GOAL_HALO, help="Goal halo used for End generation")

    parser.add_argument("--res-x", type=int, default=DEFAULT_RES_X)
    parser.add_argument("--res-y", type=int, default=DEFAULT_RES_Y)
    parser.add_argument("--win-x", type=int, default=DEFAULT_WIN_X)
    parser.add_argument("--win-y", type=int, default=DEFAULT_WIN_Y)

    parser.add_argument("--sleep-after-launch", type=float, default=8.0, help="Seconds to wait after launching UE before AirSim reset")

    parser.add_argument("--no-launch", action="store_true", help="Do not launch UE")
    parser.add_argument("--no-airsim", action="store_true", help="Do not connect/reset AirSim")
    parser.add_argument("--no-kill-first", action="store_true", help="Do not kill existing UE processes before launch")
    parser.add_argument("--stay-alive", action="store_true", help="Keep script alive until Ctrl+C")
    parser.add_argument("--dry-run", action="store_true", help="No file write, no UE launch, no AirSim connect")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    rng = random.Random(seed)

    print("=" * 72)
    print("Standalone test_env_level script")
    print("=" * 72)
    print(f"Level: {args.level} ({LEVEL_NAMES.get(args.level, 'unknown')})")
    print(f"Seed: {seed}")
    print(f"JSON: {args.json_path}")
    print(f"UE exe: {args.ue_exe}")
    print(f"uproject: {args.uproject}")
    print(f"AirSim: {args.ip}:{args.port}")
    print(f"dry-run: {args.dry_run}")
    print("-" * 72)

    if not os.path.exists(args.json_path):
        print(f"[ERROR] JSON file not found: {args.json_path}")
        return 1

    with open(args.json_path, "r", encoding="utf-8") as f:
        content = f.read().replace("nan", "NaN")
        cfg = json.loads(content)

    sampled, missing = apply_level_to_config(cfg, args.level, rng, args.goal_halo)

    print("Sampled values:")
    for k in sorted(sampled.keys()):
        print(f"  {k}: {sampled[k]}")

    if missing:
        print("[WARN] keys not found in target JSON (skipped while setting):")
        for k in missing:
            print(f"  - {k}")

    if args.dry_run:
        print("\n[DRY-RUN] Skip writing JSON, launching UE, and AirSim reset.")
        return 0

    # Backup original json before overwrite
    backup_path = args.json_path + f".bak_{int(time.time())}"
    with open(backup_path, "w", encoding="utf-8") as fb:
        json.dump(json.loads(content), fb)
    print(f"Backup saved: {backup_path}")

    with open(args.json_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    print(f"Updated JSON written: {args.json_path}")

    ue_proc: Optional[subprocess.Popen] = None

    if not args.no_launch:
        if not args.no_kill_first:
            print("Killing existing UE processes (best effort)...")
            kill_ue_processes()
            time.sleep(2.0)

        ue_proc, cmd = launch_ue(
            ue_exe=args.ue_exe,
            uproject=args.uproject,
            res_x=args.res_x,
            res_y=args.res_y,
            win_x=args.win_x,
            win_y=args.win_y,
        )
        print("UE launch command:")
        print("  " + quote_cmd(cmd))
        print(f"UE process started, pid={ue_proc.pid}")
        if args.sleep_after_launch > 0:
            print(f"Waiting {args.sleep_after_launch:.1f}s for UE startup...")
            time.sleep(args.sleep_after_launch)

    if not args.no_airsim:
        print("Connecting to AirSim and sending reset sequence...")
        connect_and_reset_airsim(
            ip=args.ip,
            port=args.port,
            takeoff_z=args.takeoff_z,
            do_unreal_reset=True,
        )
        print("AirSim reset sequence completed.")

    print("\nDone.")

    if args.stay_alive:
        print("Stay-alive mode enabled. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        raise SystemExit(130)
