#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Take off in the ST AirSim environment and switch to its top-down view.

By default this script starts the ST UE4 project, waits for its AirSim RPC
server, and then performs the normal takeoff.  The UE4-side camera is spawned
automatically at the world origin and frames the fixed 40 m x 40 m environment.
After moving the vehicle to the requested NED height, it invokes the camera
actor method; screenshots remain manual.  Use ``--no-launch`` only when ST is
already running.

Example::

    python test_st_topdown.py
    python test_st_topdown.py --flight-height -1.2 --settle-seconds 2
"""

from __future__ import annotations

import argparse
import time
from typing import Any


DEFAULT_ST_PROJECT = "/mnt/d/Projects/ST/ST.uproject"
DEFAULT_UNREAL_EDITOR = (
    "/mnt/d/SoftWare/Epic Games/Game/UE_4.18/Engine/Binaries/Win64/UE4Editor.exe"
)


def switch_to_topdown_camera(vehicle: Any) -> None:
    """Invoke ASTTopDownCamera::SwitchToTopDownView through AirSim RPC."""

    rpc_client = getattr(vehicle, "client", None)
    if rpc_client is None or not hasattr(rpc_client, "call"):
        raise RuntimeError("This AirSim client does not expose its RPC connection")

    if not bool(rpc_client.call("simSwitchToTopDownCamera")):
        raise RuntimeError(
            "UE4 did not find STTopDownCamera. Rebuild ST and restart the UE4 project."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Take off and switch the ST UE4 viewport to the fixed 40x40 m top-down camera."
    )
    parser.add_argument("--ip", default="127.0.0.1", help="AirSim server IP")
    parser.add_argument("--port", type=int, default=41451, help="AirSim RPC port")
    parser.add_argument(
        "--project",
        default=DEFAULT_ST_PROJECT,
        help="ST UE4 project path",
    )
    parser.add_argument(
        "--unreal-editor",
        default=DEFAULT_UNREAL_EDITOR,
        help="UE4Editor.exe path",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Do not launch UE4; use this only when ST is already running",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        help="Maximum time to wait for AirSim RPC after launching UE4",
    )
    parser.add_argument(
        "--flight-height",
        type=float,
        default=-0.9,
        help="Target height in AirSim NED metres; negative values are above the floor",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Wait after reaching flight height before switching the view",
    )
    args = parser.parse_args()
    if args.flight_height >= 0.0:
        parser.error("--flight-height must be negative in AirSim NED coordinates")
    if args.port <= 0 or args.settle_seconds < 0.0 or args.startup_timeout <= 0.0:
        parser.error("port and startup timeout must be positive; settle time cannot be negative")
    return args


def _launch_st_project(args: argparse.Namespace) -> None:
    """Launch the ST standalone game through the existing WSL/Windows launcher."""

    from eval.launcher import SceneGameHandler

    launcher = SceneGameHandler(
        project_file=args.project,
        unreal_editor=args.unreal_editor,
    )
    print(f"[st-topdown] Launching UE4 project: {args.project}")
    launcher.restart_game()
    print("[st-topdown] UE4 process started; waiting for AirSim RPC...")


def _connect_with_retry(airsim, args: argparse.Namespace):
    """Connect after UE4 startup, whose RPC server may appear several seconds later."""

    vehicle = airsim.MultirotorClient(ip=args.ip, port=args.port)
    deadline = time.monotonic() + args.startup_timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            if vehicle.ping():
                vehicle.confirmConnection()
                return vehicle
        except Exception as exc:
            last_error = exc
        print("[st-topdown] Waiting for AirSim RPC at " f"{args.ip}:{args.port}...")
        time.sleep(2.0)

    raise RuntimeError(
        f"Could not connect to AirSim at {args.ip}:{args.port} within "
        f"{args.startup_timeout:.0f}s. Start ST with --no-launch only if it is already running. "
        f"Last error: {last_error}"
    )


def main() -> int:
    args = parse_args()

    if not args.no_launch:
        _launch_st_project(args)

    import airsim

    vehicle = _connect_with_retry(airsim, args)

    try:
        # Make repeated runs deterministic if the previous run left the drone
        # airborne.
        try:
            vehicle.landAsync().join()
        except Exception as exc:
            print(f"[st-topdown] Initial landing was skipped: {exc}")

        vehicle.enableApiControl(True)
        vehicle.armDisarm(True)
        print("[st-topdown] Taking off...")
        vehicle.takeoffAsync().join()
        print(f"[st-topdown] Moving to {args.flight_height:.2f} m NED")
        vehicle.moveToZAsync(args.flight_height, 1.0).join()
        time.sleep(args.settle_seconds)

        print("[st-topdown] Switching to ASTTopDownCamera...")
        switch_to_topdown_camera(vehicle)
        print("[st-topdown] Done. Take the screenshot manually from the UE4 window.")

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[st-topdown] Stopped; UE4 is left running.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
