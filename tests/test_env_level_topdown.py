#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ``test_env_level.py`` and switch the viewport to the UE4 top-down camera.

This is a small wrapper around the existing level-test workflow. It does not
change ``test_env_level.py``: the selected level is assigned in the child
process before its existing ``main()`` function is run. Once that script has
finished preparing the environment, this wrapper performs a normal takeoff,
climbs to the requested flight height, and invokes the UE4 actor method over
the AirSim RPC connection.

Examples::

    python test_env_level_topdown.py --level 0
    python test_env_level_topdown.py --level 2 --flight-height -1.2

After the final message appears, take the screenshot from the UE4 window.
The camera height is computed inside UE4 from the current ArenaSize and the
viewport aspect ratio, with a 25% framing margin and a wall-clearance check.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
READY_MARKER = "环境已准备就绪!"


def _build_test_env_command(level: int) -> list[str]:
    """Run the existing level script without modifying its source file."""

    child_code = (
        "import test_env_level\n"
        f"test_env_level.TEST_LEVEL = {int(level)}\n"
        "test_env_level.main()\n"
    )
    # The parent waits for the readiness line, so keep the child's stdout
    # unbuffered while forwarding the existing test script's output.
    return [sys.executable, "-u", "-c", child_code]


def _start_test_env(level: int, timeout: float) -> subprocess.Popen:
    process = subprocess.Popen(
        _build_test_env_command(level),
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    deadline = time.monotonic() + timeout
    assert process.stdout is not None
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line:
            print(f"[test_env_level] {line}", end="")
            if READY_MARKER in line:
                return process
            continue

        if process.poll() is not None:
            raise RuntimeError(
                f"test_env_level.py exited before the UE4 environment was ready "
                f"(exit code {process.returncode})"
            )
        time.sleep(0.1)

    raise TimeoutError(
        f"Timed out after {timeout:.0f}s waiting for test_env_level.py to prepare UE4"
    )


def switch_to_topdown_camera(airsim_client) -> None:
    """Invoke ``AAirLearningTopDownCamera::SwitchToTopDownView`` through AirSim."""

    rpc_client = getattr(airsim_client, "client", None)
    if rpc_client is None or not hasattr(rpc_client, "call"):
        raise RuntimeError(
            "The AirSim client does not expose its RPC connection; "
            "use the project's AirSim Python package."
        )

    switched = rpc_client.call("simSwitchToTopDownCamera")
    if not bool(switched):
        raise RuntimeError(
            "UE4 did not find AirLearningTopDownCamera. "
            "Rebuild the UE4 project and restart it before running this wrapper."
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run test_env_level.py for one curriculum level, take off, and "
            "switch the UE4 viewport to the dynamically framed top-down camera."
        )
    )
    parser.add_argument("--level", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument(
        "--flight-height",
        type=float,
        default=-0.9,
        help="Target height in AirSim NED metres; negative values are above the floor.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        help="Maximum time to wait for test_env_level.py to report readiness.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Wait after reaching flight height before switching the viewport.",
    )
    args = parser.parse_args(argv)
    if args.flight_height >= 0.0:
        parser.error("--flight-height must be negative in AirSim NED coordinates")
    if args.startup_timeout <= 0.0 or args.settle_seconds < 0.0:
        parser.error("timeouts must be positive and settle time cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    child = _start_test_env(args.level, args.startup_timeout)

    try:
        from settings_folder import settings
        from gym_airsim.envs.airlearningclient import AirLearningClient

        client_ip = getattr(settings, "ip", "127.0.0.1")
        client_port = getattr(
            settings,
            "airsim_port",
            getattr(settings, "port", 41451),
        )
        airlearning_client = AirLearningClient(
            z=args.flight_height,
            ip=client_ip,
            port=client_port,
        )
        vehicle = airlearning_client.client

        print("[topdown] Landing once so the wrapper starts from the normal takeoff state")
        try:
            vehicle.landAsync().join()
        except Exception as exc:
            print(f"[topdown] Initial landing was skipped: {exc}")

        vehicle.enableApiControl(True)
        vehicle.armDisarm(True)
        print("[topdown] Taking off...")
        vehicle.takeoffAsync().join()
        print(f"[topdown] Moving to flight height {args.flight_height:.2f}m (NED)")
        vehicle.moveToZAsync(float(args.flight_height), 1.0).join()
        time.sleep(args.settle_seconds)

        print("[topdown] Calling AAirLearningTopDownCamera::SwitchToTopDownView()")
        switch_to_topdown_camera(vehicle)
        print(
            "[topdown] Done. The UE4 viewport is now the dynamically framed "
            "top-down view; take your screenshot manually."
        )

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[topdown] Wrapper stopped; UE4 is left running.")
        return 0
    finally:
        # test_env_level.py intentionally keeps UE4 alive. Do not terminate the
        # child here; only close the parent's copy of its output pipe.
        if child.stdout is not None:
            child.stdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
