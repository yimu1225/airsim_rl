#!/usr/bin/env python3
"""UE4 trajectory rendering RPCs and successful-rollout data persistence."""

from __future__ import annotations

import csv
import datetime as dt
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class UE4TrajectoryRpcError(RuntimeError):
    """Raised when the ST AirSim plugin cannot render the trajectory."""


class UE4TrajectoryRenderer:
    """Small client for the ST project's in-engine trajectory renderer."""

    def __init__(
        self,
        airsim_vehicle: Any,
        *,
        retry_attempts: int = 20,
        retry_interval_seconds: float = 0.1,
    ) -> None:
        rpc_client = getattr(airsim_vehicle, "client", None)
        if rpc_client is None or not hasattr(rpc_client, "call"):
            raise UE4TrajectoryRpcError(
                "The AirSim client does not expose its raw RPC connection."
            )
        if retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        if retry_interval_seconds < 0:
            raise ValueError("retry_interval_seconds must not be negative")
        self._rpc_client = rpc_client
        self._retry_attempts = int(retry_attempts)
        self._retry_interval_seconds = float(retry_interval_seconds)

    def _call(self, method: str, *args: Any) -> None:
        last_exception: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                result = self._rpc_client.call(method, *args)
                last_exception = None
            except Exception as exc:
                result = False
                last_exception = exc

            if bool(result):
                return
            if attempt < self._retry_attempts and self._retry_interval_seconds:
                time.sleep(self._retry_interval_seconds)

        if last_exception is not None:
            raise UE4TrajectoryRpcError(
                f"UE4 RPC '{method}' failed after {self._retry_attempts} attempts. "
                "Rebuild and restart the ST project after applying "
                "scripts/patch_st_trajectory_renderer.py."
            ) from last_exception
        raise UE4TrajectoryRpcError(
            f"UE4 RPC '{method}' could not find STTopDownCamera after "
            f"{self._retry_attempts} attempts."
        )

    def clear(self) -> None:
        """Remove the current in-engine path before another attempt."""

        self._call("simClearTrajectory")

    def append_current_vehicle_position(self) -> None:
        """Append the vehicle Actor's current UE4 world position."""

        self._call("simAppendTrajectoryPoint")

    def set_goal_ned(self, goal_ned_m: Sequence[float]) -> None:
        """Draw the configured navigation goal from AirSim local NED metres."""

        goal = np.asarray(goal_ned_m, dtype=np.float64).reshape(-1)
        if goal.size != 3 or not np.all(np.isfinite(goal)):
            raise ValueError("goal_ned_m must contain three finite coordinates")
        self._call("simSetTrajectoryGoal", *(float(value) for value in goal))

    def finalize_success(self) -> None:
        """Keep the line and add start/end markers for a successful episode."""

        self._call("simFinalizeTrajectory")

    def switch_to_topdown(self) -> None:
        """Make the ST overview camera the active UE4 viewport camera."""

        self._call("simSwitchToTopDownCamera")


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    elapsed_seconds: float
    x_ned_m: float
    y_ned_m: float
    z_ned_m: float
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    action_forward_m_s: float
    action_yaw_rate_rad_s: float
    action_vertical_m_s: float
    reward: float
    cumulative_reward: float
    distance_to_goal_m: float
    path_length_m: float
    is_terminal: bool


def _numeric_matrix(points: Sequence[TrajectoryPoint]) -> dict[str, np.ndarray]:
    rows = [asdict(point) for point in points]
    if not rows:
        raise ValueError("A successful trajectory must contain at least one point")
    return {
        key: np.asarray([row[key] for row in rows])
        for key in rows[0]
    }


def default_trajectory_output_dir(
    repo_root: Path,
    *,
    model_seed: int,
    episode_seed: int,
) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    return (
        repo_root
        / "results"
        / "trajectories"
        / "CL-VSSM-SAC"
        / f"seed{int(model_seed)}"
        / f"episode_seed{int(episode_seed)}"
        / stamp
    )


def save_successful_trajectory(
    output_dir: Path,
    points: Iterable[TrajectoryPoint],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Save a successful path as human-readable CSV, NPZ, and metadata JSON."""

    point_list = list(points)
    arrays = _numeric_matrix(point_list)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "successful_episode_trajectory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(arrays))
        writer.writeheader()
        writer.writerows(asdict(point) for point in point_list)

    npz_path = output_dir / "successful_episode_trajectory.npz"
    np.savez_compressed(npz_path, **arrays)

    metadata_path = output_dir / "metadata.json"
    serializable_metadata = dict(metadata)
    serializable_metadata.update(
        {
            "trajectory_point_count": len(point_list),
            "trajectory_csv": csv_path.name,
            "trajectory_npz": npz_path.name,
            "saved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return {"csv": csv_path, "npz": npz_path, "metadata": metadata_path}


__all__ = [
    "TrajectoryPoint",
    "UE4TrajectoryRenderer",
    "UE4TrajectoryRpcError",
    "default_trajectory_output_dir",
    "save_successful_trajectory",
]
