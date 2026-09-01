from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from eval.ue4_trajectory import (
    TrajectoryPoint,
    UE4TrajectoryRenderer,
    UE4TrajectoryRpcError,
    save_successful_trajectory,
)
from tests import test_st_vssm_trajectory as trajectory_runner
from tests.test_st_vssm_trajectory import _append_episode_trajectory_point


class _FakeRpcClient:
    def __init__(self, results=None):
        self.calls = []
        self.arguments = []
        self.results = dict(results or {})

    def call(self, method, *args):
        self.calls.append(method)
        self.arguments.append(args)
        result = self.results.get(method, True)
        if isinstance(result, list):
            return result.pop(0)
        return result


def _point(step: int, *, terminal: bool = False) -> TrajectoryPoint:
    return TrajectoryPoint(
        step=step,
        elapsed_seconds=step * 0.1,
        x_ned_m=float(step),
        y_ned_m=float(step + 1),
        z_ned_m=-1.0,
        vx_m_s=1.0,
        vy_m_s=0.0,
        vz_m_s=0.0,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.2,
        action_forward_m_s=1.0,
        action_yaw_rate_rad_s=0.1,
        action_vertical_m_s=0.0,
        reward=2.0,
        cumulative_reward=2.0 * step,
        distance_to_goal_m=10.0 - step,
        path_length_m=float(step),
        is_terminal=terminal,
    )


class UE4TrajectoryTests(unittest.TestCase):
    def test_default_checkpoint_falls_back_to_async_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            model_dir = repo_root / "models" / "CL-VSSM-SAC" / "seed28"
            model_dir.mkdir(parents=True)
            expected = model_dir / "async_final.pth"
            expected.touch()

            with mock.patch.object(trajectory_runner, "REPO_ROOT", repo_root):
                checkpoint = trajectory_runner._default_checkpoint(28)

            self.assertEqual(Path(checkpoint), expected)

    def test_renderer_uses_st_trajectory_rpc_sequence(self):
        raw_rpc = _FakeRpcClient()
        renderer = UE4TrajectoryRenderer(SimpleNamespace(client=raw_rpc))

        renderer.clear()
        renderer.append_current_vehicle_position()
        renderer.finalize_success()
        renderer.switch_to_topdown()

        self.assertEqual(
            raw_rpc.calls,
            [
                "simClearTrajectory",
                "simAppendTrajectoryPoint",
                "simFinalizeTrajectory",
                "simSwitchToTopDownCamera",
            ],
        )

    def test_renderer_sends_configured_goal_in_local_ned_coordinates(self):
        raw_rpc = _FakeRpcClient()
        renderer = UE4TrajectoryRenderer(SimpleNamespace(client=raw_rpc))

        renderer.set_goal_ned([12.5, -7.25, -1.4])

        self.assertEqual(raw_rpc.calls, ["simSetTrajectoryGoal"])
        self.assertEqual(raw_rpc.arguments, [(12.5, -7.25, -1.4)])

    def test_renderer_reports_missing_ue4_actor(self):
        raw_rpc = _FakeRpcClient({"simAppendTrajectoryPoint": False})
        renderer = UE4TrajectoryRenderer(
            SimpleNamespace(client=raw_rpc), retry_attempts=1
        )

        with self.assertRaisesRegex(UE4TrajectoryRpcError, "STTopDownCamera"):
            renderer.append_current_vehicle_position()

    def test_renderer_retries_while_ue4_actor_is_spawning(self):
        raw_rpc = _FakeRpcClient(
            {"simAppendTrajectoryPoint": [False, False, True]}
        )
        renderer = UE4TrajectoryRenderer(
            SimpleNamespace(client=raw_rpc),
            retry_attempts=3,
            retry_interval_seconds=0.0,
        )

        renderer.append_current_vehicle_position()

        self.assertEqual(
            raw_rpc.calls,
            ["simAppendTrajectoryPoint"] * 3,
        )

    def test_episode_renderer_skips_append_after_environment_restarts(self):
        raw_rpc = _FakeRpcClient()
        old_vehicle = SimpleNamespace(client=raw_rpc)
        new_vehicle = SimpleNamespace(client=_FakeRpcClient())
        renderer = UE4TrajectoryRenderer(old_vehicle)

        appended = _append_episode_trajectory_point(
            renderer,
            episode_vehicle=old_vehicle,
            current_vehicle=new_vehicle,
            info={"ue4_restarted": True},
        )

        self.assertFalse(appended)
        self.assertEqual(raw_rpc.calls, [])

    def test_episode_renderer_turns_missing_camera_into_failed_attempt(self):
        raw_rpc = _FakeRpcClient({"simAppendTrajectoryPoint": False})
        vehicle = SimpleNamespace(client=raw_rpc)
        renderer = UE4TrajectoryRenderer(vehicle, retry_attempts=1)

        appended = _append_episode_trajectory_point(
            renderer,
            episode_vehicle=vehicle,
            current_vehicle=vehicle,
            info={},
        )

        self.assertFalse(appended)

    def test_save_successful_trajectory_writes_csv_npz_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_successful_trajectory(
                Path(temp_dir),
                [_point(0), _point(1, terminal=True)],
                {"algorithm": "CL-VSSM-SAC", "termination": "success"},
            )

            with paths["csv"].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["step"]) for row in rows], [0, 1])
            self.assertEqual(rows[-1]["is_terminal"], "True")

            with np.load(paths["npz"]) as arrays:
                np.testing.assert_allclose(arrays["x_ned_m"], [0.0, 1.0])
                self.assertEqual(arrays["is_terminal"].tolist(), [False, True])

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["algorithm"], "CL-VSSM-SAC")
            self.assertEqual(metadata["trajectory_point_count"], 2)
            self.assertEqual(metadata["trajectory_csv"], paths["csv"].name)

    def test_empty_successful_trajectory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "at least one point"):
                save_successful_trajectory(
                    Path(temp_dir), [], {"termination": "success"}
                )


if __name__ == "__main__":
    unittest.main()
