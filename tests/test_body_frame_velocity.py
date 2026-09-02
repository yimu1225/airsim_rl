from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import airsim
import numpy as np

from gym_airsim.envs.AirGym import AirSimEnv
from gym_airsim.envs.airlearningclient import AirLearningClient


class _CompletedCommand:
    def join(self):
        return None


class _FakeMultirotorClient:
    def __init__(self):
        self.body_velocity_commands = []
        self.pause_calls = []

    def simPause(self, paused):
        self.pause_calls.append(bool(paused))

    def moveByVelocityBodyFrameAsync(
        self,
        vx,
        vy,
        vz,
        duration,
        drivetrain,
        yaw_mode,
    ):
        self.body_velocity_commands.append(
            (vx, vy, vz, duration, drivetrain, yaw_mode)
        )
        return _CompletedCommand()

    def simGetCollisionInfo(self):
        return SimpleNamespace(has_collided=False)


class BodyFrameVelocityTests(unittest.TestCase):
    def test_precise_pause_sends_xyz_velocity_in_body_frame(self):
        rpc_client = _FakeMultirotorClient()
        client = AirLearningClient.__new__(AirLearningClient)
        client.client = rpc_client

        collided = client.take_continuous_action_3d_precise_pause(
            np.array([1.25, -0.75, 0.2], dtype=np.float32),
            duration=0.1,
        )

        self.assertFalse(collided)
        self.assertEqual(len(rpc_client.body_velocity_commands), 2)
        for vx, vy, vz, duration, drivetrain, yaw_mode in rpc_client.body_velocity_commands:
            self.assertAlmostEqual(vx, 1.25)
            self.assertAlmostEqual(vy, -0.75)
            self.assertAlmostEqual(vz, 0.2)
            self.assertAlmostEqual(duration, 0.05)
            self.assertEqual(drivetrain, airsim.DrivetrainType.MaxDegreeOfFreedom)
            self.assertTrue(yaw_mode.is_rate)
            self.assertAlmostEqual(yaw_mode.yaw_or_rate, 0.0)
        self.assertEqual(rpc_client.pause_calls, [False, True, False, True])

    def test_world_velocity_is_rotated_into_body_frame(self):
        orientation = airsim.to_quaternion(0.0, 0.0, math.pi / 2.0)
        velocity = SimpleNamespace(x_val=0.0, y_val=2.0, z_val=-0.25)
        state = SimpleNamespace(
            kinematics_estimated=SimpleNamespace(linear_velocity=velocity)
        )
        fake_client = SimpleNamespace(
            getMultirotorState=lambda: state,
            simGetVehiclePose=lambda: SimpleNamespace(orientation=orientation),
        )
        client = AirLearningClient.__new__(AirLearningClient)
        client.client = fake_client

        np.testing.assert_allclose(
            client.get_body_velocity(),
            np.array([2.0, 0.0, -0.25], dtype=np.float32),
            atol=1e-6,
        )

    def test_navigation_state_has_source_order_and_airsim_ned_vertical_terms(self):
        state = AirSimEnv._navigation_base_state(
            goal=np.array([4.0, 6.0, -2.0], dtype=np.float32),
            position=np.array([1.0, 2.0, -1.0], dtype=np.float32),
            body_velocity=np.array([3.0, 4.0, 0.2], dtype=np.float32),
            yaw=0.25,
        )

        np.testing.assert_allclose(
            state,
            np.array(
                [
                    math.log(6.0),
                    5.0,
                    math.atan2(4.0, 3.0),
                    math.atan2(4.0, 3.0),
                    -1.0,
                    0.2,
                    0.25,
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_reward_uses_raw_terms_without_speed_limit_penalty(self):
        env = AirSimEnv.__new__(AirSimEnv)
        env.goal = np.array([3.0, 4.0, -1.0], dtype=np.float32)
        env.airgym = SimpleNamespace(
            get_ryp=lambda: (0.0, 0.0, 0.0),
            get_body_velocity=lambda: np.array([4.0, 0.0, 0.0], dtype=np.float32),
        )
        env.prev_velocity = np.zeros(3, dtype=np.float32)
        env.prev_goal_dist = math.sqrt(26.0)
        env.use_stagnation_penalty = False
        env.displacement_window = []
        env.stagnation_window = 1
        env.stagnation_window_threshold = 0.0
        env.stagnation_weight = 0.0
        env.last_distance_sensor_scan_distance = np.array([], dtype=np.float32)
        env._compute_distance_sensor_log_penalty = lambda *_args, **_kwargs: 0.0

        reward = env.computeReward(
            np.zeros(3, dtype=np.float32),
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            velocity_after=np.array([0.3, 0.4, 0.0], dtype=np.float32),
        )

        expected = -math.log(6.0) - 1.0 - math.atan2(4.0, 3.0) - 0.5
        self.assertAlmostEqual(reward, expected, places=6)
        self.assertEqual(env.last_reward_components["speed_penalty_disabled"], 0.0)
        self.assertIn("legacy_obstacle_penalty_inactive", env.last_reward_components)


if __name__ == "__main__":
    unittest.main()
