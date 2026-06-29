#!/usr/bin/env python3
"""AirSim Gym environment for scene evaluation.

This environment intentionally avoids GameConfigHandler and EnvGenConfig.json.
The UE4 scene is assumed to be static and already contains the geometry.
"""

from __future__ import annotations

import collections
import time

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from gym_airsim.envs.AirGym import AirSimEnv
from gym_airsim.envs.airlearningclient import AirLearningClient
from settings_folder import settings

from eval.launcher import DEFAULT_SCENE_PROJECT, DEFAULT_UNREAL_EDITOR, SceneGameHandler


class SceneEvalAirSimEnv(AirSimEnv):
    """AirSimEnv-compatible wrapper for the UE4 evaluation scene."""

    def __init__(
        self,
        takeoff_height: float = -1.0,
        config=None,
        stack_frames: int = 4,
        *,
        goal_xyz=(32.0, 32.0, -1.0),  # AirSim NED 坐标系: X=北, Y=东, Z=下 (负值=向上, -1.0 即离地 1m)
        project_file: str | None = None,
        unreal_editor: str | None = None,
    ) -> None:
        self.config = config
        self.stack_frames = int(stack_frames)
        self.episode_reward = 0.0
        self.base_dim = 11
        self.depth_shape = None
        self.observation_space = None
        self.depth_stack = collections.deque(maxlen=self.stack_frames)
        self.clean_depth_stack = collections.deque(maxlen=self.stack_frames)

        algorithm_name_upper = str(getattr(config, "algorithm_name", "")).upper()
        self.use_clean_privileged_obs = "PL_" in algorithm_name_upper or algorithm_name_upper.startswith("PL")

        self.max_altitude = float(config.max_flight_altitude)
        self.min_altitude = float(config.min_flight_altitude)
        self.min_altitude_penalty = float(config.min_altitude_penalty)
        self.max_altitude_penalty = float(config.max_altitude_penalty)
        self.altitude_penalty_value = float(config.altitude_penalty_value)
        self.use_stagnation_penalty = bool(config.use_stagnation_penalty)
        self.stagnation_window = int(config.stagnation_window)
        self.stagnation_window_threshold = float(config.stagnation_window_threshold)
        self.stagnation_weight = float(config.stagnation_weight)
        self.displacement_window = collections.deque(maxlen=self.stagnation_window)

        if settings.control_mode == "moveByVelocity":
            self.action_space = spaces.Box(
                np.array([-0.3, -0.3], dtype=np.float32),
                np.array([+0.3, +0.3], dtype=np.float32),
                dtype=np.float32,
            )
        elif settings.control_mode == "Continuous":
            self.action_space = spaces.Box(
                low=np.array([config.min_forward_speed, -config.max_yaw_rate, -config.max_vertical_speed], dtype=np.float32),
                high=np.array([config.max_forward_speed, config.max_yaw_rate, config.max_vertical_speed], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(8)

        self.use_curriculum = False
        self.curriculum_mode = "scene"
        self.curriculum_progress_ratio = 1.0
        self.curriculum_difficulty = 1.0
        self.curriculum_number_of_objects_min = None
        self.curriculum_number_of_objects_max = None
        self.level = -1
        self.base_seed = int(config.seed)
        self.seed(self.base_seed)

        self.game_handler = None if bool(config.disable_game_restart) else SceneGameHandler(
            project_file=project_file
            or getattr(config, "scene_project_file", getattr(config, "st_project_file", DEFAULT_SCENE_PROJECT)),
            unreal_editor=unreal_editor
            or getattr(config, "scene_unreal_editor", getattr(config, "st_unreal_editor", DEFAULT_UNREAL_EDITOR)),
        )
        if self.game_handler is not None:
            self.game_handler.restart_game()

        client_ip = config.airsim_ip if config is not None else None
        client_port = config.airsim_port if config is not None else None
        self.airgym = AirLearningClient(z=takeoff_height, ip=client_ip, port=client_port, config=config)

        try:
            depth_sample = self.airgym.getScreenDepth(max_attempts=3)
            state_depth_h, state_depth_w = depth_sample.shape
            print(f"Detected AirSim depth resolution: {state_depth_w}x{state_depth_h}")
        except Exception as exc:
            print(f"WARNING: Failed to get depth sample during init, falling back to 128x128: {exc}")
            state_depth_h, state_depth_w = 128, 128

        self.enable_takeoff_obstacle_check = bool(config.enable_takeoff_obstacle_check)
        self.takeoff_obstacle_reset_retries = max(0, int(config.takeoff_obstacle_reset_retries))
        self.distance_sensor_count = max(1, int(config.distance_sensor_count))
        self.distance_sensor_prefix = str(config.distance_sensor_prefix).strip()
        self.distance_sensor_start_index = int(config.distance_sensor_start_index)
        self.distance_sensor_names = self.airgym._resolve_distance_sensor_names(
            sensor_names=config.distance_sensor_names,
            sensor_prefix=self.distance_sensor_prefix,
            sensor_count=self.distance_sensor_count,
            start_index=self.distance_sensor_start_index,
        )
        self.distance_sensor_count = len(self.distance_sensor_names)
        self.distance_sensor_log_penalty_min = float(config.distance_sensor_log_penalty_min)
        self.distance_sensor_penalty_max_distance = max(1e-6, float(config.distance_sensor_penalty_max_distance))
        self.distance_sensor_penalty_eps = max(1e-6, float(config.distance_sensor_penalty_eps))
        self.distance_sensor_query_max_attempts = max(1, int(config.distance_sensor_query_max_attempts))
        self.distance_sensor_query_retry_sleep = max(0.0, float(config.distance_sensor_query_retry_sleep))
        self.last_distance_sensor_obstacle_penalty = 0.0
        self.last_distance_sensor_max_distance = np.ones((self.distance_sensor_count,), dtype=np.float32)
        self.last_distance_sensor_scan_distance = np.full((self.distance_sensor_count,), 1.0, dtype=np.float32)
        self.distance_sensor_read_fail_count = 0

        self.depth_shape = (self.stack_frames, state_depth_h, state_depth_w)
        observation_spaces = {
            "depth": spaces.Box(low=np.float32(0), high=np.float32(255), shape=self.depth_shape, dtype=np.float32),
            "base": spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_dim,), dtype=np.float32),
            "distance_sensor": spaces.Box(low=0.0, high=np.inf, shape=(self.distance_sensor_count,), dtype=np.float32),
        }
        if self.use_clean_privileged_obs:
            observation_spaces["clean_depth"] = spaces.Box(
                low=np.float32(0), high=np.float32(255), shape=self.depth_shape, dtype=np.float32
            )
        self.observation_space = spaces.Dict(observation_spaces)

        self.action_duration = float(config.action_duration)
        self.success_count = 0
        self.episodeN = 0
        self.stepN = 0
        self.total_step_count = 0
        self.goal = np.asarray(goal_xyz, dtype=np.float32)
        self.success = False
        self.success_deque = collections.deque(maxlen=256)
        self.ue4_rpc_fail_count = 0
        self.ue4_rpc_fail_threshold = int(config.ue4_rpc_fail_threshold)
        self.ue4_health_check_interval = float(config.ue4_health_check_interval)
        self.ue4_window_check_interval = float(config.ue4_window_check_interval)
        self.ue4_process_check_interval = max(3.0, self.ue4_health_check_interval * 3.0)
        self._last_ue4_health_check_ts = 0.0
        self._last_process_check_ts = 0.0
        self._last_window_check_ts = 0.0
        self._cached_process_alive = True
        self._cached_window_alive = None
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32) if hasattr(self.action_space, "shape") and self.action_space.shape else 0
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.prev_pos_xy = None
        self.prev_goal_dist = 0.0

        self.init_state_f()
        self.prev_state = self.get_obs()
        self.init_state = self.prev_state

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_curriculum_progress(self, progress_ratio):
        return None

    def get_curriculum_info(self):
        return {
            "progress_ratio": 1.0,
            "difficulty": 1.0,
            "level": self.level,
            "number_of_objects_min": None,
            "number_of_objects_max": None,
        }

    def check_ue4_status(self, force_restart: bool = False, reason: str = "") -> bool:
        if self.game_handler is None:
            return False

        now = time.time()
        if not force_restart and (now - self._last_process_check_ts) < self.ue4_process_check_interval:
            return False
        self._last_process_check_ts = now

        if force_restart or not self.game_handler.is_game_process_alive():
            print(f"WARNING: UE4 evaluation scene unhealthy ({reason or 'process_missing'}). Restarting.")
            self.game_handler.restart_game()
            self.airgym = AirLearningClient(
                z=self.airgym.z,
                ip=self.config.airsim_ip if self.config else None,
                port=self.config.airsim_port if self.config else None,
                config=self.config,
            )
            return True
        return False

    def randomize_env(self):
        return False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.check_ue4_status()
        self.airgym.client.simPause(False)
        try:
            self.airgym.client.reset()
            self.airgym.client.enableApiControl(True)
            self.airgym.client.armDisarm(True)
            self.airgym.client.moveToZAsync(float(self.airgym.z), 1.0).join()
            self.airgym.client.hoverAsync().join()
        except Exception:
            self.airgym.AirSim_reset()

        now = self.airgym.drone_pos()
        if abs(now[2] - self.airgym.z) > 0.1:
            self.airgym.client.moveToZAsync(self.airgym.z, 3).join()
            now = self.airgym.drone_pos()

        self.airgym.client.simPause(True)
        self.on_episode_start()
        state = self.init_state_f()
        self._update_distance_sensor_obstacle_distance()
        self.prev_state = state
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32) if hasattr(self.action_space, "shape") and self.action_space.shape else 0
        self.prev_velocity = self.airgym.drone_velocity()
        self.prev_pos_xy = None
        self.prev_goal_dist = float(np.linalg.norm(
            np.array([self.goal[0] - now[0], self.goal[1] - now[1], self.goal[2] - now[2]], dtype=np.float32)
        ))
        self.displacement_window.clear()
        return state, {}
