"""SSVM-SAC 专属的 AirSim 环境。

课程从 easy 插值到配置的最终等级（默认 level 3）。只有终点为 level 3
时加入动态障碍物；level 0-2 使用对应的静态障碍物密度。
"""

import numpy as np

from common import utils
from gym_airsim.envs.AirGym import AirSimEnv
from settings_folder import settings


class AirSimEnvSSVM(AirSimEnv):
    def set_curriculum_progress(self, progress_ratio):
        final_level = getattr(self.config, "curriculum_final_level", 3)
        target_config = {
            0: settings.easy_range_dic,
            1: settings.medium_range_dic,
            2: settings.hard_range_dic,
            3: settings.dynamic_obstacles_dic,
        }[final_level]
        progress_ratio = float(np.clip(progress_ratio, 0.0, 1.0))
        difficulty = float(
            np.clip(progress_ratio / self.curriculum_progress_max_ratio, 0.0, 1.0)
        )

        # 静态障碍物从 easy 线性插值到选定终点。
        easy_min, easy_max = self._get_number_of_objects_bounds(
            settings.easy_range_dic["NumberOfObjects"]
        )
        dynamic_static_min, dynamic_static_max = self._get_number_of_objects_bounds(
            target_config["NumberOfObjects"]
        )
        cur_min = int(round(easy_min + difficulty * (dynamic_static_min - easy_min)))
        cur_max = int(round(easy_max + difficulty * (dynamic_static_max - easy_max)))
        self.game_config_handler.set_range(
            ("NumberOfObjects", list(range(cur_min, cur_max)))
        )

        # 静态等级的目标范围为 [0]，整个课程都不生成动态障碍物。
        dynamic_min, dynamic_max = self._get_number_of_objects_bounds(
            target_config["NumberOfDynamicObjects"]
        )
        cur_dynamic_min = int(round(difficulty * dynamic_min))
        cur_dynamic_max = max(cur_dynamic_min + 1, int(round(difficulty * dynamic_max)))
        self.game_config_handler.set_range(
            ("NumberOfDynamicObjects", list(range(cur_dynamic_min, cur_dynamic_max)))
        )

        self.curriculum_progress_ratio = progress_ratio
        self.curriculum_difficulty = difficulty
        self.curriculum_number_of_objects_min = cur_min
        self.curriculum_number_of_objects_max = cur_max

        self.level = min(final_level, int(difficulty * (final_level + 1)))

        return {
            "progress_ratio": progress_ratio,
            "difficulty": difficulty,
            "level": self.level,
            "number_of_objects_min": cur_min,
            "number_of_objects_max": cur_max,
        }

    def randomize_env(self):
        # NumberOfDynamicObjects 像静态障碍物一样每个 episode 重采样，
        # 使动态障碍物数量随课程进度平滑递增（不改全局 settings）。
        frequency = dict(settings.environment_change_frequency)
        frequency["NumberOfDynamicObjects"] = 1
        vars_to_randomize = []
        for k, v in frequency.items():
            if (self.episodeN + 1) % v == 0:
                vars_to_randomize.append(k)
        if len(vars_to_randomize) > 0:
            print(f"Randomizing environment vars: {vars_to_randomize}")
            self.sampleGameConfig(*vars_to_randomize)
            self.goal = utils.airsimize_coordinates(
                self.game_config_handler.get_cur_item("End")
            )
            return True
        return False
