from settings_folder import settings
import os
from environment_randomization.game_config_class import *
import copy
from common import utils
from common.file_handling import *
from environment_randomization.deterministic_sampler import DeterministicSampler, get_deterministic_end_point

class GameConfigHandler:
    def __init__(self,
                 range_dic_name="settings.default_range_dic",
                 input_file_addr=settings.json_file_addr):

        range_dic=eval(range_dic_name)
        assert (os.path.isfile(input_file_addr)), input_file_addr + " doesnt exist"
        self.input_file_addr = input_file_addr

        #当前game中环境config
        self.cur_game_config = GameConfig(input_file_addr)

        #game中元素的变化范围
        self.game_config_range = copy.deepcopy(self.cur_game_config)
        self.set_range(*[el for el in range_dic.items()])#items将dict转成元组对


    def set_items_without_modifying_json(self, *arg):
        for el in arg:
            assert (type(
                el) is tuple), el + " needs to be tuple, i.e. the input needs to be provided in the form of (key, new_value)"
            key = el[0]
            value = el[1]
            assert (key in self.cur_game_config.find_all_keys()), key + " is not a key in the json file"
            self.cur_game_config.set_item(key, value)

    def update_json(self, *arg):
        self.set_items_without_modifying_json(*arg)
        outputfile = self.input_file_addr
        output_file_handle = open(outputfile, "w")
        json.dump(self.cur_game_config.config_data, output_file_handle)
        output_file_handle.close()

    def get_cur_item(self, key):
        return self.cur_game_config.get_item(key)

    def set_range(self, *arg):
        for el in arg:

            assert (type(el) is tuple), str(
                el) + " needs to be tuple, i.e. the input needs to be provided in the form of (key, range)"
            assert (type(el[1]) is list), str(
                el) + " needs to be list, i.e. the range needs to be provided in the form of [lower_bound,..., upper_bound]"
            key = el[0]
            value = el[1]
            assert (key in self.cur_game_config.find_all_keys()), key + " is not a key in the json file"
            self.game_config_range.set_item(key, value)

    def get_range(self, key):
        return self.game_config_range.get_item(key)

    # sampling within the entire range
    def sample(self, *arg, np_random=None, change_counter=None, base_seed=None):
        """
        采样环境参数。
        
        如果提供 change_counter 和 base_seed，使用确定性采样，确保：
        - 相同 (base_seed, change_counter) → 相同环境
        - 不同 base_seed → 不同环境序列
        - 第 N 次环境变化总是产生相同的环境（无论什么时候运行）
        
        Args:
            *arg: 要采样的参数名
            np_random: numpy 随机数生成器（向后兼容）
            change_counter: 环境变化计数器（第几次环境变化）
            base_seed: 基础种子（用于确定性采样）
        """
        all_keys = self.game_config_range.find_all_keys()
        if len(arg) == 0:
            arg = all_keys
        
        # 判断是否使用确定性采样
        use_deterministic = (change_counter is not None and base_seed is not None)
        if use_deterministic:
            sampler = DeterministicSampler(base_seed)

        for el in arg:
            assert el in all_keys, str(el) + " is not a key in the json file"

            # corner cases
            if el in ["Indoor", "GameSetting"]:
                continue
            
            param_range = self.game_config_range.get_item(el)
            
            if use_deterministic:
                # 确定性采样：基于 change_counter 和 base_seed
                random_val = sampler.choice(change_counter, el, param_range)
            else:
                # 向后兼容：使用 np_random
                low_bnd = 0
                up_bnd = len(param_range)
                idx = np_random.choice(list(range(low_bnd, up_bnd)))
                random_val = param_range[idx]

            self.cur_game_config.set_item(el, random_val)

        # end
        if "End" in arg and self.game_config_range.get_item("End")[0] == "Mutable":
            if use_deterministic:
                # 确定性生成终点
                end_point = get_deterministic_end_point(
                    self.cur_game_config.get_item("ArenaSize"),
                    change_counter,
                    base_seed)
            else:
                # 向后兼容
                end_point = utils.get_random_end_point(
                    self.cur_game_config.get_item("ArenaSize"),
                    0, 1, np_random)
            self.cur_game_config.set_item("End", end_point)

        outputfile = self.input_file_addr
        output_file_handle = open(outputfile, "w")
        json.dump(self.cur_game_config.config_data, output_file_handle)
        output_file_handle.close()
        if not( settings.ip == '127.0.0.1'):
            utils.copy_json_to_server(outputfile)
