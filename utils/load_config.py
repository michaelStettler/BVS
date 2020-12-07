import os
import json


def load_config(config_name):
    config_path = 'configs/norm_base_config'
    config_file_path = os.path.join(config_path, config_name)
    with open(config_file_path) as json_file:
        config = json.load(json_file)
    return config