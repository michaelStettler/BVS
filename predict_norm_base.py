import json
import os
import numpy as np
from utils.load_data import load_data

from models.NormBase import NormBase

congig_path = 'configs/norm_base_config'
config_name = 'norm_base_monkey_test.json'
config_file_path = os.path.join(congig_path, config_name)
print("config_file_path", config_file_path)

# load norm_base_config file
with open(config_file_path) as json_file:
    config = json.load(json_file)
