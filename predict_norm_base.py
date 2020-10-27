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

# load data
x, y = load_data(config, train=False)
print("[Data] -- Data loaded --")
print("[Data] shape dataX", np.shape(x))
print("[Data] shape dataY", np.shape(y))

# create model
norm_base = NormBase(config, input_shape=(224, 224, 3))

# "load" model
load_folder = os.path.join("models/saved", config['save_name'])
m = np.load(os.path.join(os.path.join(load_folder, "ref_vector.npy")))
n = np.load(os.path.join(os.path.join(load_folder, "tuning_vector.npy")))
norm_base.set_ref_vector(m)
norm_base.set_tuning_vector(n)
print("[MODEL] Set ref vector", np.shape(m))
print("[MODEL] Set tuning vector", np.shape(n))
