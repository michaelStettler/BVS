import json
import os
import numpy as np
from utils.load_data import load_data

from models.NormBase import NormBase
import matplotlib.pyplot as plt

congig_path = 'configs/norm_base_config'
config_name = 'norm_base_monkey_test.json'
config_file_path = os.path.join(congig_path, config_name)
print("config_file_path", config_file_path)

# load norm_base_config file
with open(config_file_path) as json_file:
    config = json.load(json_file)

# load data
x, y = load_data(config, train=False, sort_by=['image'])
print("[Data] -- Data loaded --")
print("[Data] shape dataX", np.shape(x))
print("[Data] shape dataY", np.shape(y))

# create model
norm_base = NormBase(config, input_shape=(224, 224, 3))

# "load" model
load_folder = os.path.join("models/saved", config['save_name'])
r = np.load(os.path.join(os.path.join(load_folder, "ref_vector.npy")))
t = np.load(os.path.join(os.path.join(load_folder, "tuning_vector.npy")))
norm_base.set_ref_vector(r)
norm_base.set_tuning_vector(t)
print("[MODEL] Set ref vector", np.shape(r))
print("[MODEL] Set tuning vector", np.shape(t))

# predict tuning vector
it_resp = norm_base.predict(x)
print("shape it_resp", np.shape(it_resp))
plt.plot(it_resp)
plt.show()
