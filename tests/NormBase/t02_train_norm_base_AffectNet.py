import json
import os
import numpy as np
from utils.load_data import load_data

from models.NormBase import NormBase

congig_path = '../../configs/norm_base_config'
# config_name = 'norm_base_monkey_test.json'
config_name = 'norm_base_affectNet_sub8_4000.json'
config_file_path = os.path.join(congig_path, config_name)
print("config_file_path", config_file_path)

# load norm_base_config file
with open(config_file_path) as json_file:
    config = json.load(json_file)

# load data
data = load_data(config)
print("[Data] -- Data loaded --")

# create model
norm_base = NormBase(config, input_shape=(224, 224, 3))
# norm_base.print_v4_summary()

# train model
m, n = norm_base.fit(data, batch_size=config['batch_size'])

print("shape m", np.shape(m))
print("shape n", np.shape(n))

# save model
save_folder = os.path.join("../../models/saved", config['save_name'])
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
np.save(os.path.join(save_folder, "ref_vector"), m)
np.save(os.path.join(save_folder, "tuning_vector"), n)


