import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.NormBase import NormBase
from utils.load_config import load_config
from utils.load_Flame import load_FLAME_csv_params

"""
run: python -m tests.NormBase.t13_raw_face_space
"""

np.set_printoptions(precision=2, linewidth=200, suppress=True)

config_path = 'NB_t13_raw_face_space_m0001.json'
# load config
config = load_config(config_path, path='configs/norm_base_config')

model = NormBase(config, tuple(config['input_shape']))
# compute ref
ref_idx = 0
# create train_x with orthogonal space
train_x = np.eye(8)
train_x[0, 0] = 0
train_y = np.arange(8)
print("label")
print(train_y)
ref = np.mean(train_x[train_y == ref_idx], axis=0)
print("ref", np.shape(ref))
print(ref)

# compute tuning
tuning_directions = []
for i in range(8):
    if i == ref_idx:
        tun_dir = np.zeros(np.shape(ref))
    else:
        tun_dir = np.mean(train_x[train_y == i], axis=0)

    tuning_directions.append(tun_dir)
tuning_directions = np.array(tuning_directions)

print("tuning_direction", np.shape(tuning_directions))
print(tuning_directions)

model.set_ref_vector(ref)
model.set_tuning_vector(tuning_directions)

print("compute it_resp")
test_x = np.eye(8)
test_y = np.arange(8)
print("test_x", np.shape(test_x))
it_resp = model._get_it_resp(test_x)
print("shape it_resp", np.shape(it_resp))
print(it_resp[0])
y_pred = np.argmax(it_resp, axis=1)
print("y_pred")
print(y_pred)
print()

accuracy = (test_y == np.argmax(it_resp, axis=1)).mean()
print("----------------------------------------------------")
print("Test accuracy:", accuracy)
print("----------------------------------------------------")
