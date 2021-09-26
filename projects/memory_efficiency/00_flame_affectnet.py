import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.NormBase import NormBase
from utils.load_config import load_config
from utils.load_Flame import load_FLAME_csv_params

"""
run: python -m projects.memory_efficiency.00_flame_affectnet
"""

np.set_printoptions(precision=2, linewidth=200, suppress=True)

config_path = 'NB_Memory_Efficiency.json'
# load config
config = load_config(config_path, path='configs/norm_base_config')


path = "/Users/michaelstettler/PycharmProjects/BVS/data/AffectNet_FLAME"
train_csv = "flame_training_params.csv"
test_csv = "flame_validation_params.csv"

df_train = pd.read_csv(os.path.join(path, train_csv), index_col=0)
df_test = pd.read_csv(os.path.join(path, test_csv), index_col=0)
print(df_train.head())
# print(df_test.head())

train_data = load_FLAME_csv_params(df_train)
test_data = load_FLAME_csv_params(df_test)

model = NormBase(config, tuple(config['input_shape']))

# compute ref
ref_idx = 0
train_x = train_data[0]
train_y = train_data[1]
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
test_x = test_data[0]
print("test_x", np.shape(test_x))
it_resp = model._get_it_resp(test_x)
print("shape it_resp", np.shape(it_resp))

