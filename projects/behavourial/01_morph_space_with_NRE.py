import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from datasets_utils.morphing_space import get_NRE_from_morph_space
from plots_utils.plot_BVS import display_image

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.01_morph_space_with_NRE
"""

#%%
# import config
config_path = 'BH_01_morph_space_with_NRE_m0001.json'
# load config
config = load_config(config_path, path='configs/behavourial')
print("-- Config loaded --")
print()

#%%
# import data
train_data = load_data(config)
print("-- Data loaded --")
print("len train_data[0]", len(train_data[0]))
print()

#%% split training for LMK and norm base
NRE_train = get_NRE_from_morph_space(train_data)
LMK_train = train_data[0]  # take all
