import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from datasets_utils.morphing_space import get_NRE_from_morph_space
from utils.extraction_model import load_extraction_model
from utils.RBF_patch_pattern.construct_patterns import construct_RBF_patterns
from plots_utils.plot_BVS import display_image
from plots_utils.plot_BVS import display_images

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.01_morph_space_with_NRE
"""

#%%
# declare script variables
load_LMK_pattern = False

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
LMK_train = train_data  # take all

print("-- Data Split --")
print("len NRE_train[0]", len(NRE_train[0]))
print("NRE_train[1]")
print(NRE_train[1])
print()

#%%
# display NRE training images
display_images(NRE_train[0], pre_processing='VGG19', n_max_col=4)

#%%
# predict images
# load feature extraction model
v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(v4_model.output)[1:3])
print("-- Extraction Model loaded --")
print("size_ft", size_ft)
print()


#%%
# train LMK detector
if load_LMK_pattern:
    print("load LMKs")

else:
    print("create patterns")
    patterns, sigma = construct_RBF_patterns(train_data[0], v4_model, lmk_type, config,
                                             init_sigma=init_sigma,
                                             im_ratio=im_ratio,
                                             k_size=k_size,
                                             use_only_last=use_only_last,
                                             loaded_patterns=patterns,
                                             loaded_sigma=sigma,
                                             train_idx=train_idx,
                                             lmk_name=lmk_name)
