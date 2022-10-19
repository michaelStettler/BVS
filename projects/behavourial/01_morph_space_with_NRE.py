import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from datasets_utils.morphing_space import get_NRE_from_morph_space
from utils.extraction_model import load_extraction_model
from utils.RBF_patch_pattern.load_RBF_patterns import load_LMK_patterns_and_sigma
from utils.RBF_patch_pattern.construct_patterns import create_RBF_LMK
from plots_utils.plot_BVS import display_images

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.01_morph_space_with_NRE
"""

#%%
# declare script variables
load_LMK_pattern = True
n_iter = 2

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
# load feature extraction model
v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(v4_model.output)[1:3])
print("-- Extraction Model loaded --")
print("size_ft", size_ft)
print()


#%%
# get RBF LMK detector
if load_LMK_pattern:
    print("load LMKs")
    FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list = \
        load_LMK_patterns_and_sigma(config, avatar_name=["human", "monkey"])

else:
    print("create patterns")
    FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list = \
        create_RBF_LMK(config, LMK_train, v4_model, n_iter=n_iter)

#%%
print("len FR_patterns_list", len(FR_patterns_list))
print("len FR_patterns_list[0]", len(FR_patterns_list[0]))
print("len FR_patterns_list[1]", len(FR_patterns_list[1]))
print("len FER_patterns_list", len(FER_patterns_list))

#%% predict LMK pos
