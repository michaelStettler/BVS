"""
Script to compare the index retained from the fitted PCA

The script train a norm base model with the three following conditions:
    - human
    - monkey
    - human + monkey
and compare the most important features kept by the PCA

run: python compare_var_index.py
"""

import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

np.set_printoptions(precision=3, linewidth=250, suppress=True)

# small testing examples
test = np.arange(36)
n_feature_map = 4
feature_map_size = 3
t = np.reshape(test, (feature_map_size, feature_map_size, n_feature_map))
print(t[:, :, 0])
print(t[:, :, 1])
print(t[:, :, 2])
print(t[:, :, 3])
x,y,z = np.unravel_index(17, (feature_map_size,feature_map_size,n_feature_map))
#x, y, z = get_feature_map_index(17, n_feature_map, feature_map_size)
print("({}, {}, {})".format(x, y, z))


# load configuration
config_name = "norm_base_investigate_PCA_m0001.json"
config = load_config(config_name)

# fit models with each condition
avatars = ["human_orig", "monkey_orig", "all_orig"]
# avatars = ["human_orig"]
indexes = []
# pca_threshold = [300, 300, 1500]
pca_threshold = [600, 600, 2000]
for i, avatar in enumerate(avatars):
    # modify condition according to loop
    config['train_avatar'] = avatar

    # define and train norm base model
    norm_base = NormBase(config, input_shape=(224, 224, 3))
    norm_base.pca.var_threshold = pca_threshold[i]
    norm_base.fit(load_data(config, train=True), fit_dim_red=True, fit_ref=False, fit_tun=False)

    # get index from the feature having the most variance
    predict_v4 = norm_base.v4_predict
    var_predict = np.std(predict_v4, axis=0)
    index = np.flip(np.argsort(var_predict))[:config['PCA']]

    # save index
    indexes.append(np.array(index))
indexes = np.array(indexes)


# get position within feature maps
v4_shape = norm_base.shape_v4
print("v4_shape", v4_shape)
n_feature_map = v4_shape[-1]
feature_map_size = v4_shape[1]
print("feature map size: ({}, {}, {})".format(feature_map_size, feature_map_size, n_feature_map))
print()

positions = []
for a, avatar in enumerate(avatars):
    avatar_positions = []
    for i in range(len(indexes[a])):
        index = indexes[a, i]
        x,y,f = np.unravel_index(index, v4_shape)
        #(x, y, f) = get_feature_map_index(index, n_feature_map, feature_map_size)
        avatar_positions.append((x, y, f))
    positions.append(np.array(avatar_positions))

positions = np.array(positions)
print("indexes:")
print(indexes)

# print positions
for i in range(np.shape(positions)[1]):
    print("h: ({}) m: ({}) h+m ({})".format(positions[0, i], positions[1, i], positions[2, i]))

# print only features maps index
f_map_h = np.sort(positions[0, :, 2])
f_map_m = np.sort(positions[1, :, 2])
f_map_hm = np.sort(positions[2, :, 2])
print(f_map_h)
print(f_map_m)
print(f_map_hm)