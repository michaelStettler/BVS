import os
import numpy as np
import pickle
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.feature_reduction import load_feature_selection
from utils.Semantic.SemanticFeatureSelection import SemanticFeatureSelection
from utils.extraction_model import load_extraction_model
from plots_utils.plot_cnn_output import plot_cnn_output
"""
Train a model with the human and monkey avatars to investigate the highest variable features

run: python -m tests.CNN.t02b_compare_PCA_idx
"""
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)


config_path = 'CNN_t02b_compare_PCA_idx_m0001.json'
load_semantic_dict = False

# load config
config = load_config(config_path, path='configs/CNN')
folder_path = os.path.join("models/saved", config['config_name'])

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# load model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input,
                          outputs=model.get_layer(config['v4_layer']).output)
print("[INIT] extraction model loaded")


def get_most_var_idx(config, model):
    # load data
    data = load_data(config)
    print("[INIT] shape data", np.shape(data[0]))

    # predict feature maps input
    preds = model.predict(data[0], verbose=True)
    preds_flat = np.reshape(preds, (len(preds), -1))
    print("[FIT] shape preds", np.shape(preds))

    # keep only highest variance
    var = np.std(preds_flat, axis=0)
    print("[FIT] shape var", np.shape(var))
    # get the n_component's max index
    index = np.flip(np.argsort(var))[:config['PCA']]
    # transform index to feature maps index
    x, y, k = np.unravel_index(index, np.shape(preds)[1:])
    print("ft index", k)

    # clean list by removing repetitions
    ft_index = []
    for idx in k:
        exists = False
        for ft_idx in ft_index:
            if idx == ft_idx:
                exists = True

        if not exists:
            ft_index.append(idx)

    print("feat maps index")
    print(ft_index)
    print()

    return ft_index


def find_common_index(index_ref, index_target):
    c_index = []

    for idx_r in index_ref:
        common = False
        for idx_t in index_target:
            if idx_r == idx_t:
                common = True

        if common:
            c_index.append(idx_r)

    return c_index


# --------------------------------------------------------------------------------------------------------------------
# Human avatar
h_idx = get_most_var_idx(config, model)

# --------------------------------------------------------------------------------------------------------------------
# Monkey avatar
config['train_avatar'] = 'monkey_orig'
m_idx = get_most_var_idx(config, model)

# --------------------------------------------------------------------------------------------------------------------
# Monkey + Human avatar
config['train_avatar'] = 'all_orig'
both_idx = get_most_var_idx(config, model)

# --------------------------------------------------------------------------------------------------------------------
# find index common between all avatars
# 1st between human and both
com_h = find_common_index(h_idx, both_idx)

# 2nd between monkey and both
com_m = find_common_index(m_idx, both_idx)

# 3rd between both common
com = find_common_index(com_h, com_m)
print("common index", com)

# --------------------------------------------------------------------------------------------------------------------
# plot common feature maps over monkey avatar

config['train_avatar'] = 'monkey_orig'
# load data
data = load_data(config)

# predict feature maps input
preds = model.predict(data[0], verbose=True)
print("[FIT] shape preds", np.shape(preds))
preds_common = preds[..., com]
print("[FIT] shape preds_common", np.shape(preds_common))

plot_cnn_output(preds_common, os.path.join("models/saved", config["config_name"]),
                config['train_avatar'] + "_common_PCA" + "_selection.gif",
                video=True,
                verbose=True)

# compute dynamic changes
preds_ref = preds_common[0]
preds_dyn_com = preds_common - np.repeat(np.expand_dims(preds_ref, axis=0), len(preds_common), axis=0)
preds_dyn_com[preds_dyn_com < 0] = 0
preds_dyn_com /= np.amax(preds_dyn_com)

plot_cnn_output(preds_dyn_com, os.path.join("models/saved", config["config_name"]),
                config['train_avatar'] + "_common_PCA_dyn" + "_selection.gif",
                video=True,
                verbose=True)

# --------------------------------------------------------------------------------------------------------------------
# plot best IoU feature maps over human avatar

config['train_avatar'] = 'human_orig'
best_eye_brow_IoU_ft = [209, 148, 59, 208]
# load data
data = load_data(config)

# predict feature maps input
preds = model.predict(data[0], verbose=True)
print("[FIT] shape preds", np.shape(preds))
preds_common = preds[..., best_eye_brow_IoU_ft]
print("[FIT] shape preds_common", np.shape(preds_common))

plot_cnn_output(preds_common, os.path.join("models/saved", config["config_name"]),
                config['train_avatar'] + "_best_eyebrow_IoU" + "_selection.gif",
                video=True,
                verbose=True)

# compute dynamic changes
preds_ref = preds_common[0]
preds_dyn_com = preds_common - np.repeat(np.expand_dims(preds_ref, axis=0), len(preds_common), axis=0)
preds_dyn_com[preds_dyn_com < 0] = 0
preds_dyn_com /= np.amax(preds_dyn_com)

plot_cnn_output(preds_dyn_com, os.path.join("models/saved", config["config_name"]),
                config['train_avatar'] + "_best_eyebrow_IoU_dyn" + "_selection.gif",
                video=True,
                verbose=True)