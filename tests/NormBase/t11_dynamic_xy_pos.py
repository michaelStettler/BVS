import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from plots_utils.plot_cnn_output import plot_cnn_output
from utils.calculate_position import calculate_position
from plots_utils.plot_ft_map_pos import plot_ft_map_pos
from models.NormBase import NormBase

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the fit a face template over different feature maps instead of simply summing them up
the term dynmic means here that I am trying to get the specific "moving parts" from the sequence 

run: python -m tests.NormBase.t11_dynamic_xy_pos
"""

# define configuration
config_path = 'NB_t11_dynamic_xy_pos_m0001.json'

# declare parameters
best_eyebrow_IoU_ft = [209, 148, 59, 208]
best_lips_IoU_ft = [77, 79, 120, 104, 141, 0, 34, 125, 15, 89, 49, 237, 174, 39, 210, 112, 111, 201, 149, 165, 80,
                         42, 128, 74, 131, 193, 133, 44, 154, 101, 173, 6, 148, 61, 27, 249, 209, 19, 247, 90, 1, 255,
                         182, 251, 186, 248]

# load config
config = load_config(config_path, path='configs/norm_base_config')

# create directory if non existant
save_path = os.path.join("models/saved", config["config_name"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

# load and define model
v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(v4_model.output)[1:3])
print("[LOAD] size_ft", size_ft)
print("[LOAD] Model loaded")
print()

nb_model_eyebrow = NormBase(config, tuple(config['input_shape']))
nb_model_lips = NormBase(config, tuple(config['input_shape']))
# -------------------------------------------------------------------------------------------------------------------
# train

# load data
data = load_data(config)

# predict
preds = v4_model.predict(data[0], verbose=1)
print("[TRAIN] shape prediction", np.shape(preds))

# get feature maps that mimic a semantic selection pipeline
# keep only highest IoU semantic score
eyebrow_preds = preds[..., best_eyebrow_IoU_ft]
print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
lips_preds = preds[..., best_lips_IoU_ft]
print("shape lips semantic feature selection", np.shape(lips_preds))

# compute dynamic directly on the feature maps
# eyebrow
# eyebrow_preds_ref = eyebrow_preds[0]
eyebrow_preds_neut0 = eyebrow_preds[:40]
eyebrow_preds_neut1 = eyebrow_preds[80:]
eyebrow_preds_ref = np.mean(np.concatenate([eyebrow_preds_neut0, eyebrow_preds_neut1]), axis=0)
dyn_eyebrow_preds = eyebrow_preds - np.repeat(np.expand_dims(eyebrow_preds_ref, axis=0), len(eyebrow_preds), axis=0)
dyn_eyebrow_preds[dyn_eyebrow_preds < 0] = 0
# lips
# lips_preds_ref = lips_preds[0]
lips_preds_neut0 = lips_preds[:40]
lips_preds_neut1 = lips_preds[80:]
lips_preds_ref = np.mean(np.concatenate([lips_preds_neut0, lips_preds_neut1]), axis=0)
dyn_lips_preds = lips_preds - np.repeat(np.expand_dims(lips_preds_ref, axis=0), len(lips_preds), axis=0)
dyn_lips_preds[dyn_lips_preds < 0] = 0
print("[TRAIN] finished computing dynamic predictions")
print()

# compute positions eyebrow
dyn_eyebrow_pos = calculate_position(dyn_eyebrow_preds[1:], mode="weighted average", return_mode="xy float flat")
nb_model_eyebrow.n_features = np.shape(dyn_eyebrow_pos)[-1]  # todo add this to init
# train manually ref vector
nb_model_eyebrow.r = np.zeros(nb_model_eyebrow.n_features)
nb_model_eyebrow._fit_reference([dyn_eyebrow_pos, data[1][1:]], config['batch_size'])
# train manually tuning vector
nb_model_eyebrow.t = np.zeros((nb_model_eyebrow.n_category, nb_model_eyebrow.n_features))
nb_model_eyebrow.t_mean = np.zeros((nb_model_eyebrow.n_category, nb_model_eyebrow.n_features))
nb_model_eyebrow._fit_tuning([dyn_eyebrow_pos, data[1][1:]], config['batch_size'])
# get it resp for eyebrows
it_train_eyebrow = nb_model_eyebrow._get_it_resp(dyn_eyebrow_pos)
print("[TRAIN] finished computing eyebrow positions")
print()

# compute positions lips
dyn_lips_pos = calculate_position(dyn_lips_preds[1:], mode="weighted average", return_mode="xy float flat")
nb_model_lips.n_features = np.shape(dyn_lips_pos)[-1]  # todo add this to init
# train manually ref vector
nb_model_lips.r = np.zeros(nb_model_lips.n_features)
nb_model_lips._fit_reference([dyn_lips_pos, data[1][1:]], config['batch_size'])
# train manually tuning vector
nb_model_lips.t = np.zeros((nb_model_lips.n_category, nb_model_lips.n_features))
nb_model_lips.t_mean = np.zeros((nb_model_lips.n_category, nb_model_lips.n_features))
nb_model_lips._fit_tuning([dyn_lips_pos, data[1][1:]], config['batch_size'])
# get it resp for lips
it_train_lips = nb_model_lips._get_it_resp(dyn_lips_pos)
print("[TRAIN] finished computing lips positions")
print()

# -------------------------------------------------------------------------------------------------------------------
# test monkey

# load data
test_data = load_data(config, train=False)
# predict
test_preds = v4_model.predict(test_data[0], verbose=1)
print("[PREDS] shape test_preds", np.shape(test_preds))

# get feature maps that mimic a semantic selection pipeline
# keep only highest IoU semantic score
test_eyebrow_preds = test_preds[..., best_eyebrow_IoU_ft]
print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
test_lips_preds = test_preds[..., best_lips_IoU_ft]
print("shape lips semantic feature selection", np.shape(lips_preds))
test_preds = [test_eyebrow_preds, test_lips_preds]

# compute dynamic feature maps
# test_eyebrow_preds_ref = test_eyebrow_preds[0]
test_eyebrow_preds_neut0 = test_eyebrow_preds[:40]
test_eyebrow_preds_neut1 = test_eyebrow_preds[80:]
test_eyebrow_preds_ref = np.mean(np.concatenate([test_eyebrow_preds_neut0, test_eyebrow_preds_neut1]), axis=0)
test_dyn_eyebrow_preds = test_eyebrow_preds - np.repeat(np.expand_dims(test_eyebrow_preds_ref, axis=0), len(test_eyebrow_preds), axis=0)
test_dyn_eyebrow_preds[test_dyn_eyebrow_preds < 0] = 0

# test_lips_preds_ref = test_lips_preds[0]
test_lips_preds_neut0 = test_lips_preds[:40]
test_lips_preds_neut1 = test_lips_preds[80:]
test_lips_preds_ref = np.mean(np.concatenate([test_lips_preds_neut0, test_lips_preds_neut1]), axis=0)
test_dyn_lips_preds = test_lips_preds - np.repeat(np.expand_dims(test_lips_preds_ref, axis=0), len(test_lips_preds), axis=0)
test_dyn_lips_preds[test_dyn_lips_preds < 0] = 0

# compute positions
dyn_test_eyebrow_pos = calculate_position(test_dyn_eyebrow_preds[1:], mode="weighted average", return_mode="xy float flat")
dyn_test_lips_pos = calculate_position(test_dyn_lips_preds[1:], mode="weighted average", return_mode="xy float flat")

it_test_eyebrow = nb_model_eyebrow._get_it_resp(dyn_test_eyebrow_pos)
it_test_lips = nb_model_lips._get_it_resp(dyn_test_lips_pos)

# test by training new ref
nb_model_eyebrow._fit_reference([dyn_test_eyebrow_pos, data[1][1:]], config['batch_size'])
nb_model_lips._fit_reference([dyn_test_lips_pos, data[1][1:]], config['batch_size'])

it_ref_test_eyebrow = nb_model_eyebrow._get_it_resp(dyn_test_eyebrow_pos)
it_ref_test_lips = nb_model_lips._get_it_resp(dyn_test_lips_pos)

# # --------------------------------------------------------------------------------------------------------------------
# # plots
# ***********************       test 00 raw output      ******************

# # for only c2
# color_seq = np.arange(len(dyn_eyebrow_preds))
# color_seq[:40] = 0
# color_seq[40:80] = 1
# color_seq[80:] = 0

# for c2 and c3
color_seq = np.arange(len(dyn_eyebrow_preds[1:]))
color_seq[:40] = 0
color_seq[40:80] = 1
color_seq[80:190] = 0
color_seq[190:230] = 2
color_seq[230:] = 0

#
# plot_cnn_output(dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(dyn_eyebrow_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_dyn_eyebrow_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

#
# plot_cnn_output(test_dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_dyn_eyebrow_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_dyn_eyebrow_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)
#
# plot_cnn_output(dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(dyn_lips_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_dyn_lips_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)
#
# plot_cnn_output(test_dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_dyn_lips_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_dyn_lips_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# ***********************       test 01 eyebrow model     ******************
# plot it responses for eyebrow model
nb_model_eyebrow.plot_it_neurons(it_train_eyebrow,
                         title="01_it_train_eyebrow",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model_eyebrow.plot_it_neurons(it_test_eyebrow,
                         title="01_it_test_eyebrow",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model_eyebrow.plot_it_neurons(it_ref_test_eyebrow,
                         title="01_it_ref_test_eyebrow",
                         save_folder=os.path.join("models/saved", config["config_name"]))

# ***********************       test 02 lips model     ******************
# plot it responses for lips model
nb_model_lips.plot_it_neurons(it_train_lips,
                         title="02_it_train_lips",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model_lips.plot_it_neurons(it_test_lips,
                         title="02_it_test_lips",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model_lips.plot_it_neurons(it_ref_test_lips,
                         title="02_it_ref_test_lips",
                         save_folder=os.path.join("models/saved", config["config_name"]))

