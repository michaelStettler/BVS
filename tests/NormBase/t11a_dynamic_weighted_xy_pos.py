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
from utils.ref_feature_map_neurons import ref_feature_map_neuron

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the fit a face template over different feature maps instead of simply summing them up
the term dynmic means here that I am trying to get the specific "moving parts" from the sequence 

run: python -m tests.NormBase.t11a_dynamic_weighted_xy_pos
"""

# define configuration
config_path = 'NB_t11a_dynamic_xy_pos_m0003.json'

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

nb_model = NormBase(config, tuple(config['input_shape']))
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
dyn_eyebrow_preds = ref_feature_map_neuron(eyebrow_preds, data[1], config, activation='relu')
# lips
dyn_lips_preds = ref_feature_map_neuron(lips_preds, data[1], config, activation='relu')

# concatenate feature maps concepts
dyn_preds = np.concatenate([dyn_eyebrow_preds, dyn_lips_preds], axis=3)
print("[TRAIN] shape dyn preds", np.shape(dyn_preds))
print("[TRAIN] finished computing dynamic predictions")

# compute positions eyebrow
dyn_pos = calculate_position(dyn_preds, mode="weighted average", return_mode="xy float flat")
print("[TRAIN] shape dyn_pos", np.shape(dyn_pos))

# add weights
pos_weights = np.ones(np.shape(dyn_pos)[-1])
pos_weights[:8] = 1 / (2 * 2 * len(best_eyebrow_IoU_ft))  # num_concept * x/y * num_ft/concept
pos_weights[8:] = 1 / (2 * 2 * len(best_lips_IoU_ft))
dyn_pos = np.multiply(dyn_pos, pos_weights)

nb_model.n_features = np.shape(dyn_pos)[-1]  # todo add this to init
# train manually ref vector
nb_model.r = np.zeros(nb_model.n_features)
nb_model._fit_reference([dyn_pos, data[1]], config['batch_size'])
# train manually tuning vector
nb_model.t = np.zeros((nb_model.n_category, nb_model.n_features))
nb_model.t_mean = np.zeros((nb_model.n_category, nb_model.n_features))
nb_model._fit_tuning([dyn_pos, data[1]], config['batch_size'])
# get it resp for eyebrows
it_train = nb_model._get_it_resp(dyn_pos)
print("[TRAIN] finished computing positions")
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
test_dyn_eyebrow_preds = ref_feature_map_neuron(test_eyebrow_preds, test_data[1], config, activation='relu')
test_dyn_lips_preds = ref_feature_map_neuron(test_lips_preds, test_data[1], config, activation='relu')

# concatenate concepts
test_dyn_preds = np.concatenate([test_dyn_eyebrow_preds, test_dyn_lips_preds], axis=3)

# compute positions
dyn_test_pos = calculate_position(test_dyn_preds, mode="weighted average", return_mode="xy float flat")
print("[TEST] shape dyn_test_pos", np.shape(dyn_test_pos))
dyn_test_pos = np.multiply(dyn_test_pos, pos_weights)

# get IT responses of the model
it_test = nb_model._get_it_resp(dyn_test_pos)

# test by training new ref
nb_model._fit_reference([dyn_test_pos, data[1]], config['batch_size'])
it_ref_test = nb_model._get_it_resp(dyn_test_pos)

# # --------------------------------------------------------------------------------------------------------------------
# # plots
# ***********************       test 00 raw output      ******************
# for c2 and c3
color_seq = np.arange(len(dyn_eyebrow_preds))
color_seq[:40] = 0
color_seq[40:80] = 1
color_seq[80:190] = 0
color_seq[190:230] = 2
color_seq[230:] = 0


# plot_cnn_output(dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(dyn_eyebrow_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_dyn_eyebrow_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# plot_cnn_output(dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(dyn_lips_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_dyn_lips_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# for c2 and c3
color_seq = np.arange(len(test_dyn_eyebrow_preds))
color_seq[:40] = 0
color_seq[40:80] = 1
color_seq[80:190] = 0
color_seq[190:230] = 2
color_seq[230:] = 0
# color_seq[18:28] = 3

# # to plot bumps of c3
# color_seq = np.arange(len(test_dyn_eyebrow_preds))
# color_seq[:20] = 0
# color_seq[20:40] = 1
# color_seq[40:70] = 2
# color_seq[70:100] = 3
# color_seq[100:] = 4

# plot_cnn_output(test_dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_dyn_eyebrow_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_dyn_eyebrow_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# plot_cnn_output(test_dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_dyn_lips_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_dyn_lips_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# ***********************       test 01 eyebrow model     ******************
# plot it responses for eyebrow model
nb_model.plot_it_neurons(it_train,
                         title="01_it_train",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model.plot_it_neurons(it_test,
                         title="01_it_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model.plot_it_neurons(it_ref_test,
                         title="01_it_ref_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))

# plot it responses with reference
nb_model.plot_it_neurons_per_sequence(it_train,
                         title="02_it_train",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model.plot_it_neurons_per_sequence(it_test,
                         title="02_it_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model.plot_it_neurons_per_sequence(it_ref_test,
                         title="02_it_ref_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))

