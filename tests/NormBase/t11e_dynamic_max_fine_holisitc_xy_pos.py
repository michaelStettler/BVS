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
test script to try an implementation of a "fine"-holistic representation of the face
namely discarding part of the feature maps from their semantic labels but now I am splitting the
mask to have different zones so we don't have only a 2 x 2D vector space

run: python -m tests.NormBase.t11e_dynamic_max_fine_holisitc_xy_pos
"""

# define configuration
config_path = 'NB_t11e_dynamic_max_fine_holistic_xy_pos_m0005.json'

# declare parameters
best_eyebrow_IoU_ft = [209, 148, 59, 208]
best_lips_IoU_ft = [77, 79, 120, 104, 141, 0, 34, 125, 15, 89, 49, 237, 174, 39, 210, 112, 111, 201, 149, 165, 80,
                         42, 128, 74, 131, 193, 133, 44, 154, 101, 173, 6, 148, 61, 27, 249, 209, 19, 247, 90, 1, 255,
                         182, 251, 186, 248]

# load config
config = load_config(config_path, path='configs/norm_base_config')
config['tun_func'] = 'ft_2norm'
config["nu"] = 8

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

# # test IT computation
# x = [[3, 2, 2, 2], [2, 0, 2, 0]]
# nb_model.r = np.array([0, 0, 0, 0])
# # nb_model.t = np.array([[2/3.6, 3/3.6, 1/3.16, 3/3.16], [2/2.23, -1/2.23, 3/3.16, -1/3.16]])  # for each ft map
# nb_model.t = np.array([[2/4.8, 3/4.8, 1/4.8, 3/4.8], [2/3.87, -1/3.87, 3/3.87, -1/3.87]])   # for 4d vector
# it = nb_model._get_it_resp(x)
# print("IT")
# print(it)

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

# max activation
max_eyebrow_preds = np.expand_dims(np.amax(eyebrow_preds, axis=-1), axis=3)
max_lips_preds = np.expand_dims(np.amax(lips_preds, axis=-1), axis=3)
print("max_eyebrow_preds", np.shape(max_eyebrow_preds))
print("max_lips_preds", np.shape(max_lips_preds))

# add holistic constraints
# for eyebrow, create two eyebrow zones
left_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
left_eyebrow[:, 6:15, 8:15] = max_eyebrow_preds[:, 6:15, 8:15]
right_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
right_eyebrow[:, 6:15, 13:20] = max_eyebrow_preds[:, 6:15, 13:20]
# for lips, create three mouth zones
left_lips = np.zeros(np.shape(max_lips_preds))
left_lips[:, 16:26, 8:13] = max_lips_preds[:, 16:26, 8:13]
middle_lips = np.zeros(np.shape(max_lips_preds))
middle_lips[:, 16:26, 12:17] = max_lips_preds[:, 16:26, 12:17]
right_lips = np.zeros(np.shape(max_lips_preds))
right_lips[:, 16:26, 16:21] = max_lips_preds[:, 16:26, 16:21]
# right_lips = max_lips_preds

preds = np.concatenate([left_eyebrow, right_eyebrow, left_lips, middle_lips, right_lips], axis=3)
print("[TRAIN] preds", np.shape(preds))

# compute dynamic directly on the feature maps
# left eyebrow
dyn_left_eyebrow = ref_feature_map_neuron(left_eyebrow, data[1], config, activation='relu')
# right_eyebrow
dyn_right_eyebrow = ref_feature_map_neuron(right_eyebrow, data[1], config, activation='relu')
# lips
dyn_left_lips = ref_feature_map_neuron(left_lips, data[1], config, activation='relu')
dyn_middle_lips = ref_feature_map_neuron(middle_lips, data[1], config, activation='relu')
dyn_right_lips = ref_feature_map_neuron(right_lips, data[1], config, activation='relu')

dyn_preds = np.concatenate([dyn_left_eyebrow, dyn_right_eyebrow, dyn_left_lips, dyn_middle_lips, dyn_right_lips],
                           axis=3)
print("[TRAIN] shape dyn preds", np.shape(dyn_preds))
print("[TRAIN] finished computing dynamic predictions")
print()

# compute positions eyebrow
# dyn_pos = calculate_position(dyn_preds, mode="weighted average", return_mode="xy float flat")
dyn_pos = calculate_position(preds, mode="weighted average", return_mode="xy float flat")
# pos_weights = np.ones(np.shape(dyn_pos)[-1])
# pos_weights[:4] = 1 / (2 * 2)  # num_concept * num_ft/concept
# pos_weights[4:] = 1 / (2 * 3)
# dyn_pos = np.multiply(dyn_pos, pos_weights)

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
print("[TRAIN] shape it_train", np.shape(it_train))
print("[TRAIN] finished computing positions")
print()

# -------------------------------------------------------------------------------------------------------------------
# test monkey

# load data
test_data = load_data(config, train=False)

# # test on one frame
# idx = [0, 50, 150, 210]
# test_data[0] = test_data[0][idx]
# test_data[1] = test_data[1][idx]
# print("shape test_data[0]", np.shape(test_data[0]))
# print("shape test_data[1]", np.shape(test_data[1]))
# print(test_data[1])

# predict
test_preds = v4_model.predict(test_data[0], verbose=1)
print("[PREDS] shape test_preds", np.shape(test_preds))

# get feature maps that mimic a semantic selection pipeline
# keep only highest IoU semantic score
test_eyebrow_preds = test_preds[..., best_eyebrow_IoU_ft]
print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
test_lips_preds = test_preds[..., best_lips_IoU_ft]
print("shape lips semantic feature selection", np.shape(test_lips_preds))
test_preds = [test_eyebrow_preds, test_lips_preds]

# max activation
test_max_eyebrow_preds = np.expand_dims(np.amax(test_eyebrow_preds, axis=-1), axis=3)
test_max_lips_preds = np.expand_dims(np.amax(test_lips_preds, axis=-1), axis=3)
print("test_max_eyebrow_preds", np.shape(test_max_eyebrow_preds))
print("test_max_lips_preds", np.shape(test_max_lips_preds))

# add holistic
test_left_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
test_left_eyebrow[:, 3:12, 5:13] = test_max_eyebrow_preds[:, 3:12, 5:13]
test_right_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
test_right_eyebrow[:, 3:12, 14:22] = test_max_eyebrow_preds[:, 3:12, 14:22]
# for lips, create three mouth zones
test_left_lips = np.zeros(np.shape(test_max_lips_preds))
test_left_lips[:, 14:26, 7:12] = test_max_lips_preds[:, 14:26, 7:12]
test_middle_lips = np.zeros(np.shape(test_max_lips_preds))
test_middle_lips[:, 14:26, 11:17] = test_max_lips_preds[:, 14:26, 11:17]
test_right_lips = np.zeros(np.shape(test_max_lips_preds))
test_right_lips[:, 14:26, 16:21] = test_max_lips_preds[:, 14:26, 16:21]

test_preds = np.concatenate([test_left_eyebrow, test_right_eyebrow, test_left_lips, test_middle_lips,
                                 test_right_lips], axis=3)
print("[TEST] shape test_preds", np.shape(test_preds))

# compute dynamic feature maps
# eyebrow
test_dyn_left_eyebrow = ref_feature_map_neuron(test_left_eyebrow, test_data[1], config, activation='relu')
test_dyn_right_eyebrow = ref_feature_map_neuron(test_right_eyebrow, test_data[1], config, activation='relu')
# lips
test_dyn_left_lips = ref_feature_map_neuron(test_left_lips, test_data[1], config, activation='relu')
test_dyn_middle_lips = ref_feature_map_neuron(test_middle_lips, test_data[1], config, activation='relu')
test_dyn_right_lips = ref_feature_map_neuron(test_right_lips, test_data[1], config, activation='relu')

# concatenate concepts
test_dyn_preds = np.concatenate([test_dyn_left_eyebrow, test_dyn_right_eyebrow, test_dyn_left_lips,
                                 test_dyn_middle_lips, test_dyn_right_lips], axis=3)
print("[TEST] shape test_dyn_preds", np.shape(dyn_preds))

# compute positions
# dyn_test_pos = calculate_position(test_dyn_preds, mode="weighted average", return_mode="xy float flat")
dyn_test_pos = calculate_position(test_preds, mode="weighted average", return_mode="xy float flat")
print("[TEST] shape dyn_test_pos", np.shape(dyn_test_pos))
# dyn_test_pos = np.multiply(dyn_test_pos, pos_weights)

# get IT responses of the model
it_test = nb_model._get_it_resp(dyn_test_pos)

# test by training new ref
nb_model._fit_reference([dyn_test_pos, test_data[1]], config['batch_size'])
it_ref_test = nb_model._get_it_resp(dyn_test_pos)

# --------------------------------------------------------------------------------------------------------------------
# plots
# ***********************       test 00 raw output      ******************

# # raw activity
# plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
#                 "00_max_feature_maps_output.gif", verbose=True, video=True)
# plot_cnn_output(dyn_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_dyn_max_feature_maps_output.gif", verbose=True, video=True)
# plot_cnn_output(test_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_test_max_maps_output.gif", verbose=True, video=True)
# plot_cnn_output(test_dyn_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_test_dyn_max_feature_maps_output.gif", verbose=True, video=True)

# # for only c2
# color_seq = np.arange(len(dyn_eyebrow_preds))
# color_seq[:40] = 0
# color_seq[40:80] = 1
# color_seq[80:] = 0

# for c2 and c3
color_seq = np.arange(len(preds))
color_seq[:40] = 0
color_seq[40:80] = 1
color_seq[80:190] = 0
color_seq[190:230] = 2
color_seq[230:] = 0
# color_seq[18:28] = 3

print("[PLOT] shape dyn_preds", np.shape(preds))
#
# plot_cnn_output(dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)
#
# plot_cnn_output(dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(dyn_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_dyn_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# # c2
# color_seq = np.arange(len(test_dyn_eyebrow_preds))
# color_seq[:40] = 0
# color_seq[40:80] = 1
# color_seq[80:] = 0

# for c2 and c3
color_seq = np.arange(len(test_preds))
color_seq[:40] = 0
color_seq[40:80] = 1
color_seq[80:190] = 0
color_seq[190:230] = 2
color_seq[230:] = 0
# color_seq[18:28] = 3

# color_seq[0] = 0
# color_seq[1] = 1
# color_seq[2] = 0
# color_seq[3] = 2

# # c3 ears
# color_seq = np.arange(len(test_dyn_eyebrow_preds))
# color_seq[:30] = 0
# color_seq[30:45] = 1
# color_seq[45:70] = 2
# color_seq[70:90] = 3
# color_seq[90:] = 4

# plot_cnn_output(test_dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# plot_cnn_output(test_dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_dyn_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_dyn_pos.png",
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

# plot it responses for eyebrow model
nb_model.plot_it_neurons_per_sequence(it_train,
                         title="02_it_train",
                         save_folder=os.path.join("models/saved", config["config_name"]))
# config['seq_length'] = 2
nb_model.plot_it_neurons_per_sequence(it_test,
                         title="02_it_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model.plot_it_neurons_per_sequence(it_ref_test,
                         title="02_it_ref_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))

