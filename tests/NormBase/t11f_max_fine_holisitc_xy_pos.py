import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.ref_feature_map_neurons import ref_feature_map_neuron
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output
from plots_utils.plot_ft_map_pos import plot_ft_map_pos
from plots_utils.plot_ft_map_pos import plot_ft_pos_on_sequence
from models.NormBase import NormBase

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try an implementation of a "fine"-holistic representation of the face
namely discarding part of the feature maps from their semantic labels but now I am splitting the
mask to have different zones so we don't have only a 2 x 2D vector space
I am also getting to the block 3 feature maps since the pooling effect seems that expression c4 is too sublte

run: python -m tests.NormBase.t11f_max_fine_holisitc_xy_pos
"""

# define configuration
config_path = 'NB_t11f_max_fine_holistic_xy_pos_m0006.json'

# declare parameters
best_eyebrow_IoU_ft = [68, 125]
best_lips_IoU_ft = [235, 203, 68, 125, 3, 181, 197, 2, 87, 240, 6, 95, 60, 157, 227, 111]

# load config
config = load_config(config_path, path='configs/norm_base_config')
config['tun_func'] = 'ft_2norm'

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
# for eyebrow, create four eyebrow zones
left_ext_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
left_ext_eyebrow[:, 16:21, 15:20] = max_eyebrow_preds[:, 16:21, 15:20]
left_int_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
left_int_eyebrow[:, 16:21, 20:25] = max_eyebrow_preds[:, 16:21, 20:25]
right_int_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
right_int_eyebrow[:, 16:21, 29:35] = max_eyebrow_preds[:, 16:21, 29:35]
right_ext_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
right_ext_eyebrow[:, 16:21, 36:41] = max_eyebrow_preds[:, 16:21, 36:41]
# for lips, create four mouth zones
left_lips = np.zeros(np.shape(max_lips_preds))
left_lips[:, 33:45, 19:26] = max_lips_preds[:, 33:45, 19:26]
middle_up_lips = np.zeros(np.shape(max_lips_preds))
middle_up_lips[:, 33:37, 24:34] = max_lips_preds[:, 33:37, 24:34]
middle_down_lips = np.zeros(np.shape(max_lips_preds))
middle_down_lips[:, 38:48, 24:33] = max_lips_preds[:, 38:48, 24:33]
right_lips = np.zeros(np.shape(max_lips_preds))
right_lips[:, 33:45, 32:37] = max_lips_preds[:, 33:45, 32:37]
# right_lips = max_lips_preds

preds = np.concatenate([left_ext_eyebrow, left_int_eyebrow, right_int_eyebrow, right_ext_eyebrow,
                        middle_up_lips, middle_down_lips, left_lips, right_lips], axis=3)
print("[TRAIN] preds", np.shape(preds))

# compute positions eyebrow
pos = calculate_position(preds, mode="weighted average", return_mode="xy float flat")
print("[TRAIN] shape pos", np.shape(pos))

nb_model.n_features = np.shape(pos)[-1]  # todo add this to init
# train manually ref vector
nb_model.r = np.zeros(nb_model.n_features)
nb_model._fit_reference([pos, data[1]], config['batch_size'])
# train manually tuning vector
nb_model.t = np.zeros((nb_model.n_category, nb_model.n_features))
nb_model.t_mean = np.zeros((nb_model.n_category, nb_model.n_features))
nb_model._fit_tuning([pos, data[1]], config['batch_size'])
# get it resp for eyebrows
it_train = nb_model._get_it_resp(pos)
print("[TRAIN] shape it_train", np.shape(it_train))

ds_train = nb_model._get_decisions_neurons(it_train, config['seq_length'])
print("[TRAIN] shape ds_train", np.shape(ds_train))
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
# create four eyebrow zones
test_left_ext_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
test_left_ext_eyebrow[:, 10:14, 17:20] = test_max_eyebrow_preds[:, 10:14, 17:20]
test_left_int_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
test_left_int_eyebrow[:, 10:14, 20:23] = test_max_eyebrow_preds[:, 10:14, 20:23]
test_right_int_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
test_right_int_eyebrow[:, 10:14, 32:35] = test_max_eyebrow_preds[:, 10:14, 32:35]
test_right_ext_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
test_right_ext_eyebrow[:, 10:14, 35:39] = test_max_eyebrow_preds[:, 10:14, 35:39]
# for lips, create four mouth zones
test_left_lips = np.zeros(np.shape(test_max_lips_preds))
test_left_lips[:, 30:41, 16:24] = test_max_lips_preds[:, 30:41, 16:24]
test_middle_up_lips = np.zeros(np.shape(test_max_lips_preds))
test_middle_up_lips[:, 30:34, 24:35] = test_max_lips_preds[:, 30:34, 24:35]
test_middle_down_lips = np.zeros(np.shape(test_max_lips_preds))
test_middle_down_lips[:, 36:50, 21:33] = test_max_lips_preds[:, 36:50, 21:33]
test_right_lips = np.zeros(np.shape(test_max_lips_preds))
test_right_lips[:, 30:41, 33:40] = test_max_lips_preds[:, 30:41, 33:40]

test_preds = np.concatenate([test_left_ext_eyebrow, test_left_int_eyebrow, test_right_int_eyebrow, test_right_ext_eyebrow,
                             test_middle_up_lips, test_middle_down_lips, test_left_lips, test_right_lips], axis=3)
print("[TEST] shape test_preds", np.shape(test_preds))

# compute positions
test_pos = calculate_position(test_preds, mode="weighted average", return_mode="xy float flat")
print("[TEST] shape test_pos", np.shape(test_pos))

# get IT responses of the model
it_test = nb_model._get_it_resp(test_pos)

# test by training new ref
nb_model._fit_reference([test_pos, test_data[1]], config['batch_size'])
it_ref_test = nb_model._get_it_resp(test_pos)

ds_test = nb_model._get_decisions_neurons(it_ref_test, config['seq_length'])
print("[TEST] shape ds_test", np.shape(ds_test))

# --------------------------------------------------------------------------------------------------------------------
# plots
# ***********************       test 00 raw output      ******************

# raw activity
# plot_cnn_output(max_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_max_feature_maps_eyebrow_output.gif", verbose=True, video=True)
# plot_cnn_output(max_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_max_feature_maps_lips_output.gif", verbose=True, video=True)
# plot_cnn_output(test_max_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_test_max_feature_maps_eyebrow_output.gif", verbose=True, video=True)
# plot_cnn_output(test_max_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_test_max_feature_maps_lips_output.gif", verbose=True, video=True)

# plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
#                 "00a_max_feature_maps_output.gif", verbose=True, video=True)
# plot_cnn_output(test_preds, os.path.join("models/saved", config["config_name"]),
#                 "00a_test_max_maps_output.gif", verbose=True, video=True)

# for c2 and c3
color_seq = np.arange(len(preds))
color_seq[:40] = 0
color_seq[40:80] = 1
color_seq[80:190] = 0
color_seq[190:230] = 2
color_seq[230:320] = 0
color_seq[320:390] = 3
color_seq[360:490] = 0
color_seq[490:540] = 4
color_seq[540:] = 0
# color_seq[18:28] = 3

print("[PLOT] shape dyn_preds", np.shape(preds))
plot_ft_map_pos(calculate_position(preds, mode="weighted average", return_mode="xy float"),
                fig_name="00b_human_train_pos.png",
                path=os.path.join("models/saved", config["config_name"]),
                color_seq=color_seq)

# for c2 and c3
color_seq = np.arange(len(test_preds))
color_seq[:40] = 0
color_seq[40:80] = 1
color_seq[80:190] = 0
color_seq[190:230] = 2
color_seq[230:320] = 0
color_seq[320:390] = 3
color_seq[360:490] = 0
color_seq[490:540] = 4
color_seq[540:] = 0

plot_ft_map_pos(calculate_position(test_preds, mode="weighted average", return_mode="xy float"),
                fig_name="00b_monkey_test_pos.png",
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
nb_model.plot_it_neurons_per_sequence(it_test,
                         title="02_it_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))
nb_model.plot_it_neurons_per_sequence(it_ref_test,
                         title="02_it_ref_test",
                         save_folder=os.path.join("models/saved", config["config_name"]))

print()
# plot tracked vector on sequence
plot_ft_pos_on_sequence(pos, data[0],
                        save_folder=os.path.join("models/saved", config["config_name"]),
                        lmk_size=1, ft_size=(56, 56))
print()

max_eyebrow_preds_plot = max_eyebrow_preds / np.amax(max_eyebrow_preds) * 255
# pos = np.zeros(np.shape(pos))
# print("shape pos", np.shape(pos))
# plot_ft_pos_on_sequence(pos, max_eyebrow_preds_plot, vid_name='eyebrow_ft.mp4',
#                         save_folder=os.path.join("models/saved", config["config_name"]),
#                         pre_proc='raw', ft_size=(56, 56))


nb_model.plot_decision_neurons(ds_train,
                               title="03_ds_train",
                               save_folder=os.path.join("models/saved", config["config_name"]),
                               normalize=True)
nb_model.plot_decision_neurons(ds_test,
                               title="03_ds_test",
                               save_folder=os.path.join("models/saved", config["config_name"]),
                               normalize=True)