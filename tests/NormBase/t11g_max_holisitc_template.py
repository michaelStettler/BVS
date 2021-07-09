import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.remove_transition_morph_space import remove_transition_frames
from utils.PatternFeatureReduction import PatternFeatureSelection
from utils.ref_feature_map_neurons import ref_feature_map_neuron
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output
from plots_utils.plot_ft_map_pos import plot_ft_map_pos
from plots_utils.plot_ft_map_pos import plot_ft_pos_on_sequence
from models.NormBase import NormBase

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try an implementation of a holistic representation model by a RBF function of the face


run: python -m tests.NormBase.t11g_max_holisitc_template
"""

# define configuration
config_path = 'NB_t11g_max_holistic_template_m0001.json'

# declare parameters
best_eyebrow_IoU_ft = [68, 125]
best_lips_IoU_ft = [235, 203, 68, 125, 3, 181, 197, 2, 87, 240, 6, 95, 60, 157, 227, 111]

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

# remove transition frames
data = remove_transition_frames(data)

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
# preds = np.concatenate((max_eyebrow_preds, max_lips_preds), axis=3)
preds = np.concatenate((eyebrow_preds, lips_preds), axis=3)
print("shape preds", np.shape(preds))

# add holistic templates
mask = [[[16, 21], [16, 21]], [[15, 20], [21, 26]], [[16, 21], [30, 35]], [[16, 21], [37, 42]]]
patterns = PatternFeatureSelection(config, mask=mask)  # 3x3  eyebrow
mask_template = np.repeat(np.expand_dims(preds, axis=0), len(mask), axis=0)
print("shape mask_template", np.shape(mask_template))
eyebrow_template = patterns.fit(mask_template)
print("shape eyebrow_template", np.shape(eyebrow_template))
eyebrow_template[eyebrow_template < 0.1] = 0

plot_cnn_output(eyebrow_template, os.path.join("models/saved", config["config_name"]),
                "00_eyebrow_template.gif", verbose=True, video=True)

pos = calculate_position(eyebrow_template, mode="weighted average", return_mode="xy float flat")

pos_2d = np.reshape(pos, (len(pos), -1, 2))
plot_ft_map_pos(pos_2d,
                fig_name="00b_human_train_pos.png",
                path=os.path.join("models/saved", config["config_name"]))


preds_plot = max_eyebrow_preds / np.amax(max_eyebrow_preds) * 255
print("shape preds_plot", np.shape(preds_plot))
plot_ft_pos_on_sequence(pos, preds_plot, vid_name='00_ft_eyebrow_pos.mp4',
                        save_folder=os.path.join("models/saved", config["config_name"]),
                        pre_proc='raw', ft_size=(56, 56))


plot_ft_pos_on_sequence(pos, data[0],
                        vid_name='00_ft_pos_human.mp4',
                        save_folder=os.path.join("models/saved", config["config_name"]),
                        lmk_size=1, ft_size=(56, 56))



# # for eyebrow, create four eyebrow zones
# left_ext_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# left_ext_eyebrow[:, 16:21, 15:20] = max_eyebrow_preds[:, 16:21, 15:20]
# left_int_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# left_int_eyebrow[:, 16:21, 20:25] = max_eyebrow_preds[:, 16:21, 20:25]
# right_int_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# right_int_eyebrow[:, 16:21, 29:35] = max_eyebrow_preds[:, 16:21, 29:35]
# right_ext_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# right_ext_eyebrow[:, 16:21, 36:41] = max_eyebrow_preds[:, 16:21, 36:41]
# # for lips, create four mouth zones
# left_lips = np.zeros(np.shape(max_lips_preds))
# left_lips[:, 33:45, 19:26] = max_lips_preds[:, 33:45, 19:26]
# middle_up_lips = np.zeros(np.shape(max_lips_preds))
# middle_up_lips[:, 33:37, 24:34] = max_lips_preds[:, 33:37, 24:34]
# middle_down_lips = np.zeros(np.shape(max_lips_preds))
# middle_down_lips[:, 38:48, 24:33] = max_lips_preds[:, 38:48, 24:33]
# right_lips = np.zeros(np.shape(max_lips_preds))
# right_lips[:, 33:45, 32:37] = max_lips_preds[:, 33:45, 32:37]
#
# preds = np.concatenate([left_ext_eyebrow, left_int_eyebrow, right_int_eyebrow, right_ext_eyebrow,
#                         middle_up_lips, middle_down_lips, left_lips, right_lips], axis=3)



# print("[TRAIN] preds", np.shape(preds))
#
# # compute positions eyebrow
# pos = calculate_position(preds, mode="weighted average", return_mode="xy float flat")
# print("[TRAIN] shape pos", np.shape(pos))
#
# nb_model.n_features = np.shape(pos)[-1]  # todo add this to init
# # train manually ref vector
# nb_model.r = np.zeros(nb_model.n_features)
# nb_model._fit_reference([pos, data[1]], config['batch_size'])
# print("[TRAIN] model.r", np.shape(nb_model.r))
# print(nb_model.r)
# ref_train = np.copy(nb_model.r)
# # train manually tuning vector
# nb_model.t = np.zeros((nb_model.n_category, nb_model.n_features))
# nb_model.t_mean = np.zeros((nb_model.n_category, nb_model.n_features))
# nb_model._fit_tuning([pos, data[1]], config['batch_size'])
# ref_tuning = np.copy(nb_model.t)
# print("[TRAIN] ref_tuning[1]")
# print(ref_tuning[1])
# # get it resp
# it_train = nb_model._get_it_resp(pos)
# print("[TRAIN] shape it_train", np.shape(it_train))
#
# # ds_train = nb_model._get_decisions_neurons(it_train, config['seq_length'])
# # print("[TRAIN] shape ds_train", np.shape(ds_train))
# # print()
#
# # -------------------------------------------------------------------------------------------------------------------
# # test Full human
# # load data
# data = load_data(config)
# # predict
# preds = v4_model.predict(data[0], verbose=1)
# print("[PRED] shape prediction", np.shape(preds))
#
# # get feature maps that mimic a semantic selection pipeline
# # keep only highest IoU semantic score
# eyebrow_preds = preds[..., best_eyebrow_IoU_ft]
# print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
# lips_preds = preds[..., best_lips_IoU_ft]
# print("shape lips semantic feature selection", np.shape(lips_preds))
#
# # max activation
# max_eyebrow_preds = np.expand_dims(np.amax(eyebrow_preds, axis=-1), axis=3)
# max_lips_preds = np.expand_dims(np.amax(lips_preds, axis=-1), axis=3)
# print("max_eyebrow_preds", np.shape(max_eyebrow_preds))
# print("max_lips_preds", np.shape(max_lips_preds))
#
# # add holistic constraints
# # for eyebrow, create four eyebrow zones
# left_ext_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# left_ext_eyebrow[:, 16:21, 15:20] = max_eyebrow_preds[:, 16:21, 15:20]
# left_int_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# left_int_eyebrow[:, 16:21, 20:25] = max_eyebrow_preds[:, 16:21, 20:25]
# right_int_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# right_int_eyebrow[:, 16:21, 29:35] = max_eyebrow_preds[:, 16:21, 29:35]
# right_ext_eyebrow = np.zeros(np.shape(max_eyebrow_preds))
# right_ext_eyebrow[:, 16:21, 36:41] = max_eyebrow_preds[:, 16:21, 36:41]
# # for lips, create four mouth zones
# left_lips = np.zeros(np.shape(max_lips_preds))
# left_lips[:, 33:45, 19:26] = max_lips_preds[:, 33:45, 19:26]
# middle_up_lips = np.zeros(np.shape(max_lips_preds))
# middle_up_lips[:, 33:37, 24:34] = max_lips_preds[:, 33:37, 24:34]
# middle_down_lips = np.zeros(np.shape(max_lips_preds))
# middle_down_lips[:, 38:48, 24:33] = max_lips_preds[:, 38:48, 24:33]
# right_lips = np.zeros(np.shape(max_lips_preds))
# right_lips[:, 33:45, 32:37] = max_lips_preds[:, 33:45, 32:37]
#
# preds = np.concatenate([left_ext_eyebrow, left_int_eyebrow, right_int_eyebrow, right_ext_eyebrow,
#                         middle_up_lips, middle_down_lips, left_lips, right_lips], axis=3)
# print("[PRED] preds", np.shape(preds))
#
# # compute positions eyebrow
# pos = calculate_position(preds, mode="weighted average", return_mode="xy float flat")
# print("[PRED] shape pos", np.shape(pos))
# # get it resp for eyebrows
# it_train = nb_model._get_it_resp(pos)
# print("[PRED] shape it_train", np.shape(it_train))
#
# # ds_train = nb_model._get_decisions_neurons(it_train, config['seq_length'])
# # print("[PRED] shape ds_train", np.shape(ds_train))
# # print()
#
# # -------------------------------------------------------------------------------------------------------------------
# # test monkey
#
# # load data
# test_data = load_data(config, train=False)
#
# # # test on one frame
# # idx = [0, 50, 150, 210]
# # test_data[0] = test_data[0][idx]
# # test_data[1] = test_data[1][idx]
# # print("shape test_data[0]", np.shape(test_data[0]))
# # print("shape test_data[1]", np.shape(test_data[1]))
# # print(test_data[1])
#
# # predict
# test_preds = v4_model.predict(test_data[0], verbose=1)
# print("[PREDS] shape test_preds", np.shape(test_preds))
#
# # get feature maps that mimic a semantic selection pipeline
# # keep only highest IoU semantic score
# test_eyebrow_preds = test_preds[..., best_eyebrow_IoU_ft]
# print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
# test_lips_preds = test_preds[..., best_lips_IoU_ft]
# print("shape lips semantic feature selection", np.shape(test_lips_preds))
# test_preds = [test_eyebrow_preds, test_lips_preds]
#
# # max activation
# test_max_eyebrow_preds = np.expand_dims(np.amax(test_eyebrow_preds, axis=-1), axis=3)
# test_max_lips_preds = np.expand_dims(np.amax(test_lips_preds, axis=-1), axis=3)
# print("test_max_eyebrow_preds", np.shape(test_max_eyebrow_preds))
# print("test_max_lips_preds", np.shape(test_max_lips_preds))
#
# # add holistic
# # create four eyebrow zones
# test_left_ext_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
# test_left_ext_eyebrow[:, 10:14, 17:20] = test_max_eyebrow_preds[:, 10:14, 17:20]
# test_left_int_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
# test_left_int_eyebrow[:, 10:14, 20:23] = test_max_eyebrow_preds[:, 10:14, 20:23]
# test_right_int_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
# test_right_int_eyebrow[:, 10:14, 32:35] = test_max_eyebrow_preds[:, 10:14, 32:35]
# test_right_ext_eyebrow = np.zeros(np.shape(test_max_eyebrow_preds))
# test_right_ext_eyebrow[:, 10:14, 35:39] = test_max_eyebrow_preds[:, 10:14, 35:39]
# # for lips, create four mouth zones
# test_left_lips = np.zeros(np.shape(test_max_lips_preds))
# test_left_lips[:, 30:41, 16:24] = test_max_lips_preds[:, 30:41, 16:24]
# test_middle_up_lips = np.zeros(np.shape(test_max_lips_preds))
# test_middle_up_lips[:, 30:34, 24:35] = test_max_lips_preds[:, 30:34, 24:35]
# test_middle_down_lips = np.zeros(np.shape(test_max_lips_preds))
# test_middle_down_lips[:, 36:50, 21:33] = test_max_lips_preds[:, 36:50, 21:33]
# test_right_lips = np.zeros(np.shape(test_max_lips_preds))
# test_right_lips[:, 30:41, 33:40] = test_max_lips_preds[:, 30:41, 33:40]
#
# test_preds = np.concatenate([test_left_ext_eyebrow, test_left_int_eyebrow, test_right_int_eyebrow, test_right_ext_eyebrow,
#                              test_middle_up_lips, test_middle_down_lips, test_left_lips, test_right_lips], axis=3)
# print("[TEST] shape test_preds", np.shape(test_preds))
#
# # compute positions
# test_pos = calculate_position(test_preds, mode="weighted average", return_mode="xy float flat")
# print("[TEST] shape test_pos", np.shape(test_pos))
#
# # get IT responses of the model
# it_test = nb_model._get_it_resp(test_pos)
#
# # test by training new ref
# nb_model._fit_reference([test_pos, test_data[1]], config['batch_size'])
# ref_test = np.copy(nb_model.r)
# it_ref_test = nb_model._get_it_resp(test_pos)
#
# # ds_test = nb_model._get_decisions_neurons(it_ref_test, config['seq_length'])
# # print("[TEST] shape ds_test", np.shape(ds_test))
#
# # --------------------------------------------------------------------------------------------------------------------
# # plots
# # ***********************       test 00 raw output      ******************
#
# # raw activity
# plot_cnn_output(max_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_max_feature_maps_eyebrow_output.gif", verbose=True, video=True)
# plot_cnn_output(max_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_max_feature_maps_lips_output.gif", verbose=True, video=True)
# plot_cnn_output(test_max_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_test_max_feature_maps_eyebrow_output.gif", verbose=True, video=True)
# plot_cnn_output(test_max_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_test_max_feature_maps_lips_output.gif", verbose=True, video=True)
#
# plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
#                 "00a_max_feature_maps_output.gif", verbose=True, video=True)
# plot_cnn_output(test_preds, os.path.join("models/saved", config["config_name"]),
#                 "00a_test_max_maps_output.gif", verbose=True, video=True)
#
# # build arrows
# arrow_tail = np.repeat(np.expand_dims(np.reshape(ref_train, (-1, 2)), axis=0), config['n_category'], axis=0)
# arrow_head = np.reshape(ref_tuning, (len(ref_tuning), -1, 2))
# arrows = [arrow_tail, arrow_head]
# arrows_color = ['#0e3957', '#3b528b', '#21918c', '#5ec962', '#fde725']
#
# # put one color per label
# labels = data[1]
# color_seq = np.zeros(len(preds))
# color_seq[labels == 1] = 1
# color_seq[labels == 2] = 2
# color_seq[labels == 3] = 3
# color_seq[labels == 4] = 4
#
# print("[PLOT] shape preds", np.shape(preds))
# pos_2d = np.reshape(pos, (len(pos), -1, 2))
# print("[PLOT] shape pos_flat", np.shape(pos_2d))
# plot_ft_map_pos(pos_2d,
#                 fig_name="00b_human_train_pos.png",
#                 path=os.path.join("models/saved", config["config_name"]),
#                 color_seq=color_seq,
#                 arrows=arrows,
#                 arrows_color=arrows_color)
#
# # build arrows
# arrow_tail = np.repeat(np.expand_dims(np.reshape(ref_test, (-1, 2)), axis=0), config['n_category'], axis=0)
# arrow_head = np.reshape(ref_tuning, (len(ref_tuning), -1, 2))
# arrows = [arrow_tail, arrow_head]
# arrows_color = ['#0e3957', '#3b528b', '#21918c', '#5ec962', '#fde725']
#
# # put one color per label
# test_labels = test_data[1]
# color_seq = np.zeros(len(test_labels))
# color_seq[test_labels == 1] = 1
# color_seq[test_labels == 2] = 2
# color_seq[test_labels == 3] = 3
# color_seq[test_labels == 4] = 4
#
# test_pos_2d = np.reshape(test_pos, (len(test_pos), -1, 2))
# plot_ft_map_pos(test_pos_2d,
#                 fig_name="00b_monkey_test_pos.png",
#                 path=os.path.join("models/saved", config["config_name"]),
#                 color_seq=color_seq,
#                 arrows=arrows,
#                 arrows_color=arrows_color)
#
# # ***********************       test 01 model     ******************
# # plot it responses for eyebrow model
# nb_model.plot_it_neurons(it_train,
#                          title="01_it_train",
#                          save_folder=os.path.join("models/saved", config["config_name"]))
# # nb_model.plot_it_neurons(it_test,
# #                          title="01_it_test",
# #                          save_folder=os.path.join("models/saved", config["config_name"]))
# nb_model.plot_it_neurons(it_ref_test,
#                          title="01_it_ref_test",
#                          save_folder=os.path.join("models/saved", config["config_name"]))
#
# # ***********************       test 02 model     ******************
# # plot it responses for eyebrow model
# nb_model.plot_it_neurons_per_sequence(it_train,
#                          title="02_it_train",
#                          save_folder=os.path.join("models/saved", config["config_name"]))
# # nb_model.plot_it_neurons_per_sequence(it_test,
# #                          title="02_it_test",
# #                          save_folder=os.path.join("models/saved", config["config_name"]))
# nb_model.plot_it_neurons_per_sequence(it_ref_test,
#                          title="02_it_ref_test",
#                          save_folder=os.path.join("models/saved", config["config_name"]))
#
# print()
# # plot tracked vector on sequence
# plot_ft_pos_on_sequence(pos, data[0],
#                         vid_name='03_ft_pos_human.mp4',
#                         save_folder=os.path.join("models/saved", config["config_name"]),
#                         lmk_size=1, ft_size=(56, 56))
#
# # plot tracked vector on sequence
# plot_ft_pos_on_sequence(test_pos, test_data[0],
#                         vid_name='03_ft_pos_monkey.mp4',
#                         save_folder=os.path.join("models/saved", config["config_name"]),
#                         lmk_size=1, ft_size=(56, 56))
# print()
#
# # plot tracked vector on feature maps
# max_eyebrow_preds_plot = max_eyebrow_preds / np.amax(max_eyebrow_preds) * 255
# plot_ft_pos_on_sequence(pos[:, :8], max_eyebrow_preds_plot, vid_name='03_ft_pos_eyebrow_human.mp4',
#                         save_folder=os.path.join("models/saved", config["config_name"]),
#                         pre_proc='raw', ft_size=(56, 56))
# max_lips_preds_plot = max_lips_preds / np.amax(max_lips_preds) * 255
# plot_ft_pos_on_sequence(pos[:, 8:], max_lips_preds_plot, vid_name='03_ft_pos_lips_human.mp4',
#                         save_folder=os.path.join("models/saved", config["config_name"]),
#                         pre_proc='raw', ft_size=(56, 56))
# # monkey
# test_max_eyebrow_preds_plot = test_max_eyebrow_preds / np.amax(test_max_eyebrow_preds) * 255
# plot_ft_pos_on_sequence(test_pos[:, :8], test_max_eyebrow_preds_plot, vid_name='03_ft_pos_eyebrow_monkey.mp4',
#                         save_folder=os.path.join("models/saved", config["config_name"]),
#                         pre_proc='raw', ft_size=(56, 56))
# test_max_lips_preds_plot = test_max_lips_preds / np.amax(test_max_lips_preds) * 255
# plot_ft_pos_on_sequence(test_pos[:, 8:], test_max_lips_preds_plot, vid_name='03_ft_pos_lips_monkey.mp4',
#                         save_folder=os.path.join("models/saved", config["config_name"]),
#                         pre_proc='raw', ft_size=(56, 56))
#
#
# # ***********************       test 04 decision neuron     ******************
# # nb_model.plot_decision_neurons(ds_train,
# #                                title="04_ds_train",
# #                                save_folder=os.path.join("models/saved", config["config_name"]),
# #                                normalize=True)
# # nb_model.plot_decision_neurons(ds_test,
# #                                title="04_ds_test",
# #                                save_folder=os.path.join("models/saved", config["config_name"]),
# #                                normalize=True)
