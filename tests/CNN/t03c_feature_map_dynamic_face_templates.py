import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.PatternFeatureReduction import PatternFeatureSelection
from plots_utils.plot_cnn_output import plot_cnn_output
from utils.calculate_position import calculate_position
import matplotlib.pyplot as plt
from plots_utils.plot_ft_map_pos import plot_ft_map_pos

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the fit a face template over different feature maps instead of simply summing them up
the term dynmic means here that I am trying to get the specific "moving parts" from the sequence 

run: python -m tests.CNN.t03c_feature_map_dynamic_face_templates
"""

# define configuration
config_path = 'CNN_t03c_feature_map_dynamic_face_templates_m0002.json'

# declare parameters
best_eyebrow_IoU_ft = [209, 148, 59, 208]
best_lips_IoU_ft = [77, 79, 120, 104, 141, 0, 34, 125, 15, 89, 49, 237, 174, 39, 210, 112, 111, 201, 149, 165, 80,
                         42, 128, 74, 131, 193, 133, 44, 154, 101, 173, 6, 148, 61, 27, 249, 209, 19, 247, 90, 1, 255,
                         182, 251, 186, 248]

# load config
config = load_config(config_path, path='configs/CNN')

# create directory if non existant
save_path = os.path.join("models/saved", config["config_name"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

# load and define model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(model.output)[1:3])
print("[LOAD] size_ft", size_ft)
print("[LOAD] Model loaded")
print()

# -------------------------------------------------------------------------------------------------------------------
# train

# load data
data = load_data(config)

# predict
preds = model.predict(data[0], verbose=1)
print("[PREDS] shape prediction", np.shape(preds))

# get feature maps that mimic a semantic selection pipeline
# keep only highest IoU semantic score
eyebrow_preds = preds[..., best_eyebrow_IoU_ft]
print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
lips_preds = preds[..., best_lips_IoU_ft]
print("shape lips semantic feature selection", np.shape(lips_preds))
preds = [eyebrow_preds, lips_preds]

# compute dynamic directly on the feature maps
eyebrow_preds_ref = eyebrow_preds[0]
dyn_eyebrow_preds = eyebrow_preds - np.repeat(np.expand_dims(eyebrow_preds_ref, axis=0), len(eyebrow_preds), axis=0)
dyn_eyebrow_preds[dyn_eyebrow_preds < 0] = 0
print("shape dyn_eyebrow_preds", np.shape(dyn_eyebrow_preds))
lips_preds_ref = lips_preds[0]
dyn_lips_preds = lips_preds - np.repeat(np.expand_dims(lips_preds_ref, axis=0), len(lips_preds), axis=0)
dyn_lips_preds[dyn_lips_preds < 0] = 0
print("shape dyn_lips_preds", np.shape(dyn_lips_preds))
dyn_preds = [dyn_eyebrow_preds, dyn_lips_preds]

# # -------------------------------------------------------------------------------------------------------------------
# # fit face template
#
# # build template
# # todo set within config
# eyebrow_mask = [[7, 15], [7, 21]]
# lips_mask = [[18, 22], [10, 19]]
# config['pattern_mask'] = [eyebrow_mask, lips_mask]
# config['rbf_sigma'] = [9000, 15000]
# config['pattern_idx'] = 0
#
# # set reference frame
# patternFS = PatternFeatureSelection(config)
# dyn_preds = patternFS.fit(dyn_preds)
# print("shape dyn_preds", np.shape(dyn_preds))
# print("max dyn_preds", np.amax(dyn_preds))
#
# # predict template on normal prediction
# preds = patternFS.transform(preds)
# eyebrow_ft = np.expand_dims(preds[..., 0], axis=3)
# lips_ft = np.expand_dims(preds[..., 1], axis=3)
# print("shape preds", np.shape(preds))
# print("shape eyebrow_ft", np.shape(eyebrow_ft))
# print("max eyebrow_ft", np.amax(eyebrow_ft))
# print("shape lips_ft", np.shape(lips_ft))
# print()


# -------------------------------------------------------------------------------------------------------------------
# test monkey

# load data
test_data = load_data(config, train=False)
# predict
test_preds = model.predict(test_data[0], verbose=1)
print("[PREDS] shape test_preds", np.shape(test_preds))

# get feature maps that mimic a semantic selection pipeline
# keep only highest IoU semantic score
test_eyebrow_preds = test_preds[..., best_eyebrow_IoU_ft]
print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
test_lips_preds = test_preds[..., best_lips_IoU_ft]
print("shape lips semantic feature selection", np.shape(lips_preds))
test_preds = [test_eyebrow_preds, test_lips_preds]

# compute dynamic feature maps
test_eyebrow_preds_ref = test_eyebrow_preds[0]
test_dyn_eyebrow_preds = test_eyebrow_preds - np.repeat(np.expand_dims(test_eyebrow_preds_ref, axis=0), len(test_eyebrow_preds), axis=0)
test_dyn_eyebrow_preds[test_dyn_eyebrow_preds < 0] = 0
test_lips_preds_ref = test_lips_preds[0]
test_dyn_lips_preds = test_lips_preds - np.repeat(np.expand_dims(test_lips_preds_ref, axis=0), len(test_lips_preds), axis=0)
test_dyn_lips_preds[test_dyn_lips_preds < 0] = 0
test_dyn_preds = [test_dyn_eyebrow_preds, test_dyn_lips_preds]

# # predict pattern
# test_preds = patternFS.transform(test_preds)
# test_eyebrow_ft = np.expand_dims(test_preds[..., 0], axis=3)
# test_lips_ft = np.expand_dims(test_preds[..., 1], axis=3)
# print("shape test_preds", np.shape(test_preds))
# print("shape test_eyebrow_ft", np.shape(test_eyebrow_ft))
# print("shape test_lips_ft", np.shape(test_lips_ft))
# print()

# # --------------------------------------------------------------------------------------------------------------------
# # create test to compare with the average
#
# # train
# # compute average eyebrow feature maps
# eyebrow_ft_average = np.mean(eyebrow_preds, axis=-1)
# eyebrow_ft_average = np.expand_dims(eyebrow_ft_average, axis=3)
# # compute average lips feature maps
# lips_ft_average = np.mean(lips_preds, axis=-1)
# lips_ft_average = np.expand_dims(lips_ft_average, axis=3)
#
# # test
# # compute average test eyebrow feature maps
# test_eyebrow_ft_average = np.mean(test_eyebrow_preds, axis=-1)
# test_eyebrow_ft_average = np.expand_dims(test_eyebrow_ft_average, axis=3)
# # compute average lips feature maps
# test_lips_ft_average = np.mean(test_lips_preds, axis=-1)
# test_lips_ft_average = np.expand_dims(test_lips_ft_average, axis=3)

# # --------------------------------------------------------------------------------------------------------------------
# # plots
# plot_cnn_output(dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(dyn_eyebrow_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_dyn_eyebrow_pos.png",
                path=os.path.join("models/saved", config["config_name"]))
# plot_cnn_output(test_dyn_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_eyebrow_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_dyn_eyebrow_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_dyn_eyebrow_pos.png",
                path=os.path.join("models/saved", config["config_name"]))
# plot_cnn_output(dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_human_train_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(dyn_lips_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_human_train_dyn_lips_pos.png",
                path=os.path.join("models/saved", config["config_name"]))
# plot_cnn_output(test_dyn_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "00_monkey_test_dyn_lips_feature_maps_output.gif", verbose=True, video=True)
plot_ft_map_pos(calculate_position(test_dyn_lips_preds[1:], mode="weighted average", return_mode="xy float"),
                fig_name="00_monkey_test_dyn_lips_pos.png",
                path=os.path.join("models/saved", config["config_name"]))
#
# # plot feature maps for each concept
# plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
#                 "01_human_train_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# plot_cnn_output(test_preds, os.path.join("models/saved", config["config_name"]),
#                 "01_monkey_test_feature_maps_output.gif", verbose=True, video=True)
# print()

# # plot dynamic feature maps for each concept
# preds_ref = preds[0]
# preds_dyn = preds - np.repeat(np.expand_dims(preds_ref, axis=0), len(preds), axis=0)
# preds_dyn[preds_dyn < 0] = 0
#
# plot_cnn_output(preds_dyn, os.path.join("models/saved", config["config_name"]),
#                 "02_human_train_dyn_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# test_preds_ref = test_preds[0]
# test_dyn = test_preds - np.repeat(np.expand_dims(test_preds_ref, axis=0), len(preds), axis=0)
# test_dyn[test_dyn < 0] = 0
#
# plot_cnn_output(test_dyn, os.path.join("models/saved", config["config_name"]),
#                 "02_monkey_test_dyn_feature_maps_output.gif", verbose=True, video=True)
# print()

# # plot xy positions
# preds_pos = calculate_position(preds, mode="weighted average", return_mode="xy float")
# test_preds_pos = calculate_position(test_preds, mode="weighted average", return_mode="xy float")
# print("shape preds_pos", np.shape(preds_pos))
# print("shape test_preds_pos", np.shape(test_preds_pos))
# plot_ft_map_pos(np.concatenate((preds_pos, test_preds_pos), axis=2),
#                 fig_name="02_human_train_pos.png",
#                 path=os.path.join("models/saved", config["config_name"]),
#                 titles=["Human eyebrow", "Human lips", "Monkey eyebrow", "monkey lips"])

# # # concacenate feature maps for plotting
# # eyebrow_preds = np.concatenate((eyebrow_preds, eyebrow_ft_average), axis=-1)
# # eyebrow_preds = np.concatenate((eyebrow_preds, eyebrow_ft), axis=-1)
# # print("shape concacenate eyebrow_preds", np.shape(eyebrow_preds))
# #
# # # concacenate feature maps for plotting
# # lips_preds = np.concatenate((lips_preds, lips_ft_average), axis=-1)
# # lips_preds = np.concatenate((lips_preds, lips_ft), axis=-1)
# # print("shape concacenate lips_preds", np.shape(lips_preds))
# #
# # # concacenate test feature maps for plotting
# # test_eyebrow_preds = np.concatenate((test_eyebrow_preds, test_eyebrow_ft_average), axis=-1)
# # test_eyebrow_preds = np.concatenate((test_eyebrow_preds, test_eyebrow_ft), axis=-1)
# # print("shape concacenate test_eyebrow_preds", np.shape(test_eyebrow_preds))
# #
# # # concacenate feature maps for plotting
# # test_lips_preds = np.concatenate((test_lips_preds, test_lips_ft_average), axis=-1)
# # test_lips_preds = np.concatenate((test_lips_preds, test_lips_ft), axis=-1)
# # print("shape concacenate test_lips_preds", np.shape(test_lips_preds))
# #
# #
# # # eyebrow feature maps
# # plot_cnn_output(eyebrow_preds, os.path.join("models/saved", config["config_name"]),
# #                 "human_train_eyebrow_feature_maps_output.gif", verbose=True, video=True)
# # print()
# #
# # # test eyebrow feature maps
# # plot_cnn_output(test_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
# #                 "monkey_test_eyebrow_feature_maps_output.gif", verbose=True, video=True)
# # print()
# #
# # # plot eyebrow dynamic
# # eyebrow_preds_ref = eyebrow_preds[0]
# # eyebrow_preds_dyn = eyebrow_preds - np.repeat(np.expand_dims(eyebrow_preds_ref, axis=0), len(eyebrow_preds), axis=0)
# # eyebrow_preds_dyn[eyebrow_preds_dyn < 0] = 0
# #
# # plot_cnn_output(eyebrow_preds_dyn, os.path.join("models/saved", config["config_name"]),
# #                 "human_train_eyebrow_dyn_feature_maps_output.gif", verbose=True, video=True)
# # print()
# #
# # # plot eyebrow test dynamic
# # test_eyebrow_preds_ref = test_eyebrow_preds[0]
# # test_eyebrow_preds_dyn = test_eyebrow_preds - np.repeat(np.expand_dims(test_eyebrow_preds_ref, axis=0), len(test_eyebrow_preds), axis=0)
# # test_eyebrow_preds_dyn[test_eyebrow_preds_dyn < 0] = 0
# #
# # plot_cnn_output(test_eyebrow_preds_dyn, os.path.join("models/saved", config["config_name"]),
# #                 "monkey_test_eyebrow_dyn_feature_maps_output.gif", verbose=True, video=True)
# # print()
# #
# # # lips feature maps
# # plot_cnn_output(lips_preds, os.path.join("models/saved", config["config_name"]),
# #                 "human_train_lips_feature_maps_output.gif", verbose=True, video=True)
# # print()
# #
# # # test lips feature maps
# # plot_cnn_output(test_lips_preds, os.path.join("models/saved", config["config_name"]),
# #                 "monkey_test_lips_feature_maps_output.gif", verbose=True, video=True)
# # print()
# #
# # # plot lips dynamic
# # lips_preds_ref = lips_preds[0]
# # lips_preds_dyn = lips_preds - np.repeat(np.expand_dims(lips_preds_ref, axis=0), len(lips_preds), axis=0)
# # lips_preds_dyn[lips_preds_dyn < 0] = 0
# #
# # plot_cnn_output(lips_preds_dyn, os.path.join("models/saved", config["config_name"]),
# #                 "human_train_lips_dyn_feature_maps_output.gif", verbose=True, video=True)
# # print()
# #
# # # plot test lips dynamic
# # test_lips_preds_ref = test_lips_preds[0]
# # test_lips_preds_dyn = test_lips_preds - np.repeat(np.expand_dims(test_lips_preds_ref, axis=0), len(test_lips_preds), axis=0)
# # test_lips_preds_dyn[test_lips_preds_dyn < 0] = 0
# #
# # plot_cnn_output(test_lips_preds_dyn, os.path.join("models/saved", config["config_name"]),
# #                 "monkey_test_lips_dyn_feature_maps_output.gif", verbose=True, video=True)
# # print()