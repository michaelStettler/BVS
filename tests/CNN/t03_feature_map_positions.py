import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.load_config import load_config
from utils.load_data import load_data
from utils.load_extraction_model import load_extraction_model
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the computations of feature positions within a feature map

run: python -m tests.CNN.t03_feature_map_positions
"""

# define configuration
config_path = 'CNN_t03_feature_map_positions_m0001.json'

# load config
config = load_config(config_path, path='configs/CNN')

# choose feature map index
eyebrow_ft_idx = 148   # the index comes from t02_find_semantic_units, it is the highest IoU score for eyebrow

# load and define model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(model.output)[1:3])
print("size_ft", size_ft)

# # ----------------------------------------------------------------------------------------------------------------------
# # test 1 - control weighted average
# # define feature map controls
# num_entry = 3
# num_ft = 1
# preds1 = np.zeros((num_entry,) + size_ft + (num_ft,))
# print("[TEST 1] shape predictions", np.shape(preds1))
# # -> should get (0, .33)
# preds1[1, 0, 0, 0] = 2
# preds1[1, 0, 1, 0] = 1
#
# # create test with 1 at each corners, should get -> (13,5, 13,5)
# preds1[2, 0, 0, 0] = 1
# preds1[2, 0, -1, 0] = 1
# preds1[2, -1, 0, 0] = 1
# preds1[2, -1, -1, 0] = 1
#
# # compute position vectors
# preds1_pos = calculate_position(preds1, mode="weighted average", return_mode="xy float")[..., 0]
# print("preds1_pos")
# print(preds1_pos)
# preds1_pos = calculate_position(preds1, mode="weighted average", return_mode="array")
# print("shape preds1_pos", np.shape(preds1_pos))
#
# # plot positions
# plot_cnn_output(preds1_pos, os.path.join("models/saved", config["config_name"]),
#                 "test1_" + config['v4_layer'] + "_test1.gif",
#                 # image="",
#                 video=True,
#                 verbose=False)
# print("[TEST 1] Finished plotting test1 positions")
# print()

# ----------------------------------------------------------------------------------------------------------------------
# test 2 - control eye brow feature map
# load morphing sequence
print("[TEST 2] Test semantic selection on morphing space")
data = load_data(config)
raw_seq = load_data(config, get_raw=True)[0]

# # predict
# preds = model.predict(data)[..., eyebrow_ft_idx]
# preds = np.expand_dims(preds, axis=3)
# preds = preds / np.amax(preds)  # normalize so we can compare with the positions
# print("[TEST 2] shape predictions", np.shape(preds))
# preds_init = preds[0]
# dyn_preds = preds - np.repeat(np.expand_dims(preds_init, axis=0), np.shape(preds)[0], axis=0)
# dyn_preds[dyn_preds < 0] = 0
# print("[TEST 2] shape dyn_preds", np.shape(dyn_preds))
#
# # compute positions
# preds_pos = calculate_position(preds, mode="weighted average", return_mode="array")
# print("[TEST 2] shape preds_pos", np.shape(preds_pos))
# dyn_preds_pos = calculate_position(dyn_preds, mode="weighted average", return_mode="array")
# print("[TEST 2] shape dyn_preds_pos", np.shape(dyn_preds_pos))
#
# # concatenate prediction and position for plotting
# results = np.concatenate((preds, preds_pos, dyn_preds, dyn_preds_pos), axis=3)
# print("[TEST 2] shape results", np.shape(results))
#
# # plot results
# plot_cnn_output(results, os.path.join("models/saved", config["config_name"]),
#                 "test2_" + config['v4_layer'] + "_eye_brow_select_{}.gif".format(eyebrow_ft_idx),
#                 image=raw_seq,
#                 video=True,
#                 verbose=False)
# print("[TEST 2] Finished plotted cnn feature maps", np.shape(results))
# print()

# # ----------------------------------------------------------------------------------------------------------------------
# # test 3 - test average of floating points
# # compare between 1 feature map to all feature maps
# # predict responses for all eyebrow units
# print("[TEST 3] Get all eye brow semantic units")
# eyebrow_ft_idx = [148, 209, 208, 67, 211, 141, 90, 196, 174, 179, 59, 101, 225, 124, 125, 156]
# preds = model.predict(data)[..., eyebrow_ft_idx]
# preds = preds / np.amax(preds)  # normalize so we can compare with the positions
# print("[TEST 3] shape predictions", np.shape(preds))
# preds_init = preds[0]
# dyn_preds = preds - np.repeat(np.expand_dims(preds_init, axis=0), np.shape(preds)[0], axis=0)
# dyn_preds[dyn_preds < 0] = 0
#
# # compute floating value positions
# dyn_pos = calculate_position(dyn_preds, mode="weighted average", return_mode="xy float")
# print("[TEST 3] Shape dyn_pos", np.shape(dyn_pos))
# dyn_pos_mean = np.mean(dyn_pos, axis=2)
# print("[TEST 3] Shape dyn_pos_mean", np.shape(dyn_pos_mean))
#
# # set color to represent time
# color_seq = np.arange(len(dyn_pos_mean))
# # plot raw positions of first feature map
# plt.figure()
# plt.scatter(dyn_pos[:, 1, 0], dyn_pos[:, 0, 0], c=color_seq)  # plot first feature map
# plt.colorbar()
# plt.savefig(os.path.join("models/saved", config["config_name"], "test3_ft0_dyn_xy_float_pos"))
#
# # plot mean positions over all eyebrow feature map
# plt.figure()
# plt.scatter(dyn_pos_mean[:, 1], dyn_pos_mean[:, 0], c=color_seq)  # plot first feature map
# plt.colorbar()
# plt.savefig(os.path.join("models/saved", config["config_name"], "test3_mean_dyn_xy_float_pos"))
#
# # create array from float value to plot with cnn output function
# size_mean_ft = 56
# mean_feature_map = np.zeros((len(dyn_pos_mean), size_mean_ft, size_mean_ft, 1))
# for i in range(len(dyn_pos_mean)):
#     x = int(np.round(dyn_pos_mean[i, 0] * (size_mean_ft/28)))
#     y = int(np.round(dyn_pos_mean[i, 1] * (size_mean_ft/28)))
#     mean_feature_map[i, x, y, 0] = 1
#
# # plot mean positions
# plot_cnn_output(mean_feature_map, os.path.join("models/saved", config["config_name"]),
#                 "test3_" + config['v4_layer'] + "_mean_xy_float_pos.gif",
#                 image=raw_seq,
#                 video=True,
#                 verbose=False)
# print("[Test 3] Finish plotting feature maps")
# print()

# ----------------------------------------------------------------------------------------------------------------------
# test 4 - test mean feature maps
# compute the mean activation of all "eyebrow" specific feature map and compare with the dynamic
# compute mean directly on the feature maps
eyebrow_ft_idx = [148, 209, 208, 67, 211, 141, 90, 196, 174, 179, 59, 101, 225, 124, 125, 156]
preds = model.predict(data)[..., eyebrow_ft_idx]
eye_brow_mean_pred = np.mean(preds, axis=3)
eye_brow_mean_pred = eye_brow_mean_pred / np.amax(eye_brow_mean_pred)  # normalize so we can compare with the positions
eye_brow_mean_pred = np.expand_dims(eye_brow_mean_pred, axis=3)
print("[TEST 4] Shape eye_brow_mean_pred", np.shape(eye_brow_mean_pred))

# compute dynamic
eye_brow_mean_pred_init = np.repeat(np.expand_dims(eye_brow_mean_pred[0], axis=0), len(eye_brow_mean_pred), axis=0)
dyn_eye_brow_mean_pred = eye_brow_mean_pred - eye_brow_mean_pred_init
dyn_eye_brow_mean_pred[dyn_eye_brow_mean_pred < 0] = 0
print("[TEST 4] Prev-fm-process: min max dyn_eye_brow_mean_pred", np.amin(dyn_eye_brow_mean_pred), np.amax(dyn_eye_brow_mean_pred))

# exponential
# dyn_eye_brow_mean_pred = np.exp(2*dyn_eye_brow_mean_pred) - 1

# standardize
# dyn_eye_brow_mean_pred = dyn_eye_brow_mean_pred - np.amin(dyn_eye_brow_mean_pred)
# dyn_eye_brow_mean_pred = dyn_eye_brow_mean_pred / np.amax(dyn_eye_brow_mean_pred)  # normalize so we can compare with the positions

# spatial noise mean filter
input = tf.convert_to_tensor(dyn_eye_brow_mean_pred, dtype=tf.float32)
kernel = np.ones((3, 3)) / 9
kernel = tf.convert_to_tensor(np.expand_dims(kernel, axis=(2, 3)), dtype=tf.float32)  # build 4D kernel with input and output size of 1
dyn_eye_brow_mean_pred = tf.nn.convolution(input, kernel, strides=1, padding='SAME').numpy()

# spatial noise median filter
# dyn_eye_brow_mean_pred_clean = np.copy(dyn_eye_brow_mean_pred)
# for frame in range(len(dyn_eye_brow_mean_pred)):
#     for i in range(1, size_ft[0] - 1):
#         for j in range(1, size_ft[1] - 1):
#             patch = dyn_eye_brow_mean_pred[frame, (i-1):(i+2), (j-1):(j+2), 0]
#             # dyn_eye_brow_mean_pred_clean[frame, i, j, 0] = np.median(patch)

print("[TEST 4] Post-fm-process: min max dyn_eye_brow_mean_pred", np.amin(dyn_eye_brow_mean_pred), np.amax(dyn_eye_brow_mean_pred))
print("[TEST 4] shape dyn_preds", np.shape(dyn_eye_brow_mean_pred))

# set color to represent time
color_seq = np.arange(len(dyn_eye_brow_mean_pred))

# print raw response
plt.figure()
slice_pos = 9
plt.plot(eye_brow_mean_pred[:, :, slice_pos, 0])  # slice over the 10 column to trz to get the eyebrow
plt.savefig(os.path.join("models/saved", config["config_name"], "test4_eyebrow_raw_slice_{}".format(slice_pos)))

# plot raw responses of the dynamic
plt.figure()
slice_pos = 9
plt.plot(dyn_eye_brow_mean_pred[:, :, slice_pos, 0])  # slice over the 10 column to trz to get the eyebrow
plt.savefig(os.path.join("models/saved", config["config_name"], "test4_eyebrow_raw_dyn_slice_{}".format(slice_pos)))

# plot raw positions of first feature map
preds_pos = calculate_position(eye_brow_mean_pred, mode="weighted average", return_mode="xy float")
plt.figure()
plt.scatter(preds_pos[:, 1, 0], preds_pos[:, 0, 0], c=color_seq)  # plot first feature map
plt.xlim(13, 15)
plt.colorbar()
plt.savefig(os.path.join("models/saved", config["config_name"], "test4_eyebrow_mean_xy_float_pos"))

# plot mean positions over all eyebrow feature map
dyn_preds_pos = calculate_position(dyn_eye_brow_mean_pred, mode="weighted average", return_mode="xy float")
plt.figure()
plt.scatter(dyn_preds_pos[:, 1, 0], dyn_preds_pos[:, 0, 0], c=color_seq)  # plot first feature map
plt.xlim(13, 15)
plt.colorbar()
plt.savefig(os.path.join("models/saved", config["config_name"], "test4_eyebrow_mean_dyn_xy_float_pos"))

# compute positions
preds_pos = calculate_position(eye_brow_mean_pred, mode="weighted average", return_mode="array")
print("[TEST 4] shape preds_pos", np.shape(preds_pos))
dyn_preds_pos = calculate_position(dyn_eye_brow_mean_pred, mode="weighted average", return_mode="array")
print("[TEST 4] shape dyn_preds_pos", np.shape(dyn_preds_pos))

# concatenate prediction and position for plotting
results = np.concatenate((eye_brow_mean_pred, preds_pos, dyn_eye_brow_mean_pred, dyn_preds_pos), axis=3)
print("[TEST 4] shape results", np.shape(results))


# plot means feature maps and positions
plot_cnn_output(results, os.path.join("models/saved", config["config_name"]),
                "test4_" + config['v4_layer'] + "_mean_feature_map.gif",
                image=raw_seq,
                video=True,
                verbose=False)
print("[Test 4] Finish plotting results")
print()

# # ----------------------------------------------------------------------------------------------------------------------
# # test 5 - test mean feature maps on lips
# print("[TEST 5] Test on lips units")
# lips_ft_idx = [79, 120, 125, 0, 174, 201, 193, 247, 77, 249, 210, 149, 89, 197, 9, 251, 237, 165, 101, 90, 27, 158, 154,
#                10, 168, 156, 44, 23, 34, 85, 207]
#
# # load c3 expression to test mouth movements
# config["train_expression"] = ["c3"]
# data = load_data(config)
# raw_seq = load_data(config, get_raw=True)[0]
# print("[TEST 5] Loaded c3 expression")
#
# # predict data and keep only the lips units
# preds = model.predict(data)[..., lips_ft_idx]
# # preds = preds / np.amax(preds)  # normalize so we can compare with the positions
# print("[TEST 5] shape predictions", np.shape(preds))
#
# # compute mean directly on the feature maps
# lips_mean_pred = np.mean(preds, axis=3)
# lips_mean_pred = lips_mean_pred / np.amax(lips_mean_pred)  # normalize so we can compare with the positions
# lips_mean_pred = np.expand_dims(lips_mean_pred, axis=3)
# print("[TEST 5] Shape lips_mean_pred", np.shape(lips_mean_pred))
# lips_mean_pred_init = np.repeat(np.expand_dims(lips_mean_pred[0], axis=0), len(lips_mean_pred), axis=0)
# dyn_lips_mean_pred = lips_mean_pred - lips_mean_pred_init
# dyn_lips_mean_pred[dyn_lips_mean_pred < 0] = 0
# print("[TEST 5] shape dyn_preds", np.shape(dyn_lips_mean_pred))
#
# # plot raw positions of first feature map
# preds_pos = calculate_position(lips_mean_pred, mode="weighted average", return_mode="xy float")
# plt.figure()
# plt.scatter(preds_pos[:, 1, 0], preds_pos[:, 0, 0], c=color_seq)  # plot first feature map
# plt.colorbar()
# plt.savefig(os.path.join("models/saved", config["config_name"], "test5_lips_mean_xy_float_pos"))
#
# # plot mean positions over all eyebrow feature map
# dyn_preds_pos = calculate_position(dyn_lips_mean_pred, mode="weighted average", return_mode="xy float")
# plt.figure()
# plt.scatter(dyn_preds_pos[:, 1, 0], dyn_preds_pos[:, 0, 0], c=color_seq)  # plot first feature map
# plt.colorbar()
# plt.savefig(os.path.join("models/saved", config["config_name"], "test5_lips_mean_dyn_xy_float_pos"))
#
# # compute positions
# preds_pos = calculate_position(lips_mean_pred, mode="weighted average", return_mode="array")
# print("[TEST 5] shape preds_pos", np.shape(preds_pos))
# dyn_preds_pos = calculate_position(dyn_lips_mean_pred, mode="weighted average", return_mode="array")
# print("[TEST 5] shape dyn_preds_pos", np.shape(dyn_preds_pos))
#
# # concatenate prediction and position for plotting
# results = np.concatenate((lips_mean_pred, preds_pos, dyn_lips_mean_pred, dyn_preds_pos), axis=3)
# print("[TEST 5] shape results", np.shape(results))
#
# # plot means feature maps and positions
# plot_cnn_output(results, os.path.join("models/saved", config["config_name"]),
#                 "test5_" + config['v4_layer'] + "_lips_mean_feature_map.gif",
#                 image=raw_seq,
#                 video=True,
#                 verbose=False)
