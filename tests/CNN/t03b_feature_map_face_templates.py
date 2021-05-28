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

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the fit a face template over different feature maps instead of simply summing them up

run: python -m tests.CNN.t03b_feature_map_face_templates
"""

# define configuration
config_path = 'CNN_t03b_feature_map_face_templates_m0001.json'

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

# -------------------------------------------------------------------------------------------------------------------
# fit face template

# build template
# eyebrow_mask = [[7, 12], [8, 21]]
# eyebrow_mask = [[8, 13], [8, 21]]
# eyebrow_mask = [[9, 11], [8, 21]]
# eyebrow_mask = [[8, 10], [8, 21]]
# eyebrow_mask = [[8, 11], [8, 21]]
eyebrow_mask = [[8, 13], [7, 14]]
# eyebrow_mask = [[8, 13], [7, 22]]
# lips_mask = [[15, 19], [10, 19]]
lips_mask = [[18, 22], [10, 19]]
# config['rbf_sigma'] = [7000, 20000]
config['rbf_sigma'] = [6000, 20000]
config['pattern_mask'] = [eyebrow_mask, lips_mask]

# set reference frame
config['pattern_idx'] = 0
patternFS = PatternFeatureSelection(config)
preds = patternFS.fit(preds)
print("max preds", np.amax(preds))
eyebrow_ft = np.expand_dims(preds[..., 0], axis=3)
lips_ft = np.expand_dims(preds[..., 1], axis=3)
print("shape preds", np.shape(preds))
print("shape eyebrow_ft", np.shape(eyebrow_ft))
print("max eyebrow_ft", np.amax(eyebrow_ft))
print("shape lips_ft", np.shape(lips_ft))
print()

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

# predict pattern
test_preds = patternFS.transform(test_preds)
test_eyebrow_ft = np.expand_dims(test_preds[..., 0], axis=3)
test_lips_ft = np.expand_dims(test_preds[..., 1], axis=3)
print("shape test_preds", np.shape(test_preds))
print("shape test_eyebrow_ft", np.shape(test_eyebrow_ft))
print("shape test_lips_ft", np.shape(test_lips_ft))
print()

# --------------------------------------------------------------------------------------------------------------------
# create test to compare with the average

# train
# compute average eyebrow feature maps
eyebrow_ft_average = np.mean(eyebrow_preds, axis=-1)
eyebrow_ft_average = np.expand_dims(eyebrow_ft_average, axis=3)
# compute average lips feature maps
lips_ft_average = np.mean(lips_preds, axis=-1)
lips_ft_average = np.expand_dims(lips_ft_average, axis=3)

# test
# compute average test eyebrow feature maps
test_eyebrow_ft_average = np.mean(test_eyebrow_preds, axis=-1)
test_eyebrow_ft_average = np.expand_dims(test_eyebrow_ft_average, axis=3)
# compute average lips feature maps
test_lips_ft_average = np.mean(test_lips_preds, axis=-1)
test_lips_ft_average = np.expand_dims(test_lips_ft_average, axis=3)

# --------------------------------------------------------------------------------------------------------------------
# plots

# plot feature maps for each concept
plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
                "01_human_train_feature_maps_output.gif", verbose=True, video=True)
print()

plot_cnn_output(test_preds, os.path.join("models/saved", config["config_name"]),
                "01_monkey_test_feature_maps_output.gif", verbose=True, video=True)
print()

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

# plot xy positions
preds_pos = calculate_position(preds, mode="weighted average", return_mode="xy float")
test_preds_pos = calculate_position(test_preds, mode="weighted average", return_mode="xy float")

color_seq = np.arange(len(preds_pos))
plt.figure()
plt.subplot(2, 2, 1)
print("shape preds_pos", np.shape(preds_pos))
print("shape test_preds", np.shape(test_preds))
plt.scatter(preds_pos[:, 1, 0], preds_pos[:, 0, 0], c=color_seq)
plt.xlim(13.5, 14.0)
plt.ylim(11.7, 12.2)
plt.colorbar()
plt.title("Human Avatar Eyebrow")
plt.subplot(2, 2, 2)
plt.scatter(test_preds_pos[:, 1, 0], test_preds_pos[:, 0, 0], c=color_seq)
plt.xlim(13.5, 14.0)
plt.ylim(11.7, 12.2)
plt.colorbar()
plt.title("Monkey Avatar Eyebrow")
plt.subplot(2, 2, 3)
plt.scatter(preds_pos[:, 1, 1], preds_pos[:, 0, 1], c=color_seq)
plt.xlim(13.2, 13.6)
plt.ylim(13.6, 14.6)
plt.colorbar()
plt.title("Human Avatar Lips")
plt.subplot(2, 2, 4)
plt.scatter(test_preds_pos[:, 1, 1], test_preds_pos[:, 0, 1], c=color_seq)
plt.xlim(13.2, 13.6)
plt.ylim(13.6, 14.6)
plt.colorbar()
plt.title("Monkey Avatar Lips")

plt.savefig(os.path.join("models/saved", config["config_name"], "02_human_train_pos.png"))

# # concacenate feature maps for plotting
# eyebrow_preds = np.concatenate((eyebrow_preds, eyebrow_ft_average), axis=-1)
# eyebrow_preds = np.concatenate((eyebrow_preds, eyebrow_ft), axis=-1)
# print("shape concacenate eyebrow_preds", np.shape(eyebrow_preds))
#
# # concacenate feature maps for plotting
# lips_preds = np.concatenate((lips_preds, lips_ft_average), axis=-1)
# lips_preds = np.concatenate((lips_preds, lips_ft), axis=-1)
# print("shape concacenate lips_preds", np.shape(lips_preds))
#
# # concacenate test feature maps for plotting
# test_eyebrow_preds = np.concatenate((test_eyebrow_preds, test_eyebrow_ft_average), axis=-1)
# test_eyebrow_preds = np.concatenate((test_eyebrow_preds, test_eyebrow_ft), axis=-1)
# print("shape concacenate test_eyebrow_preds", np.shape(test_eyebrow_preds))
#
# # concacenate feature maps for plotting
# test_lips_preds = np.concatenate((test_lips_preds, test_lips_ft_average), axis=-1)
# test_lips_preds = np.concatenate((test_lips_preds, test_lips_ft), axis=-1)
# print("shape concacenate test_lips_preds", np.shape(test_lips_preds))
#
#
# # eyebrow feature maps
# plot_cnn_output(eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "human_train_eyebrow_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# # test eyebrow feature maps
# plot_cnn_output(test_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
#                 "monkey_test_eyebrow_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# # plot eyebrow dynamic
# eyebrow_preds_ref = eyebrow_preds[0]
# eyebrow_preds_dyn = eyebrow_preds - np.repeat(np.expand_dims(eyebrow_preds_ref, axis=0), len(eyebrow_preds), axis=0)
# eyebrow_preds_dyn[eyebrow_preds_dyn < 0] = 0
#
# plot_cnn_output(eyebrow_preds_dyn, os.path.join("models/saved", config["config_name"]),
#                 "human_train_eyebrow_dyn_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# # plot eyebrow test dynamic
# test_eyebrow_preds_ref = test_eyebrow_preds[0]
# test_eyebrow_preds_dyn = test_eyebrow_preds - np.repeat(np.expand_dims(test_eyebrow_preds_ref, axis=0), len(test_eyebrow_preds), axis=0)
# test_eyebrow_preds_dyn[test_eyebrow_preds_dyn < 0] = 0
#
# plot_cnn_output(test_eyebrow_preds_dyn, os.path.join("models/saved", config["config_name"]),
#                 "monkey_test_eyebrow_dyn_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# # lips feature maps
# plot_cnn_output(lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "human_train_lips_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# # test lips feature maps
# plot_cnn_output(test_lips_preds, os.path.join("models/saved", config["config_name"]),
#                 "monkey_test_lips_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# # plot lips dynamic
# lips_preds_ref = lips_preds[0]
# lips_preds_dyn = lips_preds - np.repeat(np.expand_dims(lips_preds_ref, axis=0), len(lips_preds), axis=0)
# lips_preds_dyn[lips_preds_dyn < 0] = 0
#
# plot_cnn_output(lips_preds_dyn, os.path.join("models/saved", config["config_name"]),
#                 "human_train_lips_dyn_feature_maps_output.gif", verbose=True, video=True)
# print()
#
# # plot test lips dynamic
# test_lips_preds_ref = test_lips_preds[0]
# test_lips_preds_dyn = test_lips_preds - np.repeat(np.expand_dims(test_lips_preds_ref, axis=0), len(test_lips_preds), axis=0)
# test_lips_preds_dyn[test_lips_preds_dyn < 0] = 0
#
# plot_cnn_output(test_lips_preds_dyn, os.path.join("models/saved", config["config_name"]),
#                 "monkey_test_lips_dyn_feature_maps_output.gif", verbose=True, video=True)
# print()
