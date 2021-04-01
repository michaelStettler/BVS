import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.load_config import load_config
from utils.load_data import load_data
from utils.load_extraction_model import load_extraction_model
from utils.feat_map_filter_processing import feat_map_filter_processing
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the computations of feature positions within a feature map

run: python -m tests.CNN.t03a_feature_map_positions_face_shape
"""

# define configuration
config_path = 'CNN_t03a_feature_map_positions_face_shape_m0001.json'

# declare parameters
eyebrow_ft_idx = [148, 209, 208, 67, 211, 141, 90, 196, 174, 179, 59, 101, 225, 124, 125, 156]  # from t02_find_semantic_units
slice_pos = 9

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

# ----------------------------------------------------------------------------------------------------------------------
# test 1 - compare cnn eye_brow units responses for expression C2 over Human and Monkey avatar
print("[TEST 1] Compare eyebrow feat. map for C2 across Human and Monkey avatar")
# Human Avatar
# load data
config["train_avatar"] = "human_orig"
config["train_expression"] = ["c2"]
data = load_data(config)[0]
# raw_seq = load_data(config, get_raw=True)[0]
print("[TEST 1] Human C2 loaded")
print("[TEST 1] shape data", np.shape(data))
print("[TEST 1] Start predictions")
preds_HC2 = model.predict(data)[..., eyebrow_ft_idx]
print("shape preds_HC2", np.shape(preds_HC2))
print("min max preds_HC2", np.amin(preds_HC2), np.amax(preds_HC2))
preds_HC2 = feat_map_filter_processing(preds_HC2,
                                       ref=preds_HC2[0],
                                       norm=1000,
                                       activation='ReLu',
                                       filter='spatial_mean',
                                       verbose=True)
# Monkey Avatar
# load data
config["train_avatar"] = "monkey_orig"
data = load_data(config)[0]
print("[TEST 1] Monkey C2 loaded")
print("[TEST 1] shape data", np.shape(data))
print("[TEST 1] Start predictions")
preds_MC2 = model.predict(data)[..., eyebrow_ft_idx]
print("shape preds_MC2", np.shape(preds_MC2))
print("min max preds_MC2", np.amin(preds_MC2), np.amax(preds_MC2))
preds_MC2 = feat_map_filter_processing(preds_MC2,
                                       ref=preds_MC2[0],
                                       norm=1000,
                                       activation='ReLu',
                                       filter='spatial_mean',
                                       verbose=True)

# plot raw responses of the dynamic
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(preds_HC2[:, :, slice_pos, 0])  # slice over the 10 column to trz to get the eyebrow
plt.title("Human Avatar")
plt.subplot(1, 2, 2)
plt.plot(preds_MC2[:, :, slice_pos, 0])  # slice over the 10 column to trz to get the eyebrow
plt.title("Monkey Avatar")
plt.savefig(os.path.join(save_path, "test1_Hum_vs_Monk_C2_expression_eyebrow_fm_slice_{}".format(slice_pos)))


