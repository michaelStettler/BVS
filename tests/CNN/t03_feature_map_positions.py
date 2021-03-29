import os
import numpy as np
import tensorflow as tf
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

# ----------------------------------------------------------------------------------------------------------------------
# test 1 - control weighted average
# define feature map controls
num_entry = 3
num_ft = 1
preds1 = np.zeros((num_entry,) + size_ft + (num_ft,))
print("[TEST 1] shape predictions", np.shape(preds1))
# -> should get (0, .33)
preds1[1, 0, 0, 0] = 2
preds1[1, 0, 1, 0] = 1

# create test with 1 at each corners, should get -> (13,5, 13,5)
preds1[2, 0, 0, 0] = 1
preds1[2, 0, -1, 0] = 1
preds1[2, -1, 0, 0] = 1
preds1[2, -1, -1, 0] = 1

# compute position vectors
preds1_pos = calculate_position(preds1, mode="weighted average", return_mode="xy float")
print("preds1_pos")
print(preds1_pos)
preds1_pos = calculate_position(preds1, mode="weighted average", return_mode="array")
print("shape preds1_pos", np.shape(preds1_pos))

# plot positions
plot_cnn_output(preds1_pos, os.path.join("models/saved", config["config_name"]),
                config['v4_layer'] + "_test1.gif",
                # image="",
                video=True,
                verbose=False)
print("[TEST 1] Finished plotting test1 positions")
print()

# ----------------------------------------------------------------------------------------------------------------------
# test 2 - control eye brow feature map
# load morphing sequence
print("[TEST 2] Test semantic selection on morphing space")
data = load_data(config)
raw_seq = load_data(config, get_raw=True)[0]

# predict
preds = model.predict(data)[..., eyebrow_ft_idx]
preds = np.expand_dims(preds, axis=3)
preds = preds / np.amax(preds)  # normalize so we can compare with the positions
print("[TEST 2] shape predictions", np.shape(preds))
preds_init = preds[0]
dyn_preds = preds - np.repeat(np.expand_dims(preds_init, axis=0), np.shape(preds)[0], axis=0)
dyn_preds[dyn_preds < 0] = 0
print("[TEST 2] shape dyn_preds", np.shape(dyn_preds))

# compute positions
preds_pos = calculate_position(preds, mode="weighted average", return_mode="array")
print("[TEST 2] shape preds_pos", np.shape(preds_pos))
dyn_preds_pos = calculate_position(dyn_preds, mode="weighted average", return_mode="array")
print("[TEST 2] shape dyn_preds_pos", np.shape(dyn_preds_pos))

# concatenate prediction and position for plotting
results = np.concatenate((preds, preds_pos, dyn_preds, dyn_preds_pos), axis=3)
print("[TEST 2] shape results", np.shape(results))

# # plot results
# plot_cnn_output(results, os.path.join("models/saved", config["config_name"]),
#                 config['v4_layer'] + "_eye_brow_select_{}.gif".format(eyebrow_ft_idx),
#                 image=raw_seq,
#                 video=True,
#                 verbose=False)
# print("[TEST 2] Finished plotted cnn feature maps", np.shape(results))
print()

# ----------------------------------------------------------------------------------------------------------------------
# test 3 - compute optical flow on eye brow feature map
# code taken frm: https://nanonets.com/blog/optical-flow/
import cv2
# transform to gray images
predictions = preds[..., 0]

# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros(size_ft + (3, ))  # add tuple to set up size of rgb image
print("[TEST 3] shape mask", np.shape(mask))
# Sets image saturation to maximum
mask[..., 1] = 255

for i in range(np.shape(predictions)[0] - 1):
    # compute optical flow  todo check what are all those parameters...
    flow = cv2.calcOpticalFlowFarneback(predictions[i], predictions[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # build image
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask.astype('float32'), cv2.COLOR_HSV2BGR)
    print("shape rgb", np.shape(rgb))
    # Opens a new window and displays the output frame
    # cv.imshow("dense optical flow", rgb)
