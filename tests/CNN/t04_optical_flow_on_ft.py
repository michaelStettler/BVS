import os
import numpy as np
import tensorflow as tf
import cv2

from utils.load_config import load_config
from utils.load_data import load_data
from utils.load_extraction_model import load_extraction_model
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the computations of feature positions within a feature map

run: python -m tests.CNN.t04_optical_flow_on_ft
"""

# define configuration
config_path = 'CNN_t04_optical_flow_m0001.json'

# load config
config = load_config(config_path, path='configs/CNN')

# create save folder in case
path = os.path.join("models/saved", config["config_name"])
if not os.path.exists(path):
    os.mkdir(path)

# choose feature map index
eyebrow_ft_idx = 148   # the index comes from t02_find_semantic_units, it is the highest IoU score for eyebrow

# load and define model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(model.output)[1:3])
print("size_ft", size_ft)

# load morphing sequence
data = load_data(config)
# raw_seq = load_data(config, get_raw=True)[0]
print("[DATA] loaded sequence")

# predict
preds = model.predict(data)[..., eyebrow_ft_idx]
preds = np.expand_dims(preds, axis=3)
preds = preds / np.amax(preds)  # normalize so we can compare with the positions
print("[PRED] Shape predictions", np.shape(preds))
print("[PRED] Finish to predict")
print()

# ----------------------------------------------------------------------------------------------------------------------
# test 1 - compute optical flow on eye brow feature map
# code taken frm: https://nanonets.com/blog/optical-flow/
# transform to gray images
predictions = preds[..., 0]

# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros(size_ft + (3, ))  # add tuple to set up size of rgb image
print("[TEST 1] shape mask", np.shape(mask))
# Sets image saturation to maximum
mask[..., 1] = 255

# initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
video_filename = os.path.join(path, 'output.mp4')
out = cv2.VideoWriter(video_filename, fourcc, fps, size_ft)

for i in range(1, np.shape(predictions)[0]):
    # compute optical flow  todo check what are all those parameters...
    flow = cv2.calcOpticalFlowFarneback(predictions[i - 1], predictions[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # build image
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    bgr = cv2.cvtColor(mask.astype('float32'), cv2.COLOR_HSV2BGR)

    # write image
    out.write(bgr)

out.release()
print("[TEST 1] Finish optical flow on first feature map")