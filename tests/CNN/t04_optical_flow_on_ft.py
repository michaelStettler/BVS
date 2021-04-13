import os
import numpy as np
import tensorflow as tf
import cv2

from utils.load_config import load_config
from utils.load_data import load_data
from utils.CNN.extraction_model import load_extraction_model

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
size_ft = (280, 280)
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

# plot first feature map
pred0 = np.array(predictions[0]) * 255
print("shape pred0", np.shape(pred0))
pred0_fm148 = cv2.cvtColor(pred0.astype(np.uint8), cv2.COLOR_GRAY2BGR)
cv2.imwrite(os.path.join(path, 'pred0_fm148.png'), pred0_fm148)

# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros(size_ft + (3, ))  # add tuple to set up size of rgb image
print("[TEST 1] shape mask", np.shape(mask))
# Sets image saturation to maximum
mask[..., 1] = 255

# initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
of_video_filename = os.path.join(path, 'optical_flow_fm148.mp4')
of_out = cv2.VideoWriter(of_video_filename, fourcc, fps, size_ft)
# create raw output
fm_video_filename = os.path.join(path, 'raw_output_fm148.mp4')
fm_out = cv2.VideoWriter(fm_video_filename, fourcc, fps, size_ft)

# for i in range(1, np.shape(predictions)[0]):
for i in range(1, 5):
    # get current and previous frame
    prev_frame = np.array(np.array(predictions[i - 1]) * 255).astype(np.uint8)
    curr_frame = np.array(np.array(predictions[i]) * 255).astype(np.uint8)

    # tests
    # curr_frame = np.zeros(size_ft).astype(np.uint8)
    prev_frame = np.zeros(size_ft).astype(np.uint8)
    curr_frame = np.zeros(size_ft).astype(np.uint8)
    # single dot
    # prev_frame[100:101, (30+i):(31+i)] = 255  # for size_ft (280, 280)
    # curr_frame[100:101, (31+i):(32+i)] = 255
    # cube
    prev_frame[100:110, (30+i):(40+i)] = 255  # for size_ft (280, 280)
    curr_frame[100:110, (31+i):(41+i)] = 255
    # single dot
    # prev_frame[10:11, (3+i):(4+i)] = 255  # for size_ft (28, 28)
    # curr_frame[10:11, (4+i):(5+i)] = 255
    # cube
    # prev_frame[10:14, (3+i):(7+i)] = 255  # for size_ft (28, 28)
    # curr_frame[10:14, (4+i):(8+i)] = 255
    print("shape curr_frame", np.shape(curr_frame))
    print("min max curr_frame", np.amin(curr_frame), np.amax(curr_frame))

    # transform current frame to BGR for visualization
    fm = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2BGR)
    print("min max fm", np.amin(fm), np.amax(fm))

    # compute optical flow
    # parameters explanation: https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/
    #   - winsize: It is the average window size, larger the size, the more robust the algorithm is to noise, and
    #   provide fast motion detection, though gives blurred motion fields.
    #   - poly_n : It is typically 5 or 7, it is the size of the pixel neighbourhood which is used to find polynomial
    #   expansion between the pixels.
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame,
                                        flow=None,
                                        pyr_scale=0.5,
                                        levels=1,
                                        winsize=3,  # 15
                                        iterations=5,  # 3
                                        poly_n=5,  # 5
                                        poly_sigma=1.2,
                                        flags=0)

    # build image
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    print("shape magnitude", np.shape(magnitude))
    print("min max magnitude", np.amin(magnitude), np.amax(magnitude))
    print("shape angle", np.shape(angle))
    print("min max angle", np.amin(angle), np.amax(angle))
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    print("min max mask[..., 0]", np.amin(mask[..., 0]), np.amax(mask[..., 0]))
    print("min max mask[..., 1]", np.amin(mask[..., 1]), np.amax(mask[..., 1]))
    print("min max mask[..., 2]", np.amin( mask[..., 2]), np.amax( mask[..., 2]))
    bgr = cv2.cvtColor(mask.astype('float32'), cv2.COLOR_HSV2BGR)
    # bgr = np.array(bgr).astype(np.uint8)
    # bgr[bgr < 0] = 0
    # bgr[:5, :5, 0] = 255
    print("min max bgr", np.amin(bgr), np.amax(bgr))
    print("min max b channel", np.amin(bgr[..., 0]), np.amax(bgr[..., 0]))
    print("min max g channel", np.amin(bgr[..., 1]), np.amax(bgr[..., 1]))
    print("min max r channel", np.amin(bgr[..., 2]), np.amax(bgr[..., 2]))

    cv2.imwrite(os.path.join(path, "bgr_{}.png".format(i)), bgr.astype(np.uint8))
    # write image
    of_out.write(bgr.astype(np.uint8))  # seems to create weird horizontal lines
    fm_out.write(fm)
    print()

of_out.release()
fm_out.release()
print("[TEST 1] Finish optical flow on first feature map")