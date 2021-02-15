"""
2021/01/20
    This script creates an animation of the cnn response to a moving face.
    for 150 frames this takes approximately 10 minutes
2021/01/29
    adding options to plot difference
"""

import os
import numpy as np

from utils.load_config import load_config
from utils.load_data import load_data
from utils.plot_cnn_output import plot_cnn_output
from models.NormBase import NormBase
"""
config["movie_name"]: name under which the movie is saved, extension .mp4 is automatically added
config["plot_title"]: title of the plot
config["plot_option"]: choose which property to plot
    - "cnn_output": plot output
    - "cnn_output_difference": plot difference to some reference, 
        choose config["difference_option"]
        - "first": first frame is taken as reference
        - "stride{int}": for example for "stride2" take difference to two frames before
        - "reference": take difference to tuning vector (average of neutral frames)
    - "maximum": plot index of maximal value
    - "maximum{int}": plot average index for {int} highest values [TODO]
    - "weighted average": plot index of weighted average of activation
- config["highlight_option"]: choose which feature maps to highlight, based on plotted property (plot_vector)
    - "maximum": highlight maps with maximum value
- config["plot_reduce"]: if True reduces the displayed feature maps to selection in highlight, improves performance
"""
# load config
# t0001: human_anger, t0002: human_fear, t0003: monkey_anger, t0004: monkey_fear  --> plot cnn_output
# t0005: human_anger, t0006: human_fear, t0007: monkey_anger, t0008: monkey_fear  --> plot difference, stride3, highlight max
# t0009: human_anger, t0010: human_fear, t0011: monkey_anger, t0012: monkey_fear  --> plot difference, first, highlight max
# t0013: human_anger, t0014: human_fear, t0015: monkey_anger, t0016: monkey_fear  --> plot difference, first, reduce max
# t0017: human_anger  --> plot difference, stride3, reduce max
# t0100: human_anger  --> plot maximum
# t0104: human_anger  --> plot weighted average
config = load_config("norm_base_animate_cnn_response_t0104.json")

# load images
images,_ = load_data(config, train=config["dataset"])

# load model
normbase = NormBase(config,(224,224,3))

# calculate vector and options for plot
if config["plot_option"]=='cnn_output':
    # plot cnn_response
    vector_plot = normbase.evaluate_v4(images, flatten=False)
elif config["plot_option"]=='cnn_output_difference':
    # take difference between response and reference, reference has different options
    response = normbase.evaluate_v4(images, flatten=False)
    if config["difference_option"]=='first':
        reference = response[0,...]
    elif 'stride' in config["difference_option"]:
        stride_length=int(config["difference_option"][6:])
        reference = np.roll(response, shift=stride_length, axis=0)
        reference[:stride_length,...] = response[:stride_length,...]
    elif config["difference_option"]=="reference":
        # calculate and retrieve reference vector from normbase
        raise KeyError('reference not implemented yet')
    else:
        raise KeyError(f'config["difference_option"]={config["difference_option"]} is no valid option')
    vector_plot = response - reference
elif config["plot_option"]=='maximum':
    response = normbase.evaluate_v4(images, flatten=False)
    original_shape = response.shape
    # flatten dimensions that encode x and y of each image and each feature map
    response = np.reshape(response, (response.shape[0],-1,response.shape[3]))
    # init plot vector
    vector_plot = np.zeros(response.shape)
    # compute maximum into flattened dimension
    index_flat_max = np.argmax(response, axis=1)
    # set these indices to 1 in the plot vector
    np.put_along_axis(vector_plot, np.expand_dims(index_flat_max, axis=1), 1, axis=1)
    # reshape to non-flattened shape
    vector_plot = vector_plot.reshape(original_shape)
elif config["plot_option"]=="weighted average":
    response = normbase.evaluate_v4(images, flatten=False)
    original_shape = response.shape
    # flatten dimensions that encode x and y of each image and each feature map
    response = np.reshape(response, (response.shape[0],-1,response.shape[3]))
    # init plot vector
    vector_plot = np.zeros(response.shape)
    # initialize indices with x and y indices
    indices = np.unravel_index(np.arange(response.shape[1]), original_shape[1:3])
    # compute weighted average of the indices, weighted by the neuron activation
    average_x = np.rint(
        np.average(response, axis = 1, weights = indices[0]) * np.sum(indices[0]) / np.sum(response, axis=1) ).astype(int)
    average_y = np.rint(
        np.average(response, axis = 1, weights = indices[1]) * np.sum(indices[1]) / np.sum(response, axis=1) ).astype(int)
    # ravel the indices such that they can be applied to flattened array
    raveled_indices = np.ravel_multi_index((average_x, average_y), original_shape[1:3])
    # set values at these positions to 1
    np.put_along_axis(vector_plot, np.expand_dims(raveled_indices, axis=1), 1, axis=1)
    # reshape to non-flattened shape
    vector_plot = vector_plot.reshape(original_shape)
else:
    raise KeyError(f'config["plot_option"]={config["plot_option"]} is no valid key')

# calculate which feature maps to highlight
try:
    if config["highlight_option"]=="maximum":
        max_map = np.amax(np.abs(vector_plot), axis=(0,1,2)) # max over frames, and within feature map
        highlight = np.flip(np.argsort(max_map)[-25:])
    else:
        raise KeyError(f'config["highlight_option"]={config["highlight_option"]} is no valid option')
except KeyError:
    highlight = None

# reduce feature maps if selected
try:
    if config["plot_reduce"]:
        vector_plot = vector_plot[...,highlight]
        highlight = None
except KeyError:
    pass

# make folder
folder = os.path.join("models/saved", config["save_name"])
if not os.path.exists(folder):
    os.mkdir(folder)

# animation
plot_cnn_output(vector_plot, folder, config["movie_name"]+".mp4", title=config["plot_title"], image=images, video=True, highlight=highlight)

