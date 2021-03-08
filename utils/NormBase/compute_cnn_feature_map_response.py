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
from plots_utils.plot_cnn_output import plot_cnn_output
from models.NormBase import NormBase
from utils.calculate_position import calculate_position
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
# t0108: human_anger  --> plot 10 biggest values (maximum10)
config = load_config("norm_base_animate_cnn_response_t0108.json")

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
else:
    try:
        # config["plot_option"]: "maximum", "maximum{int}", "weighted average"
        vector_plot = calculate_position(response=normbase.evaluate_v4(images, flatten=False),
                                         mode=config["plot_option"],
                                         return_mode="array")
    except KeyError:
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
folder = os.path.join("../../models/saved", config["save_name"])
if not os.path.exists(folder):
    os.mkdir(folder)

# animation
plot_cnn_output(vector_plot, folder, config["movie_name"]+".mp4", title=config["plot_title"], image=images, video=True, highlight=highlight)

