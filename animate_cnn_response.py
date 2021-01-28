"""
2021/01/20
    This script creates an animation of the cnn response to a moving face.
    for 150 frames this takes approximately 10 minutes
2021/01/28
    adding options what to plot
"""

import os
import numpy as np

from utils.load_config import load_config
from utils.load_data import load_data
from utils.plot_cnn_output import plot_cnn_output
from models.NormBase import NormBase

# load config
# t0001: human_anger, t0002: human_fear, t0003: monkey_anger, t0004: monkey_fear  --> plot cnn_output
# t0005: human_anger, t0006: human_fear, t0007: monkey_anger, t0008: monkey_fear  --> plot cnn_output, highlight most variance
config = load_config("norm_base_animate_cnn_response_t0004.json")

# load images
images,_ = load_data(config, train=config["dataset"])

# load model
normbase = NormBase(config,(224,224,3))

# calculate vector and options for plot
highlight = None
if config["plot_option"]=='cnn_output':
    # plot cnn_response
    vector_plot = normbase.evaluate_v4(images, flatten=False)
elif config["plot_option"]=='cnn_output_highlight_variance':
    # plot cnn_response
    vector_plot = normbase.evaluate_v4(images, flatten=False)
    highlight= [1,2,3] #TODO calculate
else:
    raise KeyError(f'config["plot_option"]={config["plot_option"]} is no valid key')


# make folder
folder = os.path.join("models/saved", config["save_name"])
if not os.path.exists(folder):
    os.mkdir(folder)

# animation
plot_cnn_output(vector_plot, folder, config["movie_name"]+".mp4", title=config["plot_title"], image=images, video=True, highlight=highlight)

