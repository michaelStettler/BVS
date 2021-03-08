"""
This script demonstrates the extraction of positions in utils.calculate_position.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from utils.load_config import load_config
from utils.load_data import load_data
from utils.calculate_position import calculate_position
from models.NormBase import NormBase

# expressions: human fear and monkey threat
# avatars: humanAvatar and monkeyAvatar
# frame: 0 and 80(?)
# mode: maximum, maximum10, weighted average
# n_feature_map: ???, 1 plot for each map
# plot: columns: picture, response, position mode i --> 5 columns
#       rows: expressions*avatars*frame --> 8 rows

#params
n_feature_map = 54 # seems to be the one with moving eyebrow
frame_neutral_human_expression = 0
frame_expression_human_expression = 68
frame_neutral_monkey_expression = 0
frame_expression_monkey_expression = 52

mode_list = ["maximum", "maximum10", "weighted average"]

# load config
config = load_config("norm_base_calculate_position_demo_t0001.json")
# load data
images_human_fear = load_data(config, train=1)[0]
images_monkey_threat = load_data(config, train=2)[0]
# reduce data
selection_human_expression = [frame_neutral_human_expression, frame_expression_human_expression,
                              150 + frame_neutral_human_expression, 150 + frame_expression_human_expression]
selection_monkey_expression = [frame_neutral_monkey_expression, frame_expression_monkey_expression,
                               150 + frame_neutral_monkey_expression, 150 + frame_expression_monkey_expression]
images = np.concatenate([images_human_fear[selection_human_expression] , images_monkey_threat[selection_monkey_expression]]) # (8, 224, 224, 3)
# calculate response
norm_base = NormBase(config,(224,224,3))
response = norm_base.evaluate_v4(images, flatten=False) # (8, 28, 28, 256)
# reduce response to selected feature map
response=response[...,n_feature_map] # (8, 28, 28)
response = np.expand_dims(response, axis=-1) # (8,28,28,1)

# calculate arrays based on position for different modes
position_array_list = []
position_xy_list = []
for i,mode in enumerate(mode_list):
    position_array_list.append(calculate_position(response, mode=mode, return_mode="array"))
    position_xy_list.append(calculate_position(response, mode=mode, return_mode="xy float"))

# create plot
fig, axs = plt.subplots(8, 2 + len(position_array_list), figsize=[10, 10])
for _, ax in np.ndenumerate(axs):
    #ax.axis('off')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
for ax, col in zip(axs[0], ["picture", "response"]+mode_list):
    ax.set_title(col)

# plot images
images = images/255 # transform into correct range for imshow()
for i,image in enumerate(images):
    axs[i,0].imshow(image)

# plot plain response
for i, resp in enumerate(response):
    axs[i,1].imshow(resp)

# plot positions
for i, position in enumerate(position_array_list):
    for j, pos in enumerate(position):
        axs[j, 2+i].imshow(pos)
        axs[j, 2+i].set_xlabel(f'{position_xy_list[i][0][j,0] :.1f}, {position_xy_list[i][1][j,0] :.1f}')
        #axs[j, 2+i].set_xlabel(position_xy_list[i][1][j,0])
        #axs[j, 2+i].set_ylabel(position_xy_list[i][0][j,0])

plt.savefig("models/saved/calculate_position_demo/calculate_position_demo.png")