"""
2021/01/04
This script has 2 novelties:
- model is trained in separate steps (specified in config)
- data is plotted using the projection to 2D
"""

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

from utils.load_config import load_config
from models.NormBase import NormBase
from utils.load_data import load_data

config = load_config("norm_base_plotDirections_t0009.json")
save_name = config["sub_folder"]
retrain = False

# model
try:
    if retrain:
        raise IOError("retrain = True")
    norm_base = NormBase(config, input_shape=(224, 224, 3), save_name=save_name)
except IOError:
    norm_base = NormBase(config, input_shape=(224,224,3))

    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][0]), fit_dim_red=True, fit_ref=False,
                  fit_tun=False)
    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][1]), fit_dim_red=False, fit_ref=True,
                  fit_tun=False)
    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][2]), fit_dim_red=False, fit_ref=False,
                  fit_tun=True)
    norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][3]), fit_dim_red=False, fit_ref=True,
                  fit_tun=False)
    norm_base.save_model(config, save_name)

# test
data_test = load_data(config, train=config["data_test"])

projection, labels = norm_base.projection_tuning(data_test)

# calculate constant activation lines
x_lines, lines = norm_base.line_constant_activation()

# plot
fig, axs = plt.subplots(1, projection.shape[0]-1, figsize=(5*(projection.shape[0]-1), 5), sharey=True, sharex=True)
fig.suptitle("Projection of difference vector preserving 2-norm and scalar product")
for n_plot, ax in enumerate(axs):
    category = n_plot +1
    ax.set_title(["Neutral","Expression 1", "Expression 2"][category])
    point_reference = ax.plot(0,0, 'kx', label="reference vector")
    arrow_tuning = ax.arrow(0,0,1,0, label="tuning vector", color="black", head_width=0.05, length_includes_head=True)
    #an_tuning = ax.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops={'arrowstyle': '->'}, va='center')
    scatter = ax.scatter(projection[category,:,0], projection[category,:,1], s = 1, c=labels)

    # set axis limits
    ax.set_ylim(ymin=-0.02, ymax=ax.get_ylim()[1]*1.2)
    x_max = max(max(np.abs(ax.get_xlim())), 1)
    ax.set_xlim(xmin=-x_max, xmax=x_max)

    #plot lines of constant activation
    line_activation = ax.plot(x_lines, lines, color="k", linewidth=0.5)
# legend
handles_scatter, labels_scatter = scatter.legend_elements()
def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p
fig.legend([point_reference[0], arrow_tuning]+handles_scatter+[line_activation[0]],
           ["reference vector", "tuning vector","Neutral","Expression 1","Expression 2","constant activation"],
           handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
           loc="upper right", borderaxespad=0.1)

plt.savefig(os.path.join("models/saved", config['save_name'], save_name, "scatter.png"))