import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from plots_utils.plot_ft_map_pos import _set_fig_name
from plots_utils.plot_ft_map_pos import _set_save_folder


def plot_tuning_signatures(data, ref_tuning=None, fig_name=None, save_folder=None):
    print("prout")
    print("shape data", np.shape(data))

    # set images name
    images_name = _set_fig_name(fig_name, 'signature_tuning.png')

    # set save folder
    save_folder = _set_save_folder(save_folder, '')

    # create colors
    colors = cm.rainbow(np.linspace(0, 1, len(data)))

    # create figure
    plt.figure(figsize=(5, 5), dpi=600)

    for i, d, c in zip(np.arange(len(data)), data, colors):
        plt.scatter(d[1], -d[0], color=c)
        plt.arrow(0, 0, d[1], -d[0], color=c, linewidth=1)
        plt.text(d[1] * 1.1, -d[0] * 1.1, str(i), color=c)

        if ref_tuning is not None:
            plt.scatter(ref_tuning[i, 1], -ref_tuning[i, 0], color=c)
            plt.arrow(0, 0, ref_tuning[i, 1], -ref_tuning[i, 0], color=c, linewidth=1, linestyle=':')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.savefig(os.path.join(save_folder, images_name))