import os
import numpy as np
import matplotlib.pyplot as plt


def plot_ft_map_pos(pos, fig_name=None, path=None, titles=None, color_seq=None):

    # create color sequence
    if color_seq is None:
        color_seq = np.arange(len(pos))
    else:
        color_seq = color_seq

    # retrieve parameters
    num_subplot = np.shape(pos)[1]
    n_column = int(np.sqrt(num_subplot))
    n_rows = np.ceil(num_subplot / n_column).astype(np.int)

    if fig_name is not None:
        fig_name = fig_name
    else:
        fig_name = 'feature_map_positions.png'

    if path is not None:
        path = path
    else:
        path = ''

    # create figure
    plt.figure(figsize=(n_rows*4, n_column*4))

    for i in range(num_subplot):
        plt.subplot(n_rows, n_column, i + 1)
        plt.scatter(pos[:, i, 1], pos[:, i, 0], c=color_seq)
        # plt.xlim(13.5, 14.0)
        # plt.ylim(11.7, 12.2)
        plt.colorbar()

        # add sub title if given
        if titles is not None:
            plt.title(titles[i])

    plt.savefig(os.path.join(path, fig_name))