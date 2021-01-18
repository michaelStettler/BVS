import numpy as np
import matplotlib.pyplot as plt
import math
import os

def plot_cnn_output(output, path, name, title=None, image=None):
    """
    This function creates a plot of the cnn output.
    :param output: response from cnn (28,28,256)
    :param path: folder where to save plot
    :param name: name of plot
    :param title: title of plot
    :param image: if given, the image is displayed
    :return:
    """
    n_rows = math.ceil(np.sqrt(output.shape[-1]))
    n_col = output.shape[-1] //n_rows
    vmin, vmax = np.min(output), np.max(output)
    fig, axs = plt.subplots(n_rows, n_col, figsize=(n_rows+3,n_col))
    plt.subplots_adjust(right=0.75, wspace=0.1, hspace=0.1)
    if not title is None: fig.suptitle(title)
    for (i,j), ax in np.ndenumerate(axs):
        n = i*n_col + j
        im = ax.imshow(output[...,n], vmin=vmin, vmax=vmax)
        ax.axis('off')
    if not image is None:
        ax_image = fig.add_axes([0.78,0.78,.12,.12])
        ax_image.imshow(image)
        ax_image.axis('off')
    ax_colorbar = fig.add_axes([0.83, 0.1, 0.02, 0.65])
    cbar = fig.colorbar(im, cax=ax_colorbar)

    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(os.path.join(path, name))