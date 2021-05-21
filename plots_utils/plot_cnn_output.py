import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import math
import os
from tqdm import tqdm


def plot_cnn_output(output, path, name, title=None, image=None, video=False, verbose=True, highlight=None):
    """
    This function creates a plot of the cnn output.
    :param output: response from cnn (28,28,256)
    :param path: folder where to save plot
    :param name: name of plot
    :param title: title of plot
    :param image: if given, the image is displayed
    :param video: if true an animation is created,
        output shape should then be (#frames, 28,28,256) and image either none or (#frames)
    :param verbose:
    :param highlight: list of int, highlights the feature map at given indices by a red rectangle
    :return:
    """
    if verbose:
        print("start plot_cnn_output of", name)

    num_ft = output.shape[-1]
    n_col = math.ceil(np.sqrt(num_ft))
    n_rows = math.ceil(num_ft / n_col)
    vmin, vmax = np.min(output), np.max(output)
    fig, axs = plt.subplots(n_rows, n_col, figsize=(n_rows+3, n_col))
    # add entry to matrix if there's not enough columns/rows so np.ndenumerate can unpack two values
    if n_col == 1 and n_rows == 1:
        axs = [[axs]]
    elif n_rows == 1:
        axs = [axs]

    plt.subplots_adjust(right=0.75, wspace=0.1, hspace=0.1)

    # add title
    if title is not None:
        fig.suptitle(title)

    ax_colorbar = fig.add_axes([0.83, 0.1, 0.02, 0.65])

    # add image input
    if image is not None:
        ax_image = fig.add_axes([0.78, 0.78, .12, .12])
        if np.max(image) > 1:
            if verbose:
                print("transform image data to RGB [0..1] from [0..255]")
            image = image / 255

    if not os.path.exists(path):
        os.mkdir(path)

    # add highlight
    if highlight is not None:
        for index_highlight in highlight:
            multi_index_highlight = np.unravel_index(index_highlight, (n_rows,n_col))
            if video:
                size_axis = output.shape[1]
            else:
                size_axis = output.shape[0]
            rec = Rectangle((0,0),size_axis,size_axis, fill=False, lw=size_axis*0.15, edgecolor='r')
            rec = axs[multi_index_highlight].add_patch(rec)
            rec.set_clip_on(False)

    update_list = []

    def update(n_frame):
        # if verbose and n_frame >= 0:
        #     print("frame", n_frame)
        if not video:
            data = output
        else:
            data = output[n_frame, ...]

        # loop over all feature maps
        for (i,j), ax in np.ndenumerate(axs):
            n = i*n_col + j
            try:
                # old way, not sure to understand what Tim tried to do here
                # ax.get_images()[0].set_data(data[..., n])
                im = ax.imshow(data[..., n], vmin=vmin, vmax=vmax)
                update_list.append(im)
            except IndexError:
                # old way, not sure to understand what Tim did here
                # im = ax.imshow(data[...,n], vmin=vmin, vmax=vmax)
                # update_list.append(im)
                pass
            ax.axis('off')
        # add input image
        if image is not None:
            if video:
                image_frame = image[n_frame]
            else:
                image_frame = image
            try:
                ax_image.get_images()[0].set_data(image_frame)
            except IndexError:
                im_image = ax_image.imshow(image_frame)
                update_list.append(im_image)
                ax_image.axis('off')

        # display color bar
        if n_frame <= 0:
            cbar = fig.colorbar(im, cax=ax_colorbar)

        return update_list

    if not video:
        update(-1)
        fig.savefig(os.path.join(path, name))
    else:
        def init_func():
            return []
        anim = tqdm(FuncAnimation(fig, update, frames=output.shape[0], init_func=init_func, blit=True, interval=33))
        writergif = animation.PillowWriter(fps=30)
        anim.save(os.path.join(path, name), writer=writergif)

    if verbose:
        print("finished plot_cnn_output of", name)
