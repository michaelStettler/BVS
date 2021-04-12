import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')


def plot_it_resp(data, title=None, labels=None, show_legends=True, save_folder=None):
    """

    :param data: input sequence (n_stimuli, n_frames, n_cat)
    :param title:
    :param labels:
    :param show_legends:
    :param save_folder:
    :return:
    """
    n_stimuli = len(data)

    # create figure
    fig = plt.figure(figsize=(15, 10))

    # plot each stimuli
    for i in range(len(data)):
        plt.subplot(n_stimuli, 1, i + 1)
        lineObj = plt.plot(data[i, :, :])
        plt.title(labels[i])
        plt.xlabel("Frames")
        plt.ylabel("IT responses")

    plt.suptitle(title)

    if show_legends:
        plt.legend(lineObj, labels)

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, "plot_it_resp.png"))


def plot_morphing_space(data, title=None, labels=None, show_legends=True, save_folder=None):
    """

    :param data: input sequence (n_stimuli, n_frames, n_cat)
    :param title:
    :param labels:
    :param show_legends:
    :param save_folder:
    :return:
    """
    n_category = np.shape(data)[-1]

    # create figure
    fig = plt.figure(figsize=(15, 10))

    # create meshgrid
    nx, ny = (np.shape(data)[0], np.shape(data)[1])
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    (xv, xy) = np.meshgrid(x, y)

    for i in range(n_category):
        # create sub figures
        plt.subplot(2, 2, i + 1)

        plt.contourf(xv, xy, data[..., i])

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, "plot_morph_space.png"))
