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
    n_stimuli = np.shape(data)[0]
    n_category = np.shape(data)[-1]

    # create figure
    fig = plt.figure(figsize=(15, 10))

    # fear human
    map_table = [0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24]
    # create meshgrid
    nx, ny = (5, 5)
    print("nx", nx, "ny", ny)
    x = np.linspace(1, 0, nx)
    y = np.linspace(1, 0, ny)
    (xv, xy) = np.meshgrid(x, y)

    for i in range(n_category - 1):  # remove the neutral category
        # create sub figures
        plt.subplot(2, 2, i + 1)

        z = np.zeros(25)
        for v in range(n_stimuli):
            z[map_table[v]] = np.amax(data[v, :, i + 1]) / np.amax(data[v])
        z = np.reshape(z, (5, 5))

        plt.contourf(xv, xy, z)

    # set figure title
    fig_title = 'morphing_space.png'
    if title is not None:
        fig_title = title + '_' + fig_title

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, fig_title))
