import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')


def plot_semantic_histogram(data, title=None, xlabels=None, ylabels=None, save_folder=None):
    # create figure
    plt.figure()
    num_subplot = len(data)
    x = np.arange(np.shape(data)[1])

    add_ylabel = False
    if ylabels is not None:
        add_ylabel = True

    # create a subplot for each layer of interest
    for i, counts in enumerate(data):
        plt.subplot(num_subplot, 1, i + 1)
        plt.bar(x, counts)

        # add value to bar
        for j in range(len(counts)):
            plt.text(j, counts[j] + 5, counts[j], ha='center', va='bottom')

        # add y label
        if add_ylabel:
            h = plt.ylabel(ylabels[i])
            h.set_rotation(0)

    if xlabels is not None:
        plt.xticks(x, labels=xlabels)

    #  set title
    if title is None:
        title = "semantic_histogram.png"
    else:
        title = title

    # set save_folder
    if save_folder is None:
        save_folder = ''
    else:
        save_folder = save_folder

    # save
    plt.savefig(os.path.join(save_folder, title))


def plot_semantic_stacked_bar(data, title=None, xlabels=None, legend=None, save_folder=None):
    # create figure
    plt.figure()
    num_stacked = len(data)
    x = np.arange(np.shape(data)[1])

    # stack bar on top of each other for each concept
    for i, counts in enumerate(data):
        if i == 0:
            plt.bar(x, counts)
        else:
            plt.bar(x, counts, bottom=np.sum(data[:i], axis=0))

    if xlabels is not None:
        plt.xticks(x, labels=xlabels)

    if legend is not None:
        plt.legend(legend)

    #  set title
    if title is None:
        title = "semantic_bar.png"
    else:
        title = title

    # set save_folder
    if save_folder is None:
        save_folder = ''
    else:
        save_folder = save_folder

    # save
    plt.savefig(os.path.join(save_folder, title))



