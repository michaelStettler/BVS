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



