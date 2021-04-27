import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')


def plot_expressivity_level(config, expr_neurons, title=None, save_folder=None, show_legends=True):
    """
    helper function to plot the expression neuron responses from the models by condition and in function of the
    expressivity level

    :param config:
    :param expr_neurons:
    :param title:
    :param save_folder:
    :param show_legends:
    :return:
    """
    print("shape expr_neurons", np.shape(expr_neurons))

    # ----------------------------------------------------------------------------------------------------------------
    # plot raw expression neurons per category
    plt.figure()
    for c in range(config['n_category']):
        # create subplot
        plt.subplot(config['n_category'], 1, c+1)

        # get expressivity levels per condition
        idx = 4 * c
        plt.plot(expr_neurons[idx, :, c])  # 0.25
        plt.plot(expr_neurons[idx + 1, :, c])  # 0.5
        plt.plot(expr_neurons[idx + 2, :, c])  # 0.75
        plt.plot(expr_neurons[idx + 3, :, c])  # 1.0

    if show_legends:
        plt.legend()

    fig_title = 'raw_expressivity_level.png'
    if title is not None:
        fig_title = title + '_' + fig_title

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, fig_title))
    else:
        plt.savefig(fig_title)

    # ----------------------------------------------------------------------------------------------------------------
    # plot expression activity in function of the expressivity level
    plt.figure()
    for c in range(config['n_category']):
        # create subplot
        plt.subplot(config['n_category'], 1, c+1)

        # get expressivity levels per condition
        idx = 4 * c
        activity = expr_neurons[idx:idx + 4, :, c]
        activity = np.sum(activity, axis=1)

        plt.plot(activity)

    if show_legends:
        plt.legend()

    fig_title = 'expressivity_level.png'
    if title is not None:
        fig_title = title + '_' + fig_title

    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, fig_title))
    else:
        plt.savefig(fig_title)
