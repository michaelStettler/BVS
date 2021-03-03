"""
2021/01/12
This script compares the first PCA component of the human and the monkey dataset.
2021/01/18
This result is basically meaningless. (better compare tuning vectors)
Use utils.plot_cnn_output.py instead for much better plots_utils.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

# setting
retrain = False

# param, order: human, monkey, both
config_names = ["norm_base_investigate_layer_t0001.json",
                "norm_base_investigate_layer_t0002.json","norm_base_investigate_layer_t0003.json"]

# load config
configs = []
for config_name in config_names:
    configs.append(load_config(config_name))

# train models/load model
norm_base_list = []
for config in configs:
    try:
        if retrain:
            raise IOError("retrain = True")
        norm_base = NormBase(config, input_shape=(224, 224, 3), save_name=config["sub_folder"])
    except IOError:
        norm_base = NormBase(config, input_shape=(224,224,3))
        norm_base.fit(load_data(config, train=config["train_dim_ref_tun_ref"][0]), fit_dim_red=True, fit_ref=False,
                      fit_tun=False)
        norm_base.save_model(config, config["sub_folder"])
    norm_base_list.append(norm_base)

# extract PCA
pca1_human = norm_base_list[0].pca.components_[0]
pca1_monkey = norm_base_list[1].pca.components_[0]


def plot_pca(component, folder, figname):
    if isinstance(component,list):
        is_list = True
        vectors = component
    else:
        vectors = [component]

    y_offset = 0
    for component in vectors:
        y_offset = max(y_offset, 1.4*np.max(component))
    plot_list = []
    for component in vectors:
        component.shape = (28,28,256)
        component = np.abs(component)
        n_rows, n_columns = 16, 16
        x_offset = 100
        plot_x = np.zeros(component.size)
        plot_y = np.zeros(plot_x.size)
        for n_map in range(256):
            i = n_map % n_rows
            j = n_map // n_rows
            idx = np.arange(784) + n_map * 784
            plot_x[idx] = np.arange(784) + i * (x_offset + 784)
            plot_y[idx] = component[..., n_map].flatten() + (n_rows - 1 - j) * y_offset
        plot_list.append((plot_x,plot_y))

    fig2, ax = plt.subplots(1, 1, figsize=(15, 10))
    for i, plot_x_y in enumerate(plot_list):
        ax.scatter(plot_x_y[0], plot_x_y[1], s=1, alpha=0.5, label='vector {}'.format(i))
    plt.savefig(os.path.join("../../models/saved", folder, figname))

plot_pca(pca1_human, configs[0]['save_name'], "pca1_human.png")
plot_pca(pca1_monkey, configs[1]['save_name'], "pca1_monkey.png")
plot_pca([pca1_human, pca1_monkey], configs[1]['save_name'], "pca1_together.png")

# heat map of features that contribute to the PCA components
# 16x16 subplot with each plot: 28x28 pixels
# is this even possible? (200k features, screen: 2000k pixels)