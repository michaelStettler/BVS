import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_data import load_data

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.05_morph_space_with_CNN
"""

#%% import config
# config_path = 'BH_05_morph_space_with_CNN_VGG19_imagenet_w0001.json'              # OK
config_path = 'BH_05_morph_space_with_CNN_VGG19_imagenet_conv33_w0001.json'         # OK
# config_path = 'BH_05_morph_space_with_CNN_VGG19_affectnet_w0001.json'             # OK
# config_path = 'BH_05_morph_space_with_CNN_ResNet50v2_imagenet_w0001.json'         # OK
# config_path = 'BH_05_morph_space_with_CNN_ResNet50v2_affectnet_w0001.json'        # OK
# config_path = 'BH_05_morph_space_with_CNN_CORNet_affectnet_w0001.json'            #
# load config
config = load_config(config_path, path='configs/behavourial')

#%% declare script variables
# occluded and orignial are the same for this pipeline as we do not have any landmark on the ears
show_plot = True
cond = 0
conditions = ["human_orig", "monkey_orig"]
condition = conditions[cond]
morph_csv = [os.path.join(config['directory'], "morphing_space_human_orig.csv"),
             os.path.join(config['directory'], "morphing_space_monkey_orig.csv")]

# edit dictionary for single condition type
if cond is not None:
    config["train_csv"] = morph_csv[cond]
    config["condition"] = condition
    if "human" in condition[cond]:
        config["avatar_types"] = ["human"]
    else:
        config["avatar_types"] = ["monkey"]

# create directory
save_path = os.path.join(config["directory"], config["load_directory"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

#%% import data
morph_data = load_data(config, get_raw=True)
print("-- Data loaded --")
print("len train_data[0]", len(morph_data[0]))
print()

#%% load model
model = tf.keras.models.load_model(os.path.join(config["load_directory"], config["model_name"]))

#%%
preds = model.predict(morph_data[0])
print("shape preds", np.shape(preds))

#%%
def print_morph_space(amax_ms_grid=None, cat_grid=None, prob_grid=None,
                      title=None, show_plot=True, save=True, save_path=None):
    if amax_ms_grid is not None:
        fig, axs = plt.subplots(2, 2)
        pcm1 = axs[0, 0].imshow(amax_ms_grid[..., 0], cmap='viridis', interpolation='nearest')
        fig.colorbar(pcm1, ax=axs[0, 0])
        pcm2 = axs[0, 1].imshow(amax_ms_grid[..., 1], cmap='viridis', interpolation='nearest')
        fig.colorbar(pcm2, ax=axs[0, 1])
        pcm3 = axs[1, 0].imshow(amax_ms_grid[..., 2], cmap='viridis', interpolation='nearest')
        fig.colorbar(pcm3, ax=axs[1, 0])
        pcm4 = axs[1, 1].imshow(amax_ms_grid[..., 3], cmap='viridis', interpolation='nearest')
        fig.colorbar(pcm4, ax=axs[1, 1])

        if save:
            if save_path is None:
                plt.savefig(f"{title}_morph_space_read_out_values.jpeg")
            else:
                plt.savefig(os.path.join(save_path, f"{title}_morph_space_read_out_values.jpeg"))

    if cat_grid is not None:
        # print category grid
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(cat_grid[..., 0], cmap='hot', interpolation='nearest')
        axs[0, 1].imshow(cat_grid[..., 1], cmap='hot', interpolation='nearest')
        axs[1, 0].imshow(cat_grid[..., 2], cmap='hot', interpolation='nearest')
        axs[1, 1].imshow(cat_grid[..., 3], cmap='hot', interpolation='nearest')

        if save:
            if save_path is None:
                plt.savefig(f"{title}_morph_space_categories_values.jpeg")
            else:
                plt.savefig(os.path.join(save_path, f"{title}_morph_space_categories_values.jpeg"))

    # print probability grid
    if cat_grid is not None:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(prob_grid[..., 0], cmap='viridis', interpolation='nearest')
        axs[0, 1].imshow(prob_grid[..., 1], cmap='viridis', interpolation='nearest')
        axs[1, 0].imshow(prob_grid[..., 2], cmap='viridis', interpolation='nearest')
        pcm = axs[1, 1].imshow(prob_grid[..., 3], cmap='viridis', interpolation='nearest')

        fig.colorbar(pcm, ax=axs[:, 1], shrink=0.7)

        if save:
            if save_path is None:
                plt.savefig(f"{title}_morph_space_probabilities_values.jpeg")
            else:
                plt.savefig(os.path.join(save_path, f"{title}_morph_space_probabilities_values.jpeg"))

        if show_plot:
            plt.show()


morph_space_data = np.reshape(preds, [25, 150, -1])
print("shape morph_space_data", np.shape(morph_space_data))

# get max values for each video and category
amax_ms = np.amax(morph_space_data, axis=1)
print("shape amax_ms", np.shape(amax_ms))
print(amax_ms)

# make into grid
amax_ms_grid = np.reshape(amax_ms, [5, 5, -1])
amax_ms_grid = amax_ms_grid[..., 1:]
print("shape amax_ms_grid", np.shape(amax_ms_grid))

cat_grid = np.zeros((5, 5, 4))
prob_grid = np.zeros((5, 5, 4))
for i in range(np.shape(amax_ms_grid)[0]):
    for j in range(np.shape(amax_ms_grid)[0]):
        x = amax_ms_grid[i, j]  # discard neutral
        cat_grid[i, j, np.argmax(x)] = 1
        prob_grid[i, j] = np.exp(x) / sum(np.exp(x))

print("model saved in:", save_path)
title = config['project'] + "_" + condition
np.save(os.path.join(save_path, f"{title}_amax_ms_grid"), amax_ms_grid)
np.save(os.path.join(save_path, f"{title}_cat_grid"), cat_grid)
np.save(os.path.join(save_path, f"{title}_prob_grid"), prob_grid)

# print morphing space
print_morph_space(amax_ms_grid, cat_grid, prob_grid, show_plot=show_plot, title=title, save=True)
