import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from projects.behavourial.project_utils import *


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.07_compute_KL_div_CNN
"""

#%% define computer path
# computer = 'windows'
computer = 'alex'

computer_path, computer_letter = get_computer_path(computer)


#%% declare script parameters
show_plots = True
model_names = ["NRE_individual_static", "NRE_individual_dynamic",
               "NRE_frobenius_static", "NRE_frobenius_dynamic",
               "VGG19_imagenet", "VGG19_imagenet_conv33", "Resnet50v2_imagenet",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
               "CORNet_imagenet"]
conditions = ["human_orig", "monkey_orig"]


# model_names = ['NRE_frobenius_dynamic', 'VGG19_imagenet']
# model_names = ['Resnet50v2_imagenet']
# conditions = ["human_orig"]

def compute_morph_space_KL_div(P, Q):
    log = np.log(P / Q)
    return np.sum(P * log, axis=-1)


def compute_morph_space_total_variation(P, Q):
    return 0.5 * np.sum(np.abs(P - Q), axis=-1)


def compute_entropy(P):
    log = np.log(P)
    return - np.sum(P * log, axis=-1)


def get_pred(model_name, condition):
    path = os.path.join(load_path, f"{model_name}_{condition}_prob_grid.npy")
    print(path)
    preds = np.load(path)
    return preds



kl_divergences = {}
total_variations = {}
entropies = {}
for k, model_name in enumerate(model_names):
    kl_model_dict = {}
    var_model_dict = {}
    entropy_model_dict = {}
    for cond, condition in enumerate(conditions):
        load_path = os.path.join(computer_path, 'model_behav_preds')

        # load data
        # load behavioural data
        behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
        if condition == "human_orig":
            behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
        elif condition == "monkey_orig":
            behav_data = np.load(os.path.join(behavioural_path, "monkey_avatar_orig.npy"))

        behav_data = np.moveaxis(behav_data, 0, -1)
        print("shape behav_data", np.shape(behav_data))
        if k == 0:  # Get entropy of behavioural data on first iteration
            if cond == 0:
                behav_entropy = {}
            behav_entropy[condition] = compute_entropy(behav_data)
            if cond == 1:
                entropies['behavioural'] = behav_entropy


        # load model preds
        preds = get_pred(model_name, condition)

        # compute KL-divergence
        kl_div = compute_morph_space_KL_div(behav_data, preds)
        var = compute_morph_space_total_variation(behav_data, preds)
        entropy = compute_entropy(preds)

        kl_model_dict[condition] = kl_div
        var_model_dict[condition] = var
        entropy_model_dict[condition] = entropy

        print(model_name, np.sum(kl_div), np.sum(var))
        print('kl:', kl_div)
        print('var:', var)
        print('--------------------')
        # save values
        np.save(os.path.join(load_path, f"{model_names[k]}_{conditions[cond]}_KL_div"), kl_div)
        np.save(os.path.join(load_path, f"{model_names[k]}_{conditions[cond]}_total_variation"), var)
    kl_divergences[model_name] = kl_model_dict
    total_variations[model_name] = var_model_dict
    entropies[model_name] = entropy_model_dict


#%%

labels = ["NRE-indi-S", "NRE-indi-D",
               "NRE-cat-S", "NRE-cat-D",
               "VGG19-IM", "VGG19-IM-conv33", "Resnet50v2-IM",
               "VGG19-AN", "ResNet50v2-AN", "CORNet-AN",
               "CORNet-IM"]
colors = ['#FE938C', '#4281A4']

def make_bar_plot(data_dict, labels, colors):
    fig, ax = plt.subplots()
    x = np.arange(len(data_dict))
    width = 0.25
    # plot each condition
    for i, model_name in enumerate(model_names):
        for c, condition in enumerate(conditions):
            offset = width * c
            rects = plt.bar(x[i] + offset, np.sum(data_dict[model_name][condition]),
                            width, label=condition, color=colors[c])
            if i == 0:
                ax.legend()
    ax.set_xticks(x, labels)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

make_bar_plot(kl_divergences, labels, colors)
make_bar_plot(total_variations, labels, colors)
model_names.insert(0, 'behavioural')
labels.insert(0, 'Psychophysics')
make_bar_plot(entropies, labels, colors)

nre_entropy = entropies['NRE_individual_dynamic']['human_orig']
cnn_entropy = entropies['CORNet_imagenet']['human_orig']
behav_entropy = entropies['behavioural']['human_orig']
print(nre_entropy)
print(cnn_entropy)
print(behav_entropy)
max_entropy = np.maximum(np.max(nre_entropy), np.max(cnn_entropy))
max_entropy = np.maximum(np.max(behav_entropy), max_entropy)
print('Max entropy:', max_entropy)
plt.imshow(nre_entropy, vmin=0, vmax=max_entropy)
plt.colorbar()
plt.show()
plt.imshow(cnn_entropy, vmin=0, vmax=max_entropy)
plt.colorbar()
plt.show()
plt.imshow(behav_entropy, vmin=0, vmax=max_entropy)
plt.colorbar()
plt.show()