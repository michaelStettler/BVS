import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from projects.behavourial.project_utils import *
from scipy.stats import wilcoxon
from scipy.special import softmax
from G_compute_KL_div_CNN import *

'''
###
Categories:
0: Human Anger
1: Human Fear
2: Monkey Anger
3: Monkey Fear

Each row of the grid grid[i, :, :] yields the transition from Anger to fear
for a fixed morph of human/monkey
###
'''


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
plot_format = 'svg'
# model_names = ["NRE_individual_static", "NRE_individual_dynamic",
#                "NRE_frobenius_static", "NRE_frobenius_dynamic",
#                "VGG19_imagenet", "VGG19_imagenet_conv33", "Resnet50v2_imagenet",
#                "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
#                "CORNet_imagenet"]
model_names = ["NRE_individual_static", "NRE_individual_dynamic",
               "NRE_frobenius_static", "NRE_frobenius_dynamic",
               "VGG19_imagenet", "Resnet50v2_imagenet",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
               "CORNet_imagenet"]

### names for poster
model_names = ["NRE_frobenius_static", "NRE_frobenius_dynamic",
               "VGG19_imagenet", "Resnet50v2_imagenet",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
               "CORNet_imagenet"]

model_names = ['NRE_frobenius_static', "Resnet50v2_imagenet"]
###
condition = "human_orig"


def get_pred(model_name, condition):
    path = os.path.join(load_path, f"{model_name}_{condition}_prob_grid.npy")
    preds = np.load(path)
    return preds

def get_anger_values(array):
    anger_values = np.zeros(array.shape[0])
    for j in range(array.shape[0]):
        probs = array[j, :]
        anger = probs[0] + probs[2]
        fear = probs[1] + probs[3]
        anger = anger / (anger + fear)
        anger_values[j] = anger
    return anger_values

def plot_lines(preds, color):
    for i in range(5):
        anger_values = get_anger_values(preds[i, :, :])
        plt.plot(anger_values, c=color)


def softmax(array, beta):
    a = np.exp(array * beta)
    denom = np.sum(a, axis=-1, keepdims=True)
    out = a / denom
    return out




pred_dict = {}

for k, model_name in enumerate(model_names):
    pred_model_dict = {}
    kl_model_dict = {}
    var_model_dict = {}
    entropy_model_dict = {}
    entropy_diff_model_dict = {}

    load_path = os.path.join(computer_path, 'model_behav_preds')

    # load data
    # load behavioural data
    behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
    if condition == "human_orig":
        behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
    elif condition == "monkey_orig":
        behav_data = np.load(os.path.join(behavioural_path, "monkey_avatar_orig.npy"))

    behav_data = np.moveaxis(behav_data, 0, -1)

    # load model preds
    preds = get_pred(model_name, condition)

    pred_dict[model_name] = preds

nre_preds = pred_dict['NRE_frobenius_static']
cnn_preds = pred_dict['Resnet50v2_imagenet']
print('------------------------------')
# print('no softmax:', np.sum(compute_morph_space_KL_div(behav_data, nre_preds)))
# for beta in range(1, 20):
#     print(beta, np.sum(compute_morph_space_KL_div(behav_data, softmax(nre_preds, beta=beta))))
# print(nre_preds[0, 0, :], nre_preds[0, 4, :], nre_preds[4, 0, :], nre_preds[4, 4, :])
# nre_preds = softmax(nre_preds, beta=5)
# # print('KL:', np.sum(compute_morph_space_KL_div(behav_data, nre_preds)))
# pred_dict['NRE_frobenius_static'] = nre_preds

print(behav_data[0, :, :])
print(cnn_preds[0, :, :])

# print(behav_data[0, 0, :], behav_data[0, 4, :], behav_data[4, 0, :], behav_data[4, 4, :])
# print(nre_preds[0, 0, :], nre_preds[0, 4, :], nre_preds[4, 0, :], nre_preds[4, 4, :])



plot_lines(behav_data, color='black')
colors = ['blue', 'green']
for model_name, color in zip(model_names, colors):
    plot_lines(pred_dict[model_name], color=color)
plt.show()