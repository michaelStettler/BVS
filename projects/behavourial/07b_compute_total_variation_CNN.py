import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from projects.behavourial.project_utils import *
from scipy.stats import wasserstein_distance

np.set_printoptions(precision=3, suppress=True)

"""
run: python -m projects.behavourial.07_compute_total_variation_CNN
"""

#%% define computer path
# computer = 'windows'
computer = 'alex'
computer_path, computer_letter = get_computer_path(computer)


#%% declare script parameters
show_plots = True
model_names = ["VGG19_imagenet", "VGG19_imagenet_conv33", "Resnet50v2_imagenet",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
               "CORNet_imagenet"]
conditions = ["human_orig", "monkey_orig"]


def get_pred(model_name, condition):
    path = os.path.join(load_path, f"{model_name}_{condition}_prob_grid.npy")
    print(path)
    preds = np.load(path)

    return preds

for k, model_name in enumerate(model_names):
    for cond, condition in enumerate(conditions):
        load_path = os.path.join(computer_path, 'model_behav_preds')

        #%% load data

        # load behavioural data
        behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
        behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
        behav_data = np.moveaxis(behav_data, 0, -1)
        print("shape behav_data", np.shape(behav_data))

        # load model preds
        preds = get_pred(model_name, condition)
        print(f"finished loading predictions (shape: {np.shape(preds)})")

        # %% compute KL-divergence
        # store computed values
        distances = []

        total_variation = compute_morph_space_total_variation(behav_data, preds)
        print(model_name, total_variation)

        # save values
        path = os.path.join(load_path, f"{model_names[k]}_{conditions[cond]}_total_variation")
        np.save(path, total_variation)
