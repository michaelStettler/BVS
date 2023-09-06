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
model_names = ["VGG19_imagenet", "VGG19_imagenet_conv33", "Resnet50v2_imagenet",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
               "CORNet_imagenet"]
conditions = ["human_orig", "monkey_orig"]


def get_pred(model_name, condition):
    path = os.path.join(load_path, f"{model_name}_{condition}_prob_grid.npy")
    print(path)
    preds = np.load(path)

    return preds

for model_name in model_names:
    for cond, condition in enumerate(conditions):
        load_path = os.path.join(computer_path, 'model_behav_preds')

        #%% load data

        # load behavioural data
        behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
        behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
        behav_data = np.moveaxis(behav_data, 0, -1)
        print("shape behav_data", np.shape(behav_data))

        # load model preds
        predictions = []
        for model_name in model_names:
            preds = get_pred(model_name, condition)
            predictions.append(preds)
        predictions = np.array(predictions)
        print(f"finished loading predictions (shape: {np.shape(predictions)})")

        # ### Use synthetic data until real is available
        # behav_data = np.random.randn(5, 5, 4)
        # NRE_pred = np.abs(np.random.randn(*behav_data.shape))
        # NRE_pred /= np.sum(NRE_pred, axis=-1, keepdims=True)
        # CNN_pred = np.abs(np.random.randn(*behav_data.shape))
        # CNN_pred /= np.sum(CNN_pred, axis=-1, keepdims=True)
        # predictions = [NRE_pred, CNN_pred]


        # %% compute KL-divergence

        # store computed values
        kl_divergences = []

        for pred in predictions:
            kl_div = compute_morph_space_KL_div(behav_data, pred)
            kl_divergences.append(kl_div)
        kl_divergences = np.array(kl_divergences)
        print(f"finished computing KL div (shape: {np.shape(kl_divergences)})")

        # don't save until real data is available
        # save values
        for k, kl_div in enumerate(kl_divergences):
            path = os.path.join(load_path, f"{model_names[k]}_{conditions[cond]}_KL_div")
            np.save(path, kl_div)
