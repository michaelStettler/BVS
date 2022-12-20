import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.06_morph_space_comparison
"""

#%% define computer path
computer = 'windows'
if 'windows' in computer:
    computer_path = 'D:/Dataset/MorphingSpace'
    computer_letter = 'w'
elif 'mac' in computer:
    computer_path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace'
    computer_letter = 'm'

#%% declare script parameters
condition = "human_orig"
norm_type = "categorical"
CNN_model = 'ResNet50v2_imagenet'

#%% construct path
behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
NRE_path = os.path.join(computer_path, 'saved_lmks_pos', condition)
CNN_path = os.path.join(computer_path, 'output', condition)

#%% load data
behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
behav_data = np.moveaxis(behav_data, 0, -1)
NRE_pred = np.load(os.path.join(NRE_path, "prob_grid_{}.npy".format(norm_type)))
pred_data = NRE_pred[..., 1:]
CNN_pred = np.load(os.path.join(CNN_path, "prob_grid_{}_{}.npy".format(condition, CNN_model)))

print("shape behav_data", np.shape(behav_data))
print("shape NRE_pred", np.shape(NRE_pred))
print("shape CNN_pred", np.shape(CNN_pred))