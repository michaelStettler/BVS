import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.06_morph_space_comparison
"""

windows_path = 'D:/Dataset/MorphingSpace'
# mac_path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace'

condition = "human_orig"
morph_space =
behav_path = "morphing_psychophysics_result"
model_path = "saved_lmks_pos"

behavioural_path = os.path.join(windows_path, 'morphing_psychophysics_result')
NRE_path = os.path.join(windows_path, '')
CNN_path = os.path.join(windows_path, '')

if condition == "human_orig":
    behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
    behav_data = np.moveaxis(behav_data, 0, -1)
    pred_data = np.load(os.path.join(morph_space, model_path, "prob_grid_{}.npy".format(norm_type)))
    pred_data = pred_data[..., 1:]
    cat_data = np.load(os.path.join(morph_space, model_path, "cat_grid_{}.npy".format(norm_type)))
    cat_data = cat_data[..., 1:]