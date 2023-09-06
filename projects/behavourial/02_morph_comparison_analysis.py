import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from projects.behavourial.project_utils import *


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.02_morph_comparison_analysis
"""

show_plots = False
conditions = ["human_orig", "monkey_orig"]
cond = 0
condition = conditions[cond]
# norm_type = "individual"
norm_type = "frobenius"
use_dynamic = True

# declare saving and loading path
morph_space = "/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace"
behav_path = "morphing_psychophysics_result"
saved_path = "model_preds"
load_path = os.path.join(morph_space, saved_path)
save_path = load_path

modality = 'static'
if use_dynamic:
    modality = 'dynamic'

# load data
if condition == "human_orig":
    behav_data = np.load(os.path.join(morph_space, behav_path, "human_avatar_orig.npy"))
elif condition == "monkey_orig":
    behav_data = np.load(os.path.join(morph_space, behav_path, "monkey_avatar_orig.npy"))

behav_data = np.moveaxis(behav_data, 0, -1)
pred_data = np.load(os.path.join(load_path, f"NRE_{norm_type}_{modality}_{condition}_prob_grid.npy"))
cat_data = np.load(os.path.join(load_path, f"NRE_{norm_type}_{modality}_{condition}_cat_grid.npy"))

print("shape behav_data", np.shape(behav_data))
print("max behav_data", np.amax(behav_data))
print("shape pred_data", np.shape(pred_data))
print("max pred_data", np.amax(pred_data))
print("shape cat_data", np.shape(cat_data))
print(pred_data[..., 3])
print("pred_data[0, 0]", pred_data[0, 0])
sum_prob = np.sum(pred_data, axis=2)
print("sum prob", np.shape(sum_prob))
print(sum_prob)
print()


div = compute_morph_space_KL_div(behav_data, pred_data)
print("shape div", np.shape(div))
print("sum div", np.sum(div))
print("mean: {}, variance: {}".format(np.mean(div), np.std(div)))

# create figure
titles = ["behaviour", "model", "KL-div"]
fig = plt.figure(figsize=(9, 3))
outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)

for i in range(3):
    # set outer titles
    ax = plt.Subplot(fig, outer[i])
    ax.set_title(titles[i])
    ax.axis('off')
    fig.add_subplot(ax)

    if i == 0:
        data = behav_data
        n_col = 2
        n_row = 2
    elif i == 1:
        data = pred_data
        n_col = 2
        n_row = 2
    elif i == 2:
        data = div
        n_col = 1
        n_row = 1

    inner = gridspec.GridSpecFromSubplotSpec(n_row, n_col,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    if i < 2:
        for j in range(4):
            ax = plt.Subplot(fig, inner[j])
            # ax.imshow(data[..., j], cmap='viridis', interpolation='bilinear', vmax=1)
            ax.imshow(data[..., j], cmap='viridis', interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    else:
        ax = plt.Subplot(fig, inner[0])
        im = ax.imshow(data, cmap='viridis', interpolation='bilinear', vmax=1)
        fig.colorbar(im, ax=ax)
        fig.add_subplot(ax)
plt.savefig("pred_analysis_{}.jpeg".format(norm_type))

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# category analysis
behaviour_cat = np.zeros((5, 5, 4))

for i in range(5):
    for j in range(5):
        a = np.argmax(behav_data[i, j])
        behaviour_cat[i, j, a] = 1

cat_diff = behaviour_cat - cat_data
cat_diff[cat_diff < 0] = 0  # remove double counting
n_cat_diff = np.sum(cat_diff).astype(int)
print(f"n_cat_diff: {n_cat_diff}, accuracy: {(25 - n_cat_diff)/25*100}%, percentage difference: {n_cat_diff/25*100}%")

# create figure
fig = plt.figure(figsize=(6, 3))
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
titles = ["behavioural", "model"]

for i in range(2):
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    ax = plt.Subplot(fig, outer[i])
    ax.set_title(titles[i])
    ax.axis('off')
    fig.add_subplot(ax)

    if i == 0:
        data = behaviour_cat
    elif i == 1:
        data = cat_data

    for j in range(4):
        ax = plt.Subplot(fig, inner[j])
        ax.imshow(data[..., j], cmap='viridis', interpolation='nearest', vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

plt.savefig(f"NRE_{norm_type}_{modality}_category_analysis.jpeg")

# save kl div
np.save((os.path.join(save_path, f"NRE_{norm_type}_{modality}_{condition}_KL_div")), div)
np.save((os.path.join(save_path, f"NRE_{norm_type}_{modality}_{condition}_morph_acc")), (25 - n_cat_diff)/25)

# show plots
if show_plots:
    plt.show()

