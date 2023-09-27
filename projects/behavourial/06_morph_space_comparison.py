import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from G_compute_KL_div_CNN import *

np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.06_morph_space_comparison
"""

#%% define computer path
# computer = 'windows'
computer = 'alex'
plot_format = 'svg'
computer_path, computer_letter = get_computer_path(computer)

#%% declare script parameters
condition = "human_orig"
norm_type = "categorical"
CNN_model = 'VGG19_imagenet'

#%% construct path
behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
NRE_path = os.path.join(computer_path, 'saved_lmks_pos', condition)
CNN_path = os.path.join(computer_path, 'model_preds')

#%% load data
behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
behav_data = np.moveaxis(behav_data, 0, -1)
NRE_pred = np.load(os.path.join(NRE_path, f"prob_grid_{norm_type}.npy"))
NRE_pred = NRE_pred[..., 1:]
CNN_pred = np.load(os.path.join(CNN_path, f"{CNN_model}_{condition}_prob_grid.npy"))
# CNN_pred = CNN_pred[..., 1:]

# ### Use synthetic data untul real data is available
# NRE_pred = np.abs(np.random.randn(*behav_data.shape))
# NRE_pred /= np.sum(NRE_pred, axis=-1, keepdims=True)
# CNN_pred = np.abs(np.random.randn(*behav_data.shape))
# CNN_pred /= np.sum(CNN_pred, axis=-1, keepdims=True)

print("shape behav_data", np.shape(behav_data))
print("shape NRE_pred", np.shape(NRE_pred))
print("shape CNN_pred", np.shape(CNN_pred))
data = np.array([behav_data, NRE_pred, CNN_pred])

#%% compute category
categories = []
for i in range(len(data)):
    argmax = np.argmax(data[i], axis=2)
    cat = np.zeros(np.shape(data[i]))
    for m in range(5):
        for n in range(5):
            cat[m, n, argmax[m, n]] = 1
    categories.append(cat)
categories = np.array(categories)

NRE_cat_count = np.array(categories[0] - categories[1])
NRE_cat_count[NRE_cat_count < 0] = 0
NRE_cat_count = np.sum(NRE_cat_count)
print("[NRE] ratio of categorization difference", NRE_cat_count/25 * 100)

CNN_cat_count = np.array(categories[0] - categories[2])
CNN_cat_count[CNN_cat_count < 0] = 0
CNN_cat_count = np.sum(CNN_cat_count)
print("[CNN] ratio of categorization difference", CNN_cat_count/25 * 100)
print()



NRE_div = compute_morph_space_KL_div(behav_data, NRE_pred)
CNN_div = compute_morph_space_KL_div(behav_data, CNN_pred)
print('NREdiv', NRE_div)
print("[NRE] sum div", np.sum(NRE_div))
print(f"[NRE] mean: {np.mean(NRE_div)}, variance: {np.std(NRE_div)}")
print()
print("[CNN] sum div", np.sum(CNN_div))
print(f"[CNN] mean: {np.mean(CNN_div)}, variance: {np.std(CNN_div)}")
divergences = np.array([NRE_div, CNN_div])

# normalize the divergences s.t. the highest overall point is 1
max = - np.inf
for div in divergences:
    iteration_max = np.max(div)
    if iteration_max > max:
        max = iteration_max
    print(max, iteration_max)
for div in divergences:
    div /= max


#%% plot comparison
titles = ["Humans", "NRE", "CNN"]
fig = plt.figure(figsize=(9, 9))
outer = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)

# plot raw outputs
for i in range(3):
    # set outer titles
    ax = plt.Subplot(fig, outer[i])
    ax.set_title(titles[i])
    ax.axis('off')
    fig.add_subplot(ax)

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    for j in range(4):
        ax = plt.Subplot(fig, inner[j])
        normalized = np.array(data[i, :, :, j])
        normalized /= np.amax(normalized)
        ax.imshow(normalized, cmap='viridis', interpolation='bilinear', vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

# # plot categorization
# for i in range(3):
#     # set outer titles
#     ax = plt.Subplot(fig, outer[i])
#     ax.axis('off')
#     fig.add_subplot(ax)
#
#     inner = gridspec.GridSpecFromSubplotSpec(2, 2,
#                                              subplot_spec=outer[i + 3], wspace=0.1, hspace=0.1)
#
#     for j in range(4):
#         ax = plt.Subplot(fig, inner[j])
#         im = ax.imshow(categories[i, ..., j], cmap='viridis', vmax=1)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         fig.add_subplot(ax)
#
#     # if i == 2:
#     #     fig.colorbar(im, ax=ax)

# plot div
for i in range(2):
    ax = plt.Subplot(fig, outer[i + 4])
    ax.axis('off')
    im = ax.imshow(divergences[i], cmap='viridis', interpolation='bilinear',
                   vmin=0, vmax=1)
    print(divergences[i])

    # if i == 1:
    #     fig.colorbar(im, ax=ax)

    fig.add_subplot(ax)


# plot plot bar
ax = plt.Subplot(fig, outer[3])
fig.colorbar(im, ax=ax)

plt.savefig(join('plots', 'behav_results.') + plot_format, format=plot_format)
plt.show()
