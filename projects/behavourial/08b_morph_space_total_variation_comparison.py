import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from projects.behavourial.project_utils import *


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.08_morph_space_total_variation_comparison
"""

#%% define computer path
# computer = 'windows'
computer = 'alex'
computer_path, computer_letter = get_computer_path(computer)


#%% declare script parameters
show_plots = True
model_names = ["NRE-indi-S", "NRE-indi-D", "NRE-cat-S", "NRE-cat-D",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet",
               "CORNet_imagenet"]
load_path = os.path.join(computer_path, 'model_behav_preds')
conditions = ["human_orig", "monkey_orig"]


#%%
def get_path(model_name, condition):
    if 'NRE' in model_name:
        # retrieve norm_type
        norm_type = 'individual'
        if 'cat' in model_name:
            norm_type = 'frobenius'

        # retrieve modality
        modality = 'static'
        if 'D' in model_name:
            modality = 'dynamic'

        # acc_path = os.path.join(load_path, f"NRE_{norm_type}_{modality}_{condition}_morph_acc.npy")
        total_variation_path = os.path.join(load_path, f"NRE_{norm_type}_{modality}_{condition}_total_variation.npy")
    else:
        # acc_path = os.path.join(load_path, f"{model_name}_{condition}_morph_acc.npy")
        total_variation_path = os.path.join(load_path, f"{model_name}_{condition}_total_variation.npy")

    return total_variation_path


#%% load data
accuracies = []
distances = []
for model_name in model_names:
    # accuracy = []
    distance = []
    for condition in conditions:
        # create path
        total_variation_path = get_path(model_name, condition)

        # # load data
        # if acc_path is not None:
        #     acc = np.load(acc_path)
        # else:
        #     acc = 0
        if total_variation_path is not None:
            dist = np.load(total_variation_path)
        else:
            dist = np.zeros((5, 5))

        # append to condition
        # accuracy.append(acc)
        distance.append(dist)

    # append to models
    # accuracies.append(accuracy)
    distances.append(distance)

# accuracies = np.array(accuracies)
distances = np.array(distances)
# print("shape accuracies", np.shape(accuracies))
print("shape distances", np.shape(distances))


#%% plot kl divergence
print("shape distances", np.shape(distances))
sum_dist = np.sum(distances, axis=(2, 3))
print("shape sum_dist", np.shape(sum_dist))

fig, ax = plt.subplots()
x = np.arange(len(distances))
width = 0.25
# plot each condition
for c, condition in enumerate(conditions):
    offset = width * c
    rects = plt.bar(x + offset, sum_dist[:, c], width, label=condition)
    # ax.bar_label(rects, padding=3)  # add value to bar
# ax.set_xticks(x + width, model_names)
ax.set_xticks(x, model_names)
plt.xticks(rotation=90)
plt.tight_layout()
ax.legend()

plt.savefig(f"bar_plot_total_variation_analysis.svg", format='svg')

for m, model in enumerate(model_names):
    for c, condition in enumerate(conditions):
        print(f"model: {model}-{condition}, KL-div: {sum_dist[m, c]}")
    print()

if show_plots:
    plt.show()