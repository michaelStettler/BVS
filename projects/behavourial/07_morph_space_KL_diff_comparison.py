import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.07_morph_space_KL_diff_comparison
"""

#%% define computer path
# computer = 'windows'
computer = 'mac'
if 'windows' in computer:
    computer_path = 'D:/Dataset/MorphingSpace'
    computer_letter = 'w'
elif 'mac' in computer:
    computer_path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace'
    computer_letter = 'm'

#%% declare script parameters
show_plots = True
model_names = ["NRE-indi-S", "NRE-indi-D", "NRE-cat-S", "NRE-cat-D"]
load_path = os.path.join(computer_path, 'model_preds')
conditions = ["human_orig", "monkey_orig", "human_equi", "monkey_equi"]
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

        acc_path = os.path.join(load_path, f"NRE_{norm_type}_{modality}_{condition}_morph_acc.npy")
        kl_div_path = os.path.join(load_path, f"NRE_{norm_type}_{modality}_{condition}_KL_div.npy")
    else:
        print("TODO CNN models")

    return acc_path, kl_div_path


#%% load data
accuracies = []
kl_divergences = []
for model_name in model_names:
    accuracy = []
    kl_divergence = []
    for condition in conditions:
        # create path
        acc_path, kl_div_path = get_path(model_name, condition)

        # load data
        acc = np.load(acc_path)
        kl_div = np.load(kl_div_path)

        # append to condition
        accuracy.append(acc)
        kl_divergence.append(kl_div)

    # append to models
    accuracies.append(accuracy)
    kl_divergences.append(kl_divergence)

accuracies = np.array(accuracies)
kl_divergences = np.array(kl_divergences)
print("shape accuracies", np.shape(accuracies))
print("shape kl_divergences", np.shape(kl_divergences))


#%% plot kl divergence
sum_kl_div = np.sum(kl_divergences, axis=(1, 2))
print("shape sum_kl_div", np.shape(sum_kl_div))

fig, ax = plt.subplots()
x = np.arange(len(kl_divergences))
width = 0.25
# plot each condition
for c, condition in enumerate(conditions):
    offset = width * c
    rects = plt.bar(x + offset, sum_kl_div[:, c], width, label=condition)
    # ax.bar_label(rects, padding=3)  # add value to bar
ax.set_xticks(x + width, model_names)
ax.legend()
if show_plots:
    plt.show()
