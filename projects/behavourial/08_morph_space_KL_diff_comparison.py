import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.08_morph_space_KL_diff_comparison
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
model_names = ["NRE-indi-S", "NRE-indi-D", "NRE-cat-S", "NRE-cat-D",
               "VGG19_imagenet", "VGG19_imagenet_conv33", "Resnet50v2_imagenet",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet"]
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
        kl_div_path = os.path.join(load_path, f"NRE_{norm_type}_{modality}_{condition}_KL_div.npy")
    else:
        # acc_path = os.path.join(load_path, f"{model_name}_{condition}_morph_acc.npy")
        kl_div_path = os.path.join(load_path, f"{model_name}_{condition}_KL_div.npy")

    return kl_div_path


#%% load data
accuracies = []
kl_divergences = []
for model_name in model_names:
    # accuracy = []
    kl_divergence = []
    for condition in conditions:
        # create path
        kl_div_path = get_path(model_name, condition)

        # # load data
        # if acc_path is not None:
        #     acc = np.load(acc_path)
        # else:
        #     acc = 0
        if kl_div_path is not None:
            kl_div = np.load(kl_div_path)
        else:
            kl_div = np.zeros((5, 5))

        # append to condition
        # accuracy.append(acc)
        kl_divergence.append(kl_div)

    # append to models
    # accuracies.append(accuracy)
    kl_divergences.append(kl_divergence)

# accuracies = np.array(accuracies)
kl_divergences = np.array(kl_divergences)
# print("shape accuracies", np.shape(accuracies))
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

plt.savefig(f"bar_plot_kl_div_analysis.svg", format='svg')

if show_plots:
    plt.show()
