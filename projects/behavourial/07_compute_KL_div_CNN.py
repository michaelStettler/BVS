import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.07_compute_KL_div_CNN
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
model_names = ["VGG19_imagenet", "VGG19_imagenet_conv33", "Resnet50v2_imagenet",
               "VGG19_affectnet", "ResNet50v2_affectnet", "CORNet_affectnet"]
load_path = os.path.join(computer_path, 'model_behav_preds')
conditions = ["human_orig", "monkey_orig"]
cond = 0


#%%
def get_pred(model_name, condition):
    path = os.path.join(load_path, f"{model_name}_{condition}_prob_grid.npy")
    preds = np.load(path)

    return preds


#%% load data

# load behavioural data
behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
behav_data = np.moveaxis(behav_data, 0, -1)
print("shape behav_data", np.shape(behav_data))

# load model preds
predictions = []
for model_name in model_names:
    preds = get_pred(model_name, conditions[cond])
    predictions.append(preds)
predictions = np.array(predictions)
print(f"finished loading predictions (shape: {np.shape(predictions)})")


# compute KL divergence
def KL_divergence(p, q):
    return np.sum(p * np.log(p / q))


def compute_morph_space_KL_div(p, q):
    dim_x = np.shape(p)[0]
    dim_y = np.shape(p)[1]

    divergences = np.zeros((dim_x, dim_y))
    for x in range(dim_x):
        for y in range(dim_y):
            div = KL_divergence(p[x, y], q[x, y])
            divergences[x, y] = div

    return divergences

# store computed values
kl_divergences = []

for pred in predictions:
    kl_div = compute_morph_space_KL_div(behav_data, pred)
    kl_divergences.append(kl_div)
kl_divergences = np.array(kl_divergences)
print(f"finished computing KL div (shape: {np.shape(kl_divergences)})")

# save values
for k, kl_div in enumerate(kl_divergences):
    path = os.path.join(load_path, f"{model_names[k]}_{conditions[cond]}_KL_div")
    np.save(path, kl_div)
