import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_config import load_config

config = load_config("norm_base_affectNet_sub8_4000_t0006.json")

if isinstance(config['nu'], list):
    nus = config['nu']
else:
    nus = [config['nu']]

accuracies = np.zeros(len(nus))
for i, nu in enumerate(nus):
    config['nu'] = nu

    save_folder = os.path.join("models/saved", config['save_name'], 'nu_%f'%nu)
    accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
    accuracies[i] = accuracy

# create plot
fig = plt.figure(figsize=(15,10))
plt.plot(nus, accuracies)
plt.title("Accuracy over nu")
plt.xlabel("nu")
plt.ylabel("accuracy")

# save plot
plt.savefig(os.path.join("models/saved", config['save_name'], "plot_accuracy_nu.png"))