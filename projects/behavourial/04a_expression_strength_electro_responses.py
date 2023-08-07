import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from plots_utils.plot_BVS import display_images
from plots_utils.plot_sequence import plot_sequence
from plots_utils.plot_signature_analysis import plot_signature_proj_analysis

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

"""
run: python -m projects.behavourial.04a_expression_strength_electro_responses
"""

#%% declare hyper parameters
save_plot = True
ratio = 120.1

#%% import data
data = [[23.17, 43.51, 49.73, 67.64],  # fear
        [-5.91, 4.05, 20.56, 44.06],  # lip smack
        [18.27, 28.98, 40.16, 52.06]]  # angry

# normalize the values with ramona's plot ratio
# -> I measured the value on her poster image
data = np.array(data) / ratio
print("-- Data created --")
print("shape data", np.shape(data))
print()


#%% bar plot
fig, axs = plt.subplots(1, 3)
colors = np.array([(0, 0, 255), (0, 191, 0), (237, 0, 0)]) / 255
titles = ["Fear", "Lipsmack", "threat"]
# fig.suptitle('NRE Predictions for expression strength level')
for i in range(3):
    x = [0, 1, 2, 3]
    max_y = np.amax(data[i])
    y = data[i] / max_y
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    axs[i].bar(x, y, color=colors[i])
    axs[i].set_xticks(np.arange(4), ['25', '50', '75', '100'])
    axs[i].set_title(titles[i], color=colors[i])
    axs[i].plot(x, p(x), color='black', linewidth=2)

    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    if i > 0:
        axs[i].spines['left'].set_visible(False)
        axs[i].get_yaxis().set_ticks([])
    # axs[i].spines['bottom'].set_visible(False)

    axs[i].set_ylim([-0.3, 1])
plt.savefig(f"NRE_electro_bar_expr_level.svg", format='svg')
plt.show()



