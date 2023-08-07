import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
run: python -m projects.loss_optimization.01_create_alpha_plot
"""

load_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results\subset'
load_path2 = r'C:\Users\Alex\Documents\Uni\NRE\icann_results\subset_radius'
save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results\plots'
# load_path = 'D:/Dataset/FERG_DB_256/loss_optimization'

x_axis = []
only_radius, all_optimized = [], []

for i, file in enumerate(os.listdir(load_path)):
    print(file)
    with open(os.path.join(load_path, file), 'rb') as f:
        X = pickle.load(f)
    # Get last occurence of best training accuracy (flip array and use argmax)
    best_id = len(X['train_accuracies']) - 1 - np.argmax(np.flip(X['train_accuracies']))
    all_optimized.append(X['test_accuracies'][best_id])
    x_axis.append(i)

for i, file in enumerate(os.listdir(load_path2)):
    print(file)
    with open(os.path.join(load_path2, file), 'rb') as f:
        X = pickle.load(f)
    # Get last occurence of best training accuracy (flip array and use argmax)
    best_id = len(X['train_accuracies']) - 1 - np.argmax(np.flip(X['train_accuracies']))
    only_radius.append(X['test_accuracies'][best_id])

plt.rcParams.update({'font.size': 14})

tick_labels = [str(12 * 2 ** i) for i in range(len(os.listdir(load_path)))]

plt.plot(x_axis, all_optimized, label="Optimized")
plt.plot(x_axis, only_radius, label='Non-optimized')
plt.xticks(x_axis, tick_labels, rotation=40)
plt.title('Performance for Training on Subsets')
plt.xlabel('Number of training images (log-scale)')
plt.ylabel('Test set accuracy')
plt.legend()
plt.savefig(os.path.join(save_path, 'subsets.eps'),
            dpi=1200, bbox_inches = "tight")
plt.show()


