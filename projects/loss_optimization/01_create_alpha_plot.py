import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
run: python -m projects.loss_optimization.01_create_alpha_plot
"""

save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results'

with open(os.path.join(save_path, 'train_alpha_accuracy'), 'rb') as f:
    train_accuracy = pickle.load(f)

with open(os.path.join(save_path, 'test_alpha_accuracy'), 'rb') as f:
    test_accuracy = pickle.load(f)

plt.plot((test_accuracy[0.06]))
plt.show()


