import os
import numpy as np

"""
run: python -m projects.loss_optimization.01_create_alpha_plot
"""

save_path = 'D:/Dataset/FERG_DB_256/loss_optimization'

train_accuracy = np.load(os.path.join(save_path, 'train_alpha_accuracy.npy'))
test_accuracy = np.load(os.path.join(save_path, 'test_alpha_accuracy.npy'))

print("shape train_accuracy", train_accuracy.shape)
print("shape test_accuracy", test_accuracy.shape)


