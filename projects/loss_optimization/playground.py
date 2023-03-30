import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
run: python -m projects.loss_optimization.01_create_alpha_plot
"""

save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results'
# save_path = 'D:/Dataset/FERG_DB_256/loss_optimization'

with open(os.path.join(save_path, 'problems'), 'rb') as f:
    problems = pickle.load(f)

print(problems.keys())
# problem is in domain 0, feature map 8
# for k in range(10):
#     print(k, np.where(problems['x_shifted'][:, k, 0].numpy() == 0))

print(problems['projections'][:, 8, :])
for i in range(10000):
    print(problems['projections'][i, 8, :])
