# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# """
# run: python -m projects.loss_optimization.01_create_alpha_plot
# """
#
# save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results'
# # save_path = 'D:/Dataset/FERG_DB_256/loss_optimization'
#
# with open(os.path.join(save_path, 'problems'), 'rb') as f:
#     problems = pickle.load(f)
#
# print(problems.keys())
# print(problems['grad_shifts'])

import numpy as np

X = np.random.randn(50, 2) + np.array([10, 10])
v = np.random.randn(2)
v = v / np.linalg.norm(v)

u, s, vh = np.linalg.svd(X)
v = vh[0, :]

print('Dot:', np.sum(np.abs(X @ v)))

dist = np.linalg.norm(X - np.expand_dims(X @ v, 1) * np.expand_dims(v, 0), axis=1)
print(np.sum(dist))
