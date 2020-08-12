import numpy as np
import matplotlib.pyplot as plt
from bvs.layers import NormBase

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

np.set_printoptions(precision=4, linewidth=150)
"""
run: python -m tests.t14_test_norm_base_layer
"""

print("Test Norm Base Layer")

n_features = 5
n_samples = 10
n_category = 2

# create random test case
np.random.seed(0)
train_data = np.random.rand(n_samples, n_features)  # mimic one batch input
train_label = np.random.rand(n_samples)
# print("shape train data", np.shape(train_data))
# print(train_data)
# print()

# ------------------------------------------------------------------------
# -----------------           Norm Base model         --------------------
# ------------------------------------------------------------------------
print("shape train_data", np.shape(train_data))
print("shape train_label", np.shape(train_label))

# -----------------               Training            --------------------
"""
make 1 or 2 phase training?
1) update mean and direction tuning at each step according to the label
2) first compute general mean, second compute direction tuning for each category
"""
# compute references pattern m
m = np.mean(train_data, axis=0)  # todo make it to learn on batch and update the ref pattern
# todo use cumulative average, but what if batch is not the same size?
print("shape m (reference pattern)", np.shape(m))
print(m)
print()

# compute references direction tuning n
# todo: aim to changes the per_category to per samples
n = np.zeros((n_features, n_category))
diff = train_data - np.repeat(np.expand_dims(m, axis=1), n_samples, axis=1)  # compute difference u-m
diff_mean = np.mean(diff, axis=1)  # todo: here this mean is over a single cat!
dir_tuning = diff_mean/np.linalg.norm(diff_mean)

# todo: will need to try with multiple epoch to see if ti converges!!!!

print("shape dir tuning", np.shape(dir_tuning))
print(dir_tuning)
print()

# -----------------               Evaluate            --------------------
# compute response of norm-reference
f_itp = np.zeros((n_classes, n_frames, n_classes))
for c in range(n_classes):
    ref_rep = np.repeat(np.expand_dims(ref_pattern[:, c], axis=1), n_frames, axis=1)  # repeat references to match size of input
    diff = input[:, :, c] - ref_rep  # compute difference
    diag = np.diag(diff.T.dot(diff))
    diag_sqrt = np.sqrt(diag)
    resp = diff.dot(np.diag(np.power(diag_sqrt, -1)))
    f = dir_tuning.T.dot(resp)
    f[f < 0] = 0  # keep only positive values
    f = np.power(f, n_classes).dot(np.diag(diag_sqrt))
    f = f / np.max(f)  # normalize
    f_itp[:, :, c] = f

print()
print(f_itp[:, :, 0])
print(f_itp[:, :, 1])

plt.figure()
plt.subplot(211)
plt.plot(f_itp[0, :, 0], label="class1")
plt.plot(f_itp[1, :, 0], label="class2")
plt.title("class 1")
plt.legend()
plt.subplot(212)
plt.plot(f_itp[0, :, 1], label="class1")
plt.plot(f_itp[1, :, 1], label="class2")
plt.title("class 2")
plt.legend()
plt.show()
