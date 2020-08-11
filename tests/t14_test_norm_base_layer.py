import numpy as np
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
n_frames = 3
n_training_data = 2
var_thresh = 0.012

# create random test case
np.random.seed(0)
train_data = np.random.rand(n_features, n_frames, n_training_data)
# print("shape train data", np.shape(train_data))
# print(train_data)
# print()

# reshape data over features
train_data_cat = train_data.reshape((n_features, n_frames*n_training_data), order='F')
print("shape train_data_cat", np.shape(train_data_cat))
# print(train_data_cat)
# print(train_data_cat[:, 1])
# print(np.sum(train_data_cat[:, 1]))

# compute standard deviation for each features
std = np.std(train_data_cat, axis=1, ddof=1)  # ddof=1 is to match the standard practice of matlab using N-1 to compute an unbiased estimate of the variance of the infinite population
# print("std")
# # print(std)

# keep only features having enough variance
train_data_cat = train_data_cat[std > var_thresh]
print("train data thresholded", np.shape(train_data_cat))
# print(train_data_cat)

# compute PCA
# martin implemetations
# print()
# print("PCA")
n_components = 5
pca_m = np.expand_dims(train_data_cat.mean(axis=1), axis=1)  # mean over each column
pca_c = train_data_cat - pca_m * np.ones(pca_m.shape)  # center the values
pca_cov = pca_c @ pca_c.T  # compute covariance matrix
u, s, vh = np.linalg.svd(pca_cov)  # u = left singular array, s = singular values, vh = right singular array
pca_r = u[:, :n_components]  # retain only n_components
pca_p = pca_r.T @ pca_c  # project retained values
print("shape pca_p", np.shape(pca_p))
print("retained ", np.shape(pca_r)[1], " components, corresponding to ", np.sum(s[:n_components]/np.sum(s)), " of explained variance")

# from sklearn.decomposition import PCA
# pca = PCA(n_components)
# pca.fit(train_data_cat)
# print("singular values", pca.singular_values_)
# print(pca.components_)
# print("explained variance", pca.explained_variance_)
# print("retained ", np.shape(pca.singular_values_)[0], " components, corresponding to ", np.sum(pca.explained_variance_), " of explained variance")
# need to correlate with svd eigen value decomposition: https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
# to get similar results

# transofrm pca projected data back to initial 3 axis tensor
training = np.reshape(pca_p, (n_features, n_frames, n_training_data), order='F')

# ------------------------------------------------------------------------
# -----------------           Norm Base model         --------------------
# ------------------------------------------------------------------------
n_frames = 2  # number of time steps for reference feature
n_features = n_components  # number of features
n_classes = 2  # todo, is it output size?
input = training[:, :n_frames, :]  # create input array of training
print("shape input", np.shape(input))

# -----------------               Training            --------------------
# compute references pattern
ref_pattern = np.zeros((n_features, n_classes))
for c in range(n_classes):
    # input[:, :, c] is a matrix of (n_features, n_frames)
    ref_pattern[:, c] = np.mean(input[:, :, c], axis=1)
print("shape ref_pattern", np.shape(ref_pattern))
print(ref_pattern)
print()

# compute differences to reference patterns and average direction vectors
input = training # create input array for training  # todo change training to be once neutral pose, then for n_classes
n_frames = 3  # number of time steps for direction vectors
dir_tuning = np.zeros(((n_features, n_classes)))
for c in range(n_classes):
    # print(input[:, :, c])
    # print("shape ref_pattern", np.shape(ref_pattern[:, c]))
    ref_rep = np.repeat(np.expand_dims(ref_pattern[:, c], axis=1), n_frames, axis=1)  # repeat references to match size of input
    diff = input[:, :, c] - ref_rep  # compute difference
    diff_mean = np.mean(diff, axis=1)
    dir_tuning[:, c] = diff_mean/np.linalg.norm(diff_mean)

print("shape dir tuning", np.shape(dir_tuning))
print(dir_tuning)
print()
