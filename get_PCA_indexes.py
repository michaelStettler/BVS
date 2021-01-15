import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=2, linewidth=200, suppress=True)
# set a random seed
np.random.seed(5)

# built a fake dataset
data = np.random.rand(4, 5)
data = StandardScaler().fit_transform(data)
print("data", np.shape(data))
print(data)

final_dim = 2

# ======================================================================================================================
# apply normal PCA from scipy
pca = PCA(n_components=final_dim)
x_new = pca.fit_transform(data)
print("x_new", np.shape(x_new))
print(x_new)

# ======================================================================================================================
# apply personal PCA
# compute cov matrix on column
data_cov = np.cov(data, rowvar=False)

# compute the eigenvalues and eigenvectors
eig_values, eig_vectors = np.linalg.eig(data_cov)

# pick eigenvectors whose eigenvalues are highest
max_index = np.flip(np.argsort(eig_values))
feature_vector = eig_vectors[:,max_index[:final_dim]]

# transform feature
new_data = np.transpose(np.matmul(np.transpose(feature_vector), np.transpose(data)))
print("shape new data", new_data.shape)
print(new_data)

# plt.figure()
# plt.plot(new_data)
# plt.title("PCA")
# plt.savefig("PCA")

# ======================================================================================================================
# apply PCA using SVD
u, s, v = np.linalg.svd(data, full_matrices=False)  # It's not necessary to compute the full matrix of U or V

# transform feature
new_data = np.dot(u[:, :final_dim], np.diag(s[:final_dim]))
print("shape new data", new_data.shape)
print(new_data)

# plt.figure()
# plt.plot(new_data)
# plt.title("SVD")
# plt.savefig("SVD")
