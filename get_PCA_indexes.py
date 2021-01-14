import numpy as np
from sklearn.decomposition import PCA

# set a random seed
np.random.seed(0)

# built a fake dataset
data = np.random.rand(4, 5)
print("data")
print(data)

# apply normal PCA from scipy
pca = PCA(n_components=2)
x_new = pca.fit_transform(data)
print("")
print(x_new)

# apply personal PCA