import numpy as np
from utils.load_config import load_config
from models.RBF import RBF

"""
small script to test the RBF 2d convolution

run: python -m tests.RBF.t01_2d_rbf
"""

# define configuration
config_path = 'RBF_t01_2d_m0001.json'
# load config
config = load_config(config_path, path='configs/RBF')

data = np.zeros((7, 7))
data[1, 1] = .2
data[1, 2] = .5
data[1, 3] = .2
data[2, 1] = .5
data[2, 2] = 1
data[2, 3] = .5
data[3, 1] = .2
data[3, 2] = .5
data[3, 3] = .2
print("data")
print(data)
# expand data for RBF
data = np.expand_dims(data, axis=0)
data = np.expand_dims(data, axis=3)
print("shape data", np.shape(data))
print()

template = data[:, 1:4, 1:4, :]
print("template")
print(template[0, ..., 0])
print()

rbf = RBF(config)
pred = rbf.fit2d(template)
print("pred", np.shape(pred))
print(pred[0, ..., 0])

test = rbf.predict2d(data)
print("test", np.shape(data))
print(test[:, ..., 0])
