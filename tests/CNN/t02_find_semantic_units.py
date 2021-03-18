import os
import numpy as np
from utils.load_config import load_config
from utils.load_data import load_data
from utils.load_extraction_model import load_extraction_model
from utils.find_semantic_units import find_semantic_units

"""
test to try the find_semantic_function that implement the paper: 
todo put paper name

run: python -m tests.CNN.t02_find_semantic_units
"""
np.random.seed(0)
config_path = 'CNN_t02_find_semantic_units_m0001.json'
save = True

np.set_printoptions(precision=3, suppress=True, linewidth=150)

config = load_config(config_path, path='configs/CNN')

# load model
model = load_extraction_model(config)
# print(model.summary())

# load data
data = load_data(config)
print("[Loading] shape x", np.shape(data[0]))
print("[Loading] shape label", np.shape(data[1]))
print("[loading] finish loading data")
print()

# compute face units
find_semantic_units(model, data[0], data[1])