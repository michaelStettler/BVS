import numpy as np
from utils.load_config import load_config
from utils.PatternFeatureReduction import PatternFeatureSelection

np.set_printoptions(precision=3)

"""
small script to test the Pattern feature selection pipeline using RBF_patch_pattern templates

run: python -m tests.RBF_patch_pattern.t02_pattern_ft_selection
"""

# define configuration
config_path = 'RBF_t02_pattern_ft_selection_m0001.json'
# load config
config = load_config(config_path, path='configs/RBF_patch_pattern')

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
# expand data for RBF_patch_pattern
data = np.expand_dims(data, axis=0)
data = np.expand_dims(data, axis=3)
print("shape data", np.shape(data))
print()

# define mask
mask = [[[1, 4], [1, 4]]]

pattern = PatternFeatureSelection(config, mask=mask)
pred = pattern.fit([data])
print("pred", np.shape(pred))
print(pred[0, ..., 0])
