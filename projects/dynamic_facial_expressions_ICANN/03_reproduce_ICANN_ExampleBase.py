import os
import numpy as np

from utils.load_config import load_config
from utils.load_data import load_data
from models.ExampleBase import ExampleBase

"""
Reproduce the results from the ICANN paper but with the updated VGG pipeline

run: python -m projects.dynamic_facial_expressions_ICANN.03_reproduce_ICANN_ExampleBase
"""

# load config
config = load_config("EB_reproduce_ICANN_cat.json", path="configs/example_base")

# load model
model = ExampleBase(config, input_shape=tuple(config['input_shape']))

# load data
train_data = load_data(config)

# fit model
model.fit(train_data)
model.save(config)