import numpy as np
import matplotlib.pyplot as plt
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

"""
script to test the semantic feature selection within the NormBase mechanism

run: python -m tests.NormBase.t10_train_norm_base_wh_semantic
"""

# load config
config_path = 'NB_t10_train_norm_base_wh_semantic_m0001.json'
config = load_config(config_path, path='configs/norm_base_config')

# declare model
model = NormBase(config, input_shape=tuple(config['input_shape']), load_NB_model=True)  # load model set to True so we get the semantic dictionary loaded

# fit model
data = load_data(config, train=True)
model.fit(data,
          fit_dim_red=False,  # if it is the first time to run this script change this to True -> time consuming
          fit_ref=True,
          fit_tun=True)
model.save()

