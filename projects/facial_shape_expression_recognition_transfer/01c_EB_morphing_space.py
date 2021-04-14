import numpy as np
import os

from datasets_utils.tranform_morph_space_list2space import transform_morph_space_list2space
from utils.load_config import load_config
from utils.load_data import load_data
from models.ExampleBase import ExampleBase
from plots_utils.plot_morphing_space import plot_it_resp
from plots_utils.plot_morphing_space import plot_morphing_space

"""
test the naive PCA selection on the original morphing space dataset from the eLife paper

run: python -m projects.facial_shape_expression_recognition_transfer.01c_EB_morphing_space
"""

train_model = True
predict_it_resp = False

# load config
config_name = 'EB_morphing_space_m0001.json'
config = load_config(config_name, path='configs/example_base')

# --------------------------------------------------------------------------------------------------------------------
# declare model
model = ExampleBase(config, input_shape=tuple(config['input_shape']), load_EB_model=True)

# train model
if train_model:
    print("[FIT] Train model]")
    model.fit(load_data(config, train=True),
              fit_normalize=True,
              fit_dim_red=True,
              fit_snapshots=True)
    model.save()

# plot training snapshots
model.plot_snapshots(title="01_train")
