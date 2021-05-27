import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

"""
test a face transfer using norm base mechanism

run: python -m projects.facial_shape_expression_recognition_transfer.03_face_part_semantic_feature_map_selection
"""

# load config
config_name = 'NB_morph_space_semantic_pattern_m0001.json'
config = load_config(config_name, path='configs/norm_base_config')

full_train = True

# --------------------------------------------------------------------------------------------------------------------
# train model
if full_train:
    # declare model
    model = NormBase(config, input_shape=tuple(config['input_shape']))

    # load data
    data = load_data(config)

    # fit model
    face_neurons = model.fit(data)

    # save model
    model.save()