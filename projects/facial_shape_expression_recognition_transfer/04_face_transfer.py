import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

"""
test a face transfer using norm base mechanism

run: python -m projects.facial_shape_expression_recognition_transfer.04_face_transfer
"""

# load config
config_name = 'NB_morph_space_transfer_m0001.json'
config = load_config(config_name, path='configs/norm_base_config')

full_train = False

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
else:
    model = NormBase(config, input_shape=tuple(config['input_shape']), load_NB_model=True)

    # load data
    data = load_data(config)

    # fit model
    face_neurons = model.fit(data, fit_dim_red=False)

    # save model
    model.save()

# plot training
model.plot_it_neurons(face_neurons,
              title="train_full",
              save_folder=os.path.join("models/saved", config['config_name']))

model.plot_it_neurons_per_sequence(face_neurons,
                                   title="train",
                                   save_folder=os.path.join("models/saved", config['config_name']))

# --------------------------------------------------------------------------------------------------------------------
# predict model
# load data
data = load_data(config, train=False)

# predict model
face_neurons = model.predict(data)

model.plot_it_neurons_per_sequence(face_neurons,
                                   title="test",
                                   save_folder=os.path.join("models/saved", config['config_name']))
