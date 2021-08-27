import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

"""
test a face transfer using norm base mechanism

run: python -m projects.facial_shape_expression_recognition_transfer.04_basic_face_shape_transfer

"""

# --------------------------------------------------------------------------------------------------------------------
# declare parameters
# load config
config_name = 'NB_basic_face_space_transfer_Louise2Merry.json'
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
    face_neurons = model.fit(data, fit_semantic=False)

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


# --------------------------------------------------------------------------------------------------------------------
# apply face transfer using the second avatar of the config
# load data
merry_data = load_data(config, train=False)

# fit reference frames and predict model
model.update_RF(config['rbf_template_merry'], config['rbf_sigma_merry'],
                mask=config['rbf_mask_merry'],
                zeros=config['rbf_zeros_merry'])
merry_face_neurons = model.fit(merry_data,
                               fit_dim_red=True,  # need to fit this to learn the patterns with the new receptieve field
                               fit_semantic=False,  # no need to redo this
                               fit_ref=True,  # learn new reference
                               fit_tun=False)  # set to false as we want to transfer this!


# --------------------------------------------------------------------------------------------------------------------
# print decision
model.print_decision(data, face_neurons)

# print decision
model.print_decision(merry_data, merry_face_neurons)