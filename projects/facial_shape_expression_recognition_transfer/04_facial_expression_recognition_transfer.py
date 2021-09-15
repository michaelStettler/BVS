import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

"""
Reproduce result of Figure XX from the paper: 

The model learns to recognize the 6 basic expressions from the Neutral expression as reference. 
Then the model update the reference frame and the tuning direction are transferred on a new Basic Face Shape.

The model uses the semantic-pattern pipeline for feature reduction 

run: python -m projects.facial_shape_expression_recognition_transfer.04_facial_expression_recognition_transfer

"""

# --------------------------------------------------------------------------------------------------------------------
# declare parameters
# load config
config_name = 'NB_basic_face_shape_transfer_Louise2Merry.json'
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

# update Receptive Field, fit new reference frame and predict expression on transfer tuning
model.update_RF(config['rbf_template_merry'], config['rbf_sigma_merry'],
                mask=config['rbf_mask_merry'],
                zeros=config['rbf_zeros_merry'])
merry_face_neurons = model.fit(merry_data,
                               fit_dim_red=True,    # need to fit this to learn the patterns with the new RF
                               fit_semantic=False,  # no need to redo this
                               fit_ref=True,        # learn new reference
                               fit_tun=False)       # set to false as we want to transfer this!

# apply transfer to different "identity" of Merry
# load data
# config['val_avatar'] = 'Merry_all_identities'
# merry_all_ids_data = load_data(config, train=False)
# print("Shape merry_all_ids_data[0]", np.shape(merry_all_ids_data[0]))

# --------------------------------------------------------------------------------------------------------------------
# print decision
model.print_decision(data, face_neurons)

# print decision
model.print_decision(merry_data, merry_face_neurons)
