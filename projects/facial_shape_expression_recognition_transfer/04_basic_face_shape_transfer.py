import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

"""
test a face transfer using norm base mechanism

run: python -m projects.facial_shape_expression_recognition_transfer.04_basic_face_shape_transfer

"""

# todo take care of the zeros mask

# load config
config_name = 'NB_basic_face_space_transfer_Louise2Mery.json'
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

# print true labels versus the predicted label
for i, label in enumerate(data[1]):
    t_label = int(label)
    p_label = np.argmax(face_neurons[i])

    if t_label == p_label:
        print(i, "true label:", t_label, "vs. pred:", p_label, " - OK")
    else:
        print(i, "true label:", t_label, "vs. pred:", p_label, " - wrong!")
print()


# --------------------------------------------------------------------------------------------------------------------
# apply face transfer using the monkey avatar
# load data
data = load_data(config, train=False)

# --------------------------------------------------------------------------------------------------------------------
# predict model
face_neurons = model.predict(data)

# --------------------------------------------------------------------------------------------------------------------
# fit reference frames and predict model
model.update_RF(config['rbf_template_merry'], config['rbf_sigma_merry'],
                mask=config['rbf_mask_merry'],
                zeros=config['rbf_zeros_merry'])
face_neurons = model.fit(data,
                         fit_dim_red=True,  # learn the patterns with new receptieve field
                         fit_semantic=False,  # no need to redo this
                         fit_ref=True,
                         fit_tun=False)

# print true labels versus the predicted label
for i, label in enumerate(data[1]):
    t_label = int(label)
    p_label = np.argmax(face_neurons[i])

    if t_label == p_label:
        print(i, "true label:", t_label, "vs. pred:", p_label, " - OK")
    else:
        print(i, "true label:", t_label, "vs. pred:", p_label, " - wrong!")
print()