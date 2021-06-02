import numpy as np
import os

from plots_utils.plot_sequence import plot_sequence
from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

"""
test a face transfer using norm base mechanism

run: python -m projects.facial_shape_expression_recognition_transfer.03_face_part_semantic_feature_map_selection
"""

# load config
config_name = 'NB_morph_space_semantic_pattern_m0002.json'
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
    # plot_sequence(np.array(data[0]).astype(np.uint8), video_name='01_train_sequence.mp4',
    #               path=os.path.join("models/saved", config['config_name']))

    # fit model
    face_neurons = model.fit(data, fit_dim_red=False)

    # save model
    model.save()

# plot training
model.plot_it_neurons_per_sequence(face_neurons,
                                   title="01_train",
                                   save_folder=os.path.join("models/saved", config['config_name']))

# --------------------------------------------------------------------------------------------------------------------
# apply face transfer using the monkey avatar
# load data
data = load_data(config, train=False)
# plot_sequence(np.array(data[0]).astype(np.uint8), video_name='02_test_sequence.mp4',
#               path=os.path.join("models/saved", config['config_name']))

# --------------------------------------------------------------------------------------------------------------------
# predict model
face_neurons = model.predict(data)

# model.plot_it_neurons_per_sequence(face_neurons,
#                                    title="02_test",
#                                    save_folder=os.path.join("models/saved", config['config_name']))
model.plot_it_neurons(face_neurons,
                                   title="02a_test",
                                   save_folder=os.path.join("models/saved", config['config_name']))

# --------------------------------------------------------------------------------------------------------------------
# fit reference frames and predict model
face_neurons = model.fit(data, fit_dim_red=False,
          fit_ref=True,
          fit_tun=False)

model.plot_it_neurons_per_sequence(face_neurons,
                                   title="03_test_wh_ref",
                                   save_folder=os.path.join("models/saved", config['config_name']))
