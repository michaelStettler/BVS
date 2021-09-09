import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase
from utils.remove_transition_morph_space import remove_transition_frames
from plots_utils.plot_morphing_space import plot_morphing_space
from plots_utils.plot_morphing_space import plot_morphing_space_categorization

"""
test a face transfer using norm base mechanism

run: python -m projects.facial_shape_expression_recognition_transfer.05_dynamic_facial_expression_recognition_transfer
"""

# load config
config_name = 'NB_morph_space_transfer_m0001.json'
config = load_config(config_name, path='configs/norm_base_config')

# # --------------------------------------------------------------------------------------------------------------------
# # train model
#
# # declare model
# model = NormBase(config, input_shape=tuple(config['input_shape']))
#
# # load data
# data = load_data(config)
# data_no_transition = remove_transition_frames(data)
#
# # fit model
# ds_neurons, face_neurons = model.fit(data_no_transition, get_it_resp=True, fit_semantic=False)
#
# # save model
# model.save()
#
# # predict full sequence using the transition frames
# ds_neurons, face_neurons = model.predict(data, get_it_resp=True)
# print("--- Finished predicting Human protypes -----")
# print()
#
# # --------------------------------------------------------------------------------------------------------------------
# # predict entire morphing space
# config['train_expression'] = ['full']
# morph_space_hum_data = load_data(config)
# ds_morphspace_neurons, face_morphspace_neurons = model.predict(morph_space_hum_data, get_it_resp=True)
# print("--- Finished predicting Human morph space -----")
# print()
#
# # --------------------------------------------------------------------------------------------------------------------
# # apply face transfer using the monkey avatar for the 4 prototypes
# # load data
# test_data = load_data(config, train=False)
#
# # update Receptive Field, fit new reference frame and predict expression on transfer tuning
# model.update_RF(config['rbf_test_template'], config['rbf_test_sigma'],
#                 mask=config['rbf_test_mask'],
#                 zeros=config['rbf_test_zeros'])
# test_ds_neurons, test_face_neurons = model.fit(test_data,
#                                               fit_dim_red=True,    # need to fit this to learn the patterns with the new RF
#                                               fit_semantic=False,  # no need to redo this
#                                               fit_ref=True,        # learn new reference
#                                               fit_tun=False,       # set to false as we want to transfer this!
#                                               get_it_resp=True)
# print("--- Finished predicting Monkey protypes -----")
# print()
#
# # --------------------------------------------------------------------------------------------------------------------
# # predict entire morphing space
# config['val_expression'] = ['full']
# morph_space_monk_data = load_data(config, train=False)
# test_ds_morphspace_neurons, test_face_morphspace_neurons = model.predict(morph_space_monk_data, get_it_resp=True)
# print("--- Finished predicting monkey morph space -----")
# print()
#
# # --------------------------------------------------------------------------------------------------------------------
# # print decision
# data[1] = np.reshape(data[1], (-1, config["seq_length"]))  # reshape labels per sequence
# face_neurons_print = np.reshape(face_neurons, (-1, config["seq_length"], np.shape(face_neurons)[-1]))
# model.print_decision(data, face_neurons_print)
#
# # print test transfer decision
# test_data[1] = np.reshape(test_data[1], (-1, config["seq_length"]))  # reshape labels per sequence
# test_face_neurons_print = np.reshape(face_neurons, (-1, config["seq_length"], np.shape(test_face_neurons)[-1]))
# model.print_decision(test_data, test_face_neurons_print)
#
# # plot face neurons
# model.plot_it_neurons_per_sequence(face_neurons,
#                                    title="01_it_train",
#                                    save_folder=os.path.join("models/saved", config["config_name"]))
# model.plot_it_neurons_per_sequence(test_face_neurons,
#                                    title="01_it_test",
#                                    save_folder=os.path.join("models/saved", config["config_name"]))
#
# # plot decision neurons
# model.plot_decision_neurons(ds_neurons,
#                             title="02_it_train",
#                             save_folder=os.path.join("models/saved", config["config_name"]))
# model.plot_decision_neurons(test_ds_neurons,
#                             title="02_it_test",
#                             save_folder=os.path.join("models/saved", config["config_name"]))

# np.save(os.path.join("models/saved", config["config_name"], "face_morphspace_neurons"), face_morphspace_neurons)
# np.save(os.path.join("models/saved", config["config_name"], "ds_morphspace_neurons"), ds_morphspace_neurons)

# np.save(os.path.join("models/saved", config["config_name"], "test_face_morphspace_neurons"), test_face_morphspace_neurons)
# np.save(os.path.join("models/saved", config["config_name"], "test_ds_morphspace_neurons"), test_ds_morphspace_neurons)


# plot morphing space
# plot face neurons
face_morphspace_neurons = np.load(os.path.join("models/saved", config["config_name"], "face_morphspace_neurons.npy"))
print("shape face_morphspace_neurons", np.shape(face_morphspace_neurons))
face_morphspace_neurons = np.reshape(face_morphspace_neurons, (25, -1, np.shape(face_morphspace_neurons)[-1]))
print("shape face_morphspace_neurons", np.shape(face_morphspace_neurons))
plot_morphing_space(face_morphspace_neurons,
                    title="03_human",
                    save_folder=os.path.join("models/saved", config["config_name"]))
plot_morphing_space_categorization(face_morphspace_neurons,
                                   title="04_human",
                                   save_folder=os.path.join("models/saved", config["config_name"]))

test_face_morphspace_neurons = np.load(os.path.join("models/saved", config["config_name"], "test_face_morphspace_neurons.npy"))
test_face_morphspace_neurons = np.reshape(test_face_morphspace_neurons, (25, -1, np.shape(test_face_morphspace_neurons)[-1]))
print("shape test_face_morphspace_neurons", np.shape(test_face_morphspace_neurons))
plot_morphing_space(test_face_morphspace_neurons,
                    title="03_monkey",
                    save_folder=os.path.join("models/saved", config["config_name"]))
plot_morphing_space_categorization(test_face_morphspace_neurons,
                                   title="04_monkey",
                                   save_folder=os.path.join("models/saved", config["config_name"]))

# todo count how many patterns are classified the same after transfer

