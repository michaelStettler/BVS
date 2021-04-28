"""
2020/12/11
The purpose of this script is to try out different tuning functions.
This script calculates the response of the neurons like in the ICANN paper (Fig3 and FIg4)
Use 02_reproduce_ICANN_NormBase_plots.py to plot the saved results.
2021/01/18
only the function "expressivity-direction" is used by now

run: python -m projects.dynamic_facial_expressions_ICANN.01_reproduce_ICANN_NormBase
"""

import numpy as np
import os

from utils.load_data import load_data
from utils.load_config import load_config
from models.NormBase import NormBase

# t0001: 2-norm     t0002: 1-norm   t0003: simplified   t0004: direction-only   t0005: expressitivity-direction
# t0006: 2-norm-monkey_morph

# config = load_config("norm_base_reproduce_ICANN_t0015.json", path="configs/norm_base_config")
config = load_config("NB_reproduce_ICANN_m0002.json", path="configs/norm_base_config")

# --------------------------------------------------------------------------------------------------------------------
# train model
# load training data
data_train = load_data(config)

# fit and save model
norm_base = NormBase(config, input_shape=(224, 224, 3))
face_neurons = norm_base.fit(data_train)
norm_base.save()

# --------------------------------------------------------------------------------------------------------------------
# predict
# load testing data
data_test = load_data(config, train=False)

# predict model
expr_neurons, face_neurons, differentiators = norm_base.predict(data_test,
                                                                get_it_resp=True,  # needs to set get_it_resp to True in order to get the face neurons since the model is dynamic
                                                                get_differentiator=True)
print("shape expr_neurons", np.shape(expr_neurons))
print("shape face_neurons", np.shape(face_neurons))
print("shape differentiators", np.shape(differentiators))

# --------------------------------------------------------------------------------------------------------------------
# plot
norm_base.plot_it_neurons_per_sequence(face_neurons,
                                       title="01_test",
                                       save_folder=os.path.join("models/saved", config['config_name']),
                                       normalize=True)

norm_base.plot_differentiators(differentiators,
                               title="02_test",
                               save_folder=os.path.join("models/saved", config['config_name']),
                               normalize=True)

norm_base.plot_decision_neurons(expr_neurons,
                                title="03_test",
                                save_folder=os.path.join("models/saved", config['config_name']),
                                normalize=True)

# deprecated
# accuracy1, it_resp1, labels1 = norm_base.evaluate(data_train)
# accuracy2, it_resp2, labels2 = norm_base.evaluate(data_test)
# print("accuracy1", accuracy1)
# print("it_resp1.shape", it_resp1.shape)
# print("labels1.shape", labels1.shape)
# print("accuracy2", accuracy2)
# print("it_resp2.shape", it_resp2.shape)
# print("labels2.shape", labels2.shape)
#
# accuracy = (0.25*accuracy1)+(0.75*accuracy2)
# it_resp = np.concatenate([it_resp1, it_resp2], axis=0)
# labels = np.concatenate([labels1,labels2], axis=0)
#
# print("accuracy", accuracy)
# print("it_resp.shape", it_resp.shape)
# print("labels.shape", labels.shape)
#
# # save results to be used by 02_reproduce_ICANN_NormBase_plots.py
# save_folder = os.path.join("models/saved", config['save_name'], save_name)
# np.save(os.path.join(save_folder, "accuracy"), accuracy)
# np.save(os.path.join(save_folder, "it_resp"), it_resp)
# np.save(os.path.join(save_folder, "labels"), labels)