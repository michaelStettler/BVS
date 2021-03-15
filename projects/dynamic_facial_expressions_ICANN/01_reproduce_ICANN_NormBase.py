"""
2020/12/11
The purpose of this script is to try out different tuning functions.
This script calculates the response of the neurons like in the ICANN paper (Fig3 and FIg4)
Use 02_reproduce_ICANN_NormBase_plots.py to plot the saved results.
2021/01/18
only the function "expressivity-direction" is used by now
"""

import numpy as np
import os

from utils.load_data import load_data
from utils.load_config import load_config
from models.NormBase import NormBase

# t0001: 2-norm     t0002: 1-norm   t0003: simplified   t0004: direction-only   t0005: expressitivity-direction
# t0006: 2-norm-monkey_morph

config = load_config("norm_base_reproduce_ICANN_t0015.json", path="configs/norm_base_config")
save_name = config["sub_folder"]

data_train = load_data(config)

# fit and save model
norm_base = NormBase(config, input_shape=(224,224,3))
norm_base.fit(data_train)
norm_base.save_model(config, save_name)

#load model
norm_base = NormBase(config, input_shape=(224,224,3), save_name=save_name)
data_test = load_data(config, train=False, sort_by=['image'])

# evaluate model
accuracy1, it_resp1, labels1 = norm_base.evaluate(data_train)
accuracy2, it_resp2, labels2 = norm_base.evaluate(data_test)

print("accuracy1", accuracy1)
print("it_resp1.shape", it_resp1.shape)
print("labels1.shape", labels1.shape)
print("accuracy2", accuracy2)
print("it_resp2.shape", it_resp2.shape)
print("labels2.shape", labels2.shape)

accuracy = (0.25*accuracy1)+(0.75*accuracy2)
it_resp = np.concatenate([it_resp1, it_resp2], axis=0)
labels = np.concatenate([labels1,labels2], axis=0)

print("accuracy", accuracy)
print("it_resp.shape", it_resp.shape)
print("labels.shape", labels.shape)

# save results to be used by 02_reproduce_ICANN_NormBase_plots.py
save_folder = os.path.join("models/saved", config['save_name'], save_name)
np.save(os.path.join(save_folder, "accuracy"), accuracy)
np.save(os.path.join(save_folder, "it_resp"), it_resp)
np.save(os.path.join(save_folder, "labels"), labels)