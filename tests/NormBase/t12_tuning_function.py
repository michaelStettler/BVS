import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.ref_feature_map_neurons import ref_feature_map_neuron
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output
from plots_utils.plot_ft_map_pos import plot_ft_map_pos
from plots_utils.plot_ft_map_pos import plot_ft_pos_on_sequence
from models.NormBase import NormBase

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to control and try the computation of multiple tuning direction within feature maps

run: python -m tests.NormBase.t12_tuning_function
"""
# define configuration
config_path = 'NB_t12_tuning_function_m0001.json'

# load config
config = load_config(config_path, path='configs/norm_base_config')
config['tun_func'] = 'ft_2norm'

# create directory if non existant
save_path = os.path.join("models/saved", config["config_name"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

# load and define model
v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(v4_model.output)[1:3])
print("[LOAD] size_ft", size_ft)
print("[LOAD] Model loaded")
print()

nb_model = NormBase(config, tuple(config['input_shape']))

# --------------------------------------------------------------------------------------------------------------------
# build test case
labels = np.arange(5)  # build one prediction per category
preds = np.zeros((5, 5, 5, 1))
preds[0, 2, 2, :] = 1
preds[1, 1, 1, 0] = 1
preds[2, 3, 2, 0] = 1
print("[TRAIN] preds", np.shape(preds))
print("preds[0]")
print(preds[0, ..., 0])
print("preds[1]")
print(preds[1, ..., 0])
print("preds[2]")
print(preds[2, ..., 0])

# compute positions
pos = calculate_position(preds, mode="weighted average", return_mode="xy float flat")
print("[TRAIN] shape pos", np.shape(pos))
print(pos)
print()

nb_model.n_features = np.shape(pos)[-1]
# train manually ref vector
nb_model.r = np.zeros(nb_model.n_features)
nb_model._fit_reference([pos, labels], config['batch_size'])
print("model reference")
print(nb_model.r)
print()
# train manually tuning vector
nb_model.t = np.zeros((nb_model.n_category, nb_model.n_features))
nb_model.t_mean = np.zeros((nb_model.n_category, nb_model.n_features))
nb_model._fit_tuning([pos, labels], config['batch_size'])
# get it resp for eyebrows
it_train = nb_model._get_it_resp(pos, weights='ones')
print("[TRAIN] shape it_train", np.shape(it_train))

# --------------------------------------------------------------------------------------------------------------------
# plot it responses for eyebrow model
nb_model.plot_it_neurons(it_train,
                         title="01_it_train",
                         save_folder=os.path.join("models/saved", config["config_name"]))
