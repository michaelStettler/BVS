import os
import numpy as np
import pickle
import tensorflow as tf
from utils.load_config import load_config
from utils.load_data import load_data
from utils.feature_reduction import load_feature_selection
from utils.Semantic.SemanticFeatureSelection import SemanticFeatureSelection
from utils.extraction_model import load_extraction_model
from plots_utils.plot_cnn_output import plot_cnn_output
"""
test script to run the "SemanticFeatureSelection" function which implement the paper:

Network Dissection: Quantifying Interpretability of Deep Visual Representations

from David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba

the test compare the selectivity between eyebrow and lips units

run: python -m tests.CNN.t02a_fit_feature_semantic
"""
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)


config_path = 'CNN_t02a_fit_feature_semantic_m0003.json'
load_semantic_dict = True
save = True

# load config
config = load_config(config_path, path='configs/CNN')
folder_path = os.path.join("models/saved", config['config_name'])

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# load model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input,
                          outputs=model.get_layer(config['v4_layer']).output)
print("[INIT] extraction model loaded")

# load semantic feature selection
semantic = SemanticFeatureSelection(config)
if load_semantic_dict:
    sem_idx_list = pickle.load(open(os.path.join(folder_path, "semantic_dictionary.pkl"), 'rb'))
    semantic.sem_idx_list = sem_idx_list
print("[INIT] Semantic Feature Selection loaded")

# load data
data = load_data(config)
print("[INIT] shape data", np.shape(data[0]))

# --------------------------------------------------------------------------------------------------------------------
# predict data
preds = model.predict(data[0], verbose=True)
if not load_semantic_dict:
    semantic.fit(model)
preds = semantic.transform(preds)
print("[PREDS] shape preds", np.shape(preds))

if save:
    # save only the semantic dictionary
    pickle.dump(semantic.sem_idx_list,
                open(os.path.join(folder_path, "semantic_dictionary.pkl"), 'wb'))

# --------------------------------------------------------------------------------------------------------------------
# plot

# plot selection of feature maps
plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
                config['v4_layer'] + "_selection.gif",
                video=True,
                verbose=True)
print("[TEST] Finished plotting ft maps")
print()