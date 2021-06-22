import os
import numpy as np
import pickle
import tensorflow as tf
from utils.load_config import load_config
from utils.load_data import load_data
from utils.feature_reduction import load_feature_selection
from utils.Semantic.SemanticFeatureSelection import SemanticFeatureSelection
from utils.Semantic.load_coco_semantic_annotations import get_coco_cat_ids
from utils.Semantic.find_semantic_units import get_IoU_per_category
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


config_path = 'CNN_t02a_fit_feature_semantic_m0004.json'
load_semantic_dict = False
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
semantic = SemanticFeatureSelection(config, model)
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
    preds = semantic.fit(preds, activation='max')
print("[PREDS] shape preds", np.shape(preds))

if save:
    # save only the semantic dictionary
    pickle.dump(semantic.sem_idx_list,
                open(os.path.join(folder_path, "semantic_dictionary.pkl"), 'wb'))
# --------------------------------------------------------------------------------------------------------------------
# output layer idx
# translate IoU to CNN indexes
sem_idx_list = semantic.sem_idx_list
cat_ids = get_coco_cat_ids(config, config['semantic_units'], to_numpy=True)
cat_feature_map_indexes = get_IoU_per_category(sem_idx_list, cat_ids)
print("[Index] Computed category index for {}".format(cat_ids))
print()

layer_of_interest = config["v4_layer"]
for i, cat_id in enumerate(cat_ids):
    print("idx for:", config['semantic_units'][i])
    # get layer idx
    try:
        # get indexes from dictionary
        feature_map = cat_feature_map_indexes["category_{}".format(cat_id)][layer_of_interest]["indexes"]
        print("[Index] feature map indexes for category {} at layer {}: {}".format(cat_id, layer_of_interest, feature_map))
        print("[Index] num of selected feature map: {}".format(len(feature_map)))

    except:
        print("no index exists for this concept on this layer!")
print()
# --------------------------------------------------------------------------------------------------------------------
# plot

# plot selection of feature maps
plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
                config['v4_layer'] + "_selection.gif",
                video=True,
                verbose=True)
print("[TEST] Finished plotting ft maps")
print()