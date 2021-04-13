import os
import numpy as np
import pickle
import tensorflow as tf
from utils.load_config import load_config
from utils.load_data import load_data
from utils.Semantic.load_coco_semantic_annotations import load_coco_semantic_annotations
from utils.Semantic.load_coco_semantic_annotations import load_coco_categories
from utils.Semantic.load_coco_semantic_annotations import get_coco_cat_ids
from utils.CNN.extraction_model import load_extraction_model
from utils.Semantic.find_semantic_units import find_semantic_units
from utils.Semantic.find_semantic_units import get_IoU_per_category
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output
"""
test script to try the find_semantic_units function which implement the paper:

Network Dissection: Quantifying Interpretability of Deep Visual Representations

from David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba

run: python -m tests.CNN.t02_find_semantic_units
"""
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)


# config_path = 'CNN_t02_find_semantic_units_m0001.json'
config_path = 'CNN_t02_find_semantic_units_m0002.json'
save = True
load = True
do_plot = False

# load config
config = load_config(config_path, path='configs/CNN')

# load model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
# print(model.summary())

# load data
data = load_coco_semantic_annotations(config)
print("[Loading] shape x", np.shape(data[0]))
print("[Loading] shape label", np.shape(data[1]))
print("[loading] finish loading data")
print()

# compute face units
sem_idx_path = os.path.join("models/saved/", config["config_name"], 'semantic_dictionary.pkl')
if load and os.path.exists(sem_idx_path):
    with open(sem_idx_path, 'rb') as f:
        sem_idx_list = pickle.load(f)
    print("[IoU] Loaded semantic index from {}".format(sem_idx_path))
else:
    print("[IoU] Start computing semantic units")
    sem_idx_list = find_semantic_units(model, data, config, save=save)
    print("[IoU] Computed semantic units")
print()

# get index per category
print("[Index] Load categories")
categories = load_coco_categories(config, verbose=True)
cat_of_interest = ["eyebrow", "lips"]
cat_ids = get_coco_cat_ids(config, cat_of_interest, to_numpy=True)
print("[Index] cat_ids", cat_ids)

# translate IoU to CNN indexes
cat_feature_map_indexes = get_IoU_per_category(sem_idx_list, cat_ids)
print("[Index] Computed category index for {}".format(cat_ids))

# get layer idx
cat_id_of_eyebrow = cat_ids[0]
cat_id_of_lips = cat_ids[1]
layer_of_interest = config["v4_layer"]
# get indexes from dictionary
feature_map_of_eyebrow = cat_feature_map_indexes["category_{}".format(cat_id_of_eyebrow)][layer_of_interest]["indexes"]
feature_map_of_lips = cat_feature_map_indexes["category_{}".format(cat_id_of_lips)][layer_of_interest]["indexes"]
print("[Index] feature map indexes for category {} at layer {}: {}".format(cat_id_of_eyebrow, layer_of_interest, feature_map_of_eyebrow))
print("[Index] num of selected feature map: {}".format(len(feature_map_of_eyebrow)))
print("[Index] feature map indexes for category {} at layer {}: {}".format(cat_id_of_lips, layer_of_interest, feature_map_of_lips))
print("[Index] num of selected feature map: {}".format(len(feature_map_of_lips)))
print()

# test selected feature maps on our morphing sequence
print("[TEST] Test semantic selection on morphing space")
data = load_data(config)
raw_data = load_data(config, get_raw=True)[0]
print("[TEST] shape data[0]", np.shape(data[0]))

# predict sequence
print("[TEST] Predict sequence")
# cut model
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
preds = model.predict(data)
print("shape preds", np.shape(preds))

# Right now
# temporal 3 frames (filt3) sequence filtering
# dynamic actiavtion by removing the neutral pose
# ReLu activations of dynamic filter
# compute positions with weighted average
# todo per feature mal filtering
# todo sorting strongest feature map index

# keep only selected feature maps
preds_eyebrow = preds[:, :, :, feature_map_of_eyebrow]
# preds_eyebrow[preds_eyebrow < 1000] = 0  # todo per feature map cleaning
print("[TEST] num of feature map for eyebrow", len(feature_map_of_eyebrow))
preds_lips = preds[:, :, :, feature_map_of_lips]
print("[TEST] num of feature map for lips", len(feature_map_of_lips))

# compute dynamic changes
preds_eyebrow_init = preds_eyebrow[0]
print("shape preds_eyebrow_init", np.shape(preds_eyebrow_init))
dyn_preds_eyebrow = preds_eyebrow - np.repeat(np.expand_dims(preds_eyebrow_init, axis=0), np.shape(preds_eyebrow)[0], axis=0)
print("shape dyn_preds_eyebrow", np.shape(dyn_preds_eyebrow))
dyn_preds_eyebrow[dyn_preds_eyebrow < 0] = 0

# compute position vectors
eye_brow_pos = calculate_position(preds_eyebrow, mode="weighted average", return_mode="array")
print("shape eye_brow_pos", np.shape(eye_brow_pos))
dyn_eye_brow_pos = calculate_position(dyn_preds_eyebrow[1:], mode="weighted average", return_mode="array")

if do_plot:
    # # plot feature maps
    # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    #                 config['v4_layer'] + "_eye_brow.gif",
    #                 image=raw_data,
    #                 video=True,
    #                 highlight=feature_map_of_eyebrow)
    #
    # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    #                 config['v4_layer'] + "_eye_lips.gif",
    #                 image=raw_data,
    #                 video=True,
    #                 highlight=feature_map_of_lips)
    #
    # plot selection of feature maps
    plot_cnn_output(preds_eyebrow, os.path.join("models/saved", config["config_name"]),
                    config['v4_layer'] + "_eye_brow_selection.gif",
                    image=raw_data,
                    video=True,
                    verbose=False)
    print("[TEST] Finished plotting eyebrow selection")
    #
    # plot_cnn_output(preds_lips, os.path.join("models/saved", config["config_name"]),
    #                 config['v4_layer'] + "_lips_selection.gif",
    #                 image=raw_data,
    #                 video=True)

    # plot dynamic selection
    plot_cnn_output(dyn_preds_eyebrow, os.path.join("models/saved", config["config_name"]),
                    config['v4_layer'] + "_dyn_eye_brow_selection.gif",
                    image=raw_data,
                    video=True,
                    verbose=False)
    print("[TEST] Finished plotting dynamic eyebrow selection")

    # # plot positions
    # plot_cnn_output(eye_brow_pos, os.path.join("models/saved", config["config_name"]),
    #                 config['v4_layer'] + "_eye_brow_select_pos.gif",
    #                 image=raw_data,
    #                 video=True,
    #                 verbose=False)

    # plot positions
    plot_cnn_output(dyn_eye_brow_pos, os.path.join("models/saved", config["config_name"]),
                    config['v4_layer'] + "_dyn_eye_brow_select_pos.gif",
                    image=raw_data,
                    video=True,
                    verbose=False)
    print("[TEST] Finished plotting dynamic eyebrow selection position")

print("[TEST] highlight eyebrow:", feature_map_of_eyebrow)
print("[TEST] highlight lips:", feature_map_of_lips)
print()