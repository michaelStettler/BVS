import os
import numpy as np
import pickle
from utils.load_config import load_config
from utils.load_data import load_data
from utils.load_coco_semantic_annotations import load_coco_semantic_annotations
from utils.load_coco_semantic_annotations import load_coco_categories
from utils.load_coco_semantic_annotations import get_coco_cat_ids
from utils.load_extraction_model import load_extraction_model
from utils.find_semantic_units import find_semantic_units
from utils.find_semantic_units import get_IoU_per_category

"""
test script to try the find_semantic_function that implement the paper:
todo put paper name

run: python -m tests.CNN.t02_find_semantic_units
"""
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)


config_path = 'CNN_t02_find_semantic_units_m0001.json'
save = True
load = True

# load config
config = load_config(config_path, path='configs/CNN')

# load model
model = load_extraction_model(config)
# print(model.summary())

# load data
data = load_coco_semantic_annotations(config)
print("[Loading] shape x", np.shape(data[0]))
print("[Loading] shape label", np.shape(data[1]))
print("[loading] finish loading data")
print()

# compute face units
sem_idx_path = os.path.join("models/saved/semantic_units", config["config_name"], 'semantic_dictionary.pkl')
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
cat_of_interest = ["eyebrow"]
cat_ids = get_coco_cat_ids(config, cat_of_interest, to_numpy=True)
print("[Index] cat_ids", cat_ids)

# translate IoU to CNN indexes
cat_feature_map_indexes = get_IoU_per_category(sem_idx_list, cat_ids)
print("[Index] Computed category index for {}".format(cat_ids))

# get layer idx
cat_id_of_interest = cat_ids[0]
# layer_of_interest = "block1_conv2"
layer_of_interest = "block3_pool"
# get indexes from dictionary
feature_map_oi = cat_feature_map_indexes["category_{}".format(cat_id_of_interest)][layer_of_interest]["indexes"]
feature_map_oi = np.array(feature_map_oi)[0]  # transform to numpy array
print("[Index] feature map indexes for category {} at layer {}: {}".format(cat_id_of_interest, layer_of_interest, feature_map_oi))
print("[Index] num of selected feature map: {}".format(len(feature_map_oi)))

# test selected feature maps on our morphing sequence
data = load_data(config)
print("[TEST] shape data", np.shape(data))

