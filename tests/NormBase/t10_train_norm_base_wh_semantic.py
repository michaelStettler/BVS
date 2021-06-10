import numpy as np
import os

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase
from utils.Semantic.load_coco_semantic_annotations import get_coco_cat_ids
from utils.Semantic.find_semantic_units import get_IoU_per_category
from plots_utils.plot_semantic import plot_semantic_histogram
from plots_utils.plot_semantic import plot_semantic_stacked_bar


"""
script to test the semantic feature selection within the NormBase mechanism

run: python -m tests.NormBase.t10_train_norm_base_wh_semantic
"""

# load config
config_path = 'NB_t10_train_norm_base_wh_semantic_m0002.json'
config = load_config(config_path, path='configs/norm_base_config')

load_model = True
if load_model:
    load_NB_model = True
    fit_dim_red = False
else:
    load_NB_model = False
    fit_dim_red = True

# set read-out layer
config["v4_layer"] = "block4_conv3"

# declare model
model = NormBase(config, input_shape=tuple(config['input_shape']), load_NB_model=load_NB_model)  # load model set to True so we get the semantic dictionary loaded

if not load_model:
    # fit model
    data = load_data(config, train=True)
    model.fit(data,
              fit_dim_red=fit_dim_red,  # if it is the first time to run this script change this to True -> time consuming
              fit_ref=True,
              fit_tun=True)
    model.save()

# show semantic index
# get category IDs of interest (transform semantic units name to category id of COCO)
cat_ids = get_coco_cat_ids(config, config['semantic_units'], to_numpy=True)

# get CNN feature map indexes (gather all feature map index across the whole architecture for each category of interest)
cat_feature_map_indexes = get_IoU_per_category(model.feat_red.sem_idx_list, cat_ids)

categories = ["blue", "red", "green", "yellow", "braun", "beige", "black", "white", "eye lids", "face", "eyes", "hair",
              "mouth", "eyebrow", "teeth", "head", "ears", "nose", "background", "human", "lips", "clothes", "orange"]

layers_of_interest = ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4',
                      'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4']
counts = []
xlabels = []
for i, layer in enumerate(layers_of_interest):
    print("layer:", layer)
    layer_count = []
    for cat_id in cat_ids:
        # ft_index = cat_feature_map_indexes["category_{}".format(cat_id)][config['v4_layer']]["indexes"]
        ft_index = cat_feature_map_indexes["category_{}".format(cat_id)][layer]["indexes"]
        print("cat id:", cat_id, categories[cat_id])
        print("ft_index:", len(ft_index))
        print(ft_index)
        layer_count.append(len(ft_index))

        # save labels
        if i == 0:
            xlabels.append(categories[cat_id])

    counts.append(layer_count)
    print()

counts = np.array(counts)
# plot_semantic_histogram(counts, xlabels=xlabels, ylabels=layers_of_interest, save_folder=os.path.join('models/saved', config['config_name']))
plot_semantic_stacked_bar(counts, xlabels=xlabels, legend=layers_of_interest, save_folder=os.path.join('models/saved', config['config_name']))