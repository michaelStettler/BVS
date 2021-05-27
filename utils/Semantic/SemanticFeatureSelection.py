import os
import pickle
import numpy as np

from utils.Semantic.load_coco_semantic_annotations import load_coco_semantic_annotations
from utils.Semantic.load_coco_semantic_annotations import load_coco_categories
from utils.Semantic.load_coco_semantic_annotations import get_coco_cat_ids
from utils.Semantic.find_semantic_units import find_semantic_units
from utils.Semantic.find_semantic_units import get_IoU_per_category
from utils.feat_map_filter_processing import get_feat_map_filt_preds


class SemanticFeatureSelection:
    """
    class to implement the semantic feature selection pipeline
    """
    def __init__(self, config, model):
        self.config = config

        # declare variables
        self.sem_idx_list = None

        self.model = model

    def load_semantic_data(self):
        # load semantic data
        print("[FIT] Load Semantic data")
        if self.config["annotation_format"] == "coco":
            semantic_data = load_coco_semantic_annotations(self.config)
        else:
            raise ValueError("Format {} is not yet supported".format(format))
        print("[FIT] Semantic data loaded")
        return semantic_data

    def fit(self, data, activation=None, feature_channel_last=True):
        semantic_data = self.load_semantic_data()
        # find semantic units
        print("[FIT] Start computing semantic units")
        self.sem_idx_list = find_semantic_units(self.model, semantic_data, self.config)

        return self.transform(data, activation=activation, feature_channel_last=feature_channel_last)

    def transform(self, data, activation=None, feature_channel_last=True):
        """
        activation alloes to apply a activation layer
        feature_channel_last allow to follow the convention of (n_data, ft_size, ft_size, n_features)

        :param data:
        :param keep_dim:
        :return:
        """
        if self.sem_idx_list is None:
            raise ValueError("Semantic features units are None! Please either train the Semantic Feature Selection or "
                             "load a pre-trained dictionary")
        else:
            # get category IDs of interest (transform semantic units name to category id of COCO)
            cat_ids = get_coco_cat_ids(self.config, self.config['semantic_units'], to_numpy=True)

            # get CNN feature map indexes (gather all feature map index across the whole architecture for each category of interest)
            cat_feature_map_indexes = get_IoU_per_category(self.sem_idx_list, cat_ids)

            # build feature map index for the category and layer of interest
            # todo if we want multiple layers ?
            preds = []
            for cat_id in cat_ids:
                ft_index = cat_feature_map_indexes["category_{}".format(cat_id)][self.config['v4_layer']]["indexes"]
                preds.append(data[..., ft_index])

            # apply activation
            if activation is not None:
                if activation == 'mean':
                    mean_preds = []
                    for pred in preds:
                        mean_preds.append(np.mean(pred, axis=3))
                    mean_preds = np.array(mean_preds)
                    preds = mean_preds
                else:
                    raise NotImplementedError("Activation [] i snot implemented".format(activation))

            # shuffle dimensions
            if feature_channel_last:
                preds = np.moveaxis(preds, 0, -1)

        return preds

    def save(self, path):
        # save only the semantic dictionary
        pickle.dump(self.sem_idx_list,
                    open(os.path.join(path, "semantic_dictionary.pkl"), 'wb'))
        print("[SAVE] Semantic dictionary saved")

    def load(self, path):
        self.sem_idx_list = pickle.load(open(os.path.join(path, "semantic_dictionary.pkl"), 'rb'))
        print("[LOAD] Semantic dictionary loaded")
