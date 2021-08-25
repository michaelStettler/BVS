import os
import pickle
import numpy as np

from utils.Semantic.load_coco_semantic_annotations import load_coco_semantic_annotations
from utils.Semantic.load_coco_semantic_annotations import get_coco_cat_ids
from utils.Semantic.find_semantic_units import find_semantic_units
from utils.Semantic.find_semantic_units import get_IoU_per_category


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
        print("[SEMANTIC] Load Semantic data")
        if self.config["annotation_format"] == "coco":
            semantic_data = load_coco_semantic_annotations(self.config)
        else:
            raise ValueError("Format {} is not yet supported".format(format))
        print("[SEMANTIC] Semantic data loaded")
        return semantic_data

    def fit(self, data, activation=None, feature_channel_last=True):
        semantic_data = self.load_semantic_data()
        # find semantic units
        print("[SEMANTIC] Start finding units")
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
            print("[SEMANTIC] Transform Semantic")
            # get category IDs of interest (transform semantic units name to category id of COCO)
            cat_ids = get_coco_cat_ids(self.config, self.config['semantic_units'], to_numpy=True)

            # get CNN feature map indexes (gather all feature map index across the whole architecture for each category of interest)
            cat_feature_map_indexes = get_IoU_per_category(self.sem_idx_list, cat_ids)

            # build feature map index for the category and layer of interest
            # todo if we want multiple layers ?
            preds = []
            for i, cat_id in enumerate(cat_ids):
                ft_index = cat_feature_map_indexes["category_{}".format(cat_id)][self.config['v4_layer']]["indexes"]
                if len(ft_index) > 0:
                    print("[SEMANTIC] num semantic units for cat_ids ({}) {}: {}".format(
                        cat_id, self.config['semantic_units'][i], len(ft_index)))
                    preds.append(data[..., ft_index])
                else:
                    print("No indexes found for category: {}".format(cat_id))
                    print("You should change layer".format(cat_id))

            # apply activation
            if activation is not None:
                if activation == 'mean':
                    mean_preds = []
                    for pred in preds:
                        mean_preds.append(np.mean(pred, axis=3))
                    mean_preds = np.array(mean_preds)
                    preds = mean_preds

                elif activation == 'max':
                    max_preds = []
                    for pred in preds:
                        max_preds.append(np.max(pred, axis=3))
                    max_preds = np.array(max_preds)
                    preds = max_preds

                else:
                    raise NotImplementedError("Activation [] is not implemented".format(activation))
            else:
                # concatenate all features to last dimension
                concat = np.array(preds[0])
                for i in range(1, len(cat_ids)):
                    concat = np.concatenate((concat, preds[i]), axis=-1)
                preds = concat

            # move first axis to last
            if feature_channel_last:
                preds = np.moveaxis(preds, 0, -1)

        print("[SEMANTIC] Shape preds:", np.shape(preds))
        return preds

    def save(self, path):
        # save only the semantic dictionary
        pickle.dump(self.sem_idx_list,
                    open(os.path.join(path, "semantic_dictionary.pkl"), 'wb'))
        print("[SEMANTIC] Semantic dictionary saved")

    def load(self, path):
        self.sem_idx_list = pickle.load(open(os.path.join(path, "semantic_dictionary.pkl"), 'rb'))
        print("[SEMANTIC] Semantic dictionary loaded")
