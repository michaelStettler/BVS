import os
import pickle

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
    def __init__(self, config):
        self.config = config

        # declare variables
        self.sem_idx_list = None

    def fit(self, model):
        # load semantic data
        print("[FIT] Load Semantic data")
        if self.config["annotation_format"] == "coco":
            data = load_coco_semantic_annotations(self.config)
        else:
            raise ValueError("Format {} is not yet supported".format(format))
        print("[FIT] Semantic data loaded")

        # find semantic units
        print("[FIT] Start computing semantic units")
        self.sem_idx_list = find_semantic_units(model, data, self.config)

    def transform(self, preds):
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
            ft_index = []
            for cat_id in cat_ids:
                ft_index.append(cat_feature_map_indexes["category_{}".format(cat_id)][self.config['v4_layer']]["indexes"])

            # transform predictions
            print("todo change the reference for the feature map filtering!!!!!!!!!!!!!!!")
            preds = get_feat_map_filt_preds(preds,
                                            ft_index,
                                            ref_type="self0",
                                            norm=self.config['feat_map_filt_norm'],
                                            activation=self.config['feat_map_filt_activation'],
                                            filter=self.config['feat_map_filter_type'])

        return preds

    # def save(self, path):
    #     with open(os.path.join(path, 'semantic_dictionary.pkl'), 'wb') as f:
    #         pickle.dump(self.sem_idx_list, f, pickle.HIGHEST_PROTOCOL)
