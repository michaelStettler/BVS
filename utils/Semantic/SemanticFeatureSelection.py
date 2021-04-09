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
        print("prout")
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
            print("TODO WOULOUHOUOUOUOUOU")

            # get category IDs of interest
            cat_ids = get_coco_cat_ids(config, cat_of_interest, to_numpy=True)

            # translate categories IDs to CNN feature map indexes
            cat_feature_map_indexes = get_IoU_per_category(sem_idx_list, cat_ids)  # todo save this in fit?

            # transform predictions
            # todo add all parameters to config!
            preds = get_feat_map_filt_preds(preds,
                                                  ft_idx,
                                                  ref_type="self0",
                                                  norm=1000,
                                                  activation='ReLu',
                                                  filter='spatial_mean',
                                                  verbose=True)

        return preds

    # def save(self, path):
    #     with open(os.path.join(path, 'semantic_dictionary.pkl'), 'wb') as f:
    #         pickle.dump(self.sem_idx_list, f, pickle.HIGHEST_PROTOCOL)
