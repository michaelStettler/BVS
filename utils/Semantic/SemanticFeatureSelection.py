import os
import pickle

from utils.Semantic.load_coco_semantic_annotations import load_coco_semantic_annotations
from utils.Semantic.load_coco_semantic_annotations import load_coco_categories
from utils.Semantic.load_coco_semantic_annotations import get_coco_cat_ids
from utils.Semantic.find_semantic_units import find_semantic_units
from utils.Semantic.find_semantic_units import get_IoU_per_category


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

    def transform(self):
        # todo
        print("TODO WOULOUHOUOUOUOUOU")

    def save(self, path):
        print("[SAVE] Semantic units")
        with open(os.path.join(path, 'semantic_dictionary.pkl'), 'wb') as f:
            pickle.dump(self.sem_idx_list, f, pickle.HIGHEST_PROTOCOL)
