import os
import pickle
import numpy as np

from utils.Semantic.SemanticFeatureSelection import SemanticFeatureSelection
from utils.PatternFeatureReduction import PatternFeatureSelection


class SemanticPatternFeatureSelection:
    """
    class to implement the sematic and pattern feature selection pipeline

    it allows to find the semantic units using the SemanticFeatureSelection followed
    by fitting a pattern using the PatternFeatureSelection
    """
    def __init__(self, config, model):
        self.config = config

        self.semantic = SemanticFeatureSelection(config, model)
        self.pattern = PatternFeatureSelection(config)

    def fit(self, data, activation=None, feature_channel_last=True, fit_semantic=True, fit_pattern=True):
        # fit semantic feature map selection
        if fit_semantic:
            print("[FIT] Start Fitting Semantic Features")
            preds = self.semantic.fit(data, activation=None, feature_channel_last=False)  # activation needs to be set to None to get the raw output
            print("[FIT] Semantic fitted")
        else:
            print("[FIT] Start Transforming Semantic Features")
            preds = self.semantic.transform(data, activation=None, feature_channel_last=False)
            print("[FIT] Semantic Transformed")

        # extend dimension to fit the number of template
        preds = np.repeat(np.expand_dims(preds, axis=0), len(self.config['rbf_template']), axis=0)

        # fit patterns
        if fit_pattern:
            print("[FIT] Start Fitting Pattern Features")
            preds = self.pattern.fit(preds, feature_channel_last=feature_channel_last)
            print("[FIT] Pattern fitted")
        else:
            print("[FIT] Start Transforming Pattern Features")
            preds = self.pattern.transform(preds, feature_channel_last=feature_channel_last)
            print("[FIT] Pattern transformed")

        return preds

    def transform(self, data, activation=None, feature_channel_last=True):
        print("[Feat. Select] Transform")
        preds = self.semantic.transform(data, activation=None, feature_channel_last=False)

        # extend dimension to fit the number of template
        preds = np.repeat(np.expand_dims(preds, axis=0), len(self.config['rbf_template']), axis=0)

        preds = self.pattern.transform(preds, feature_channel_last=feature_channel_last)
        print("[Feat. Select] Semantic Pattern Transformed")

        return preds

    def save(self, path):
        self.semantic.save(path)
        self.pattern.save(path)

    def load(self, path):
        self.semantic.load(path)
        self.pattern.load(path)
