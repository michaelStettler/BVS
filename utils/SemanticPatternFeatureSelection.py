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

    def fit(self, data, activation=None, feature_channel_last=True):
        # fit semantic feature map selection
        print("[FIT] Start Fitting Semantic Features")
        preds = self.semantic.fit(data, activation=None, feature_channel_last=False)  # activation needs to be set to None to get the raw output
        print("[FIT] Semantic fitted")

        # fit patterns
        print("[FIT] Start Fitting Pattern Features")
        preds = self.pattern.fit(preds, feature_channel_last=feature_channel_last)
        print("[FIT] Pattern fitted")

        return preds

    def transform(self, data, activation=None, feature_channel_last=True):
        print("[Feat. Select] Transform")
        preds = self.semantic.transform(data, activation=None, feature_channel_last=False)
        preds = self.pattern.transform(preds, feature_channel_last=feature_channel_last)

        return preds

    def save(self, path):
        self.semantic.save(path)
        self.pattern.save(path)

    def load(self, path):
        self.semantic.load(path)
        self.pattern.load(path)
