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
            print("[SEM-PAT] Start Fitting Semantic Features")
            preds = self.semantic.fit(data, activation=None, feature_channel_last=False)  # activation needs to be set to None to get the raw output
            print("[SEM-PAT] Semantic fitted")
        else:
            print("[SEM-PAT] Start Transforming Semantic Features")
            print()
            print("[SEM-PAT] !!!!! LOADING SEMANTIC MODEL !!!!!!")
            print("[SEM-PAT] BE SURE THAT THIS IS THE BEHAVIOUR YOU WANT")
            print()
            self.semantic.load(os.path.join("models/saved", self.config['config_name'], "NormBase"))
            preds = self.semantic.transform(data, activation=None, feature_channel_last=False)
            print("[SEM-PAT] Semantic Transformed")
        print()

        # extend dimension to fit the number of template
        preds = np.repeat(np.expand_dims(preds, axis=0), len(self.config['rbf_template']), axis=0)

        # fit patterns
        if fit_pattern:
            print("[SEM-PAT] Start Fitting Pattern Features")
            preds = self.pattern.fit(preds, feature_channel_last=feature_channel_last)
            print("[SEM-PAT] Pattern fitted")
        else:
            print("[SEM-PAT] Start Transforming Pattern Features")
            preds = self.pattern.transform(preds, feature_channel_last=feature_channel_last)
            print("[SEM-PAT] Pattern transformed")

        return preds

    def transform(self, data, activation=None, feature_channel_last=True, use_scales=False):
        print("[SEM-PAT] Transform")
        preds = self.semantic.transform(data, activation=None, feature_channel_last=False)

        # extend dimension to fit the number of template
        preds = np.repeat(np.expand_dims(preds, axis=0), len(self.config['rbf_template']), axis=0)

        preds = self.pattern.transform(preds, feature_channel_last=feature_channel_last, use_scales=use_scales)
        print("[SEM-PAT] Semantic Pattern Transformed")

        return preds

    def update_patterns(self, template, sigmas, mask=None, zeros=None):
        self.pattern.update_patterns(template, sigmas, mask=mask, zeros=zeros)

    def save(self, path):
        self.semantic.save(path)
        self.pattern.save(path)

    def load(self, path):
        self.semantic.load(path)
        self.pattern.load(path)
