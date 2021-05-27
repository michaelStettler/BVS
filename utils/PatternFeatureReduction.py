import os
import pickle
import numpy as np

from models.RBF import RBF


class PatternFeatureSelection:
    """
    class to implement the pattern feature selection pipeline

    the script fits a spatial template over n dimension
    """
    def __init__(self, config):
        self.config = config

        # declare one rbf template per mask
        self.mask = np.array(config['pattern_mask'])
        self.n_mask = len(self.mask)
        self.rbf = []

        sigmas = config['rbf_sigma']
        # if only one sigma is provided just repeat it and set the same sigma for all RBF
        if np.isscalar(sigmas):
            sigmas = np.repeat(sigmas, self.n_mask)
        elif len(sigmas) == 1:
            sigmas = np.repeat(sigmas[0], self.n_mask)

        for i in range(self.n_mask):
            sigma = sigmas[i]
            self.rbf.append(RBF(config, sigma=sigma))

    def fit(self, data, activation=None, feature_channel_last=True):
        """
        data is a list of len(n_pattern)

        :param data: list (n_pattern)(n_data, n_feature, n_feature, n_dim)
        :return:
        """
        for i in range(len(data)):
            pred = np.array(data[i])
            # apply mask
            template = pred[self.config['pattern_idx'], self.mask[i, 0, 0]:self.mask[i, 0, 1],
                               self.mask[i, 1, 0]:self.mask[i, 1, 1]]
            template = np.expand_dims(template, axis=0)

            # fit rbf template
            self.rbf[i].fit2d(template)

        return self.transform(data, feature_channel_last=feature_channel_last)

    def transform(self, data, feature_channel_last=True):
        """
        compute activation over the data with the rbf template

        :param data: (n_pattern, n_data, n_feat, n_feat, n_dim)
        :return:
        """
        preds = []
        for i in range(len(data)):
            pred = np.array(data[i])
            preds.append(self.rbf[i].predict2d(pred))

        # transform to (n_data, n_feature, n_feature, n_pattern)
        if feature_channel_last:
            preds = np.squeeze(preds)
            preds = np.moveaxis(preds, 0, -1)
        return preds

    def save(self, path):
        # save only the rbf patterns
        pickle.dump(self.rbf,
                    open(os.path.join(path, "pattern_rbf.pkl"), 'wb'))
        print("[SAVE] Pattern RBF saved")

    def load(self, path):
        self.rbf = pickle.load(open(os.path.join(path, "pattern_rbf.pkl"), 'rb'))
        print("[LOAD] Pattern RBF loaded")
