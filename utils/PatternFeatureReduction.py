import os
import pickle
import numpy as np

from models.RBF import RBF

"""
template: position where the RBF kernel is learn
mask: position (size) of the receptive field
zeros: fine tuning to allow deleting some corners as to create more specific receptive fields (rotation etc) 
"""

# todo create better receptive field


class PatternFeatureSelection:
    """
    class to implement the pattern feature selection pipeline

    the script fits a spatial template over n dimension
    """
    def __init__(self, config, template=None, mask=None, zeros=None):
        self.config = config

        # declare one rbf template per mask
        if template is not None:
            self.template = np.array(template)
        else:
            if config.get('rbf_template') is not None:
                self.template = np.array(config['rbf_template'])
            else:
                print("[PATTERN] No template found!")

        self.n_template = len(self.template)
        self.rbf = []

        sigmas = config['rbf_sigma']

        # st the "receptieve field" of the pattern
        self.use_mask = False
        if mask is not None:
            self.mask = np.array(mask)
            self.use_mask = True
        elif config.get('rbf_mask') is not None:
            self.mask = np.array(config['rbf_mask'])
            self.use_mask = True

        self.use_zeros = False
        if zeros is not None:
            self.zeros = zeros
            self.use_zeros = True

        # if only one sigma is provided just repeat it and set the same sigma for all RBF
        if np.isscalar(sigmas):
            sigmas = np.repeat(sigmas, self.n_template)
        elif len(sigmas) == 1:
            sigmas = np.repeat(sigmas[0], self.n_template)

        for i in range(self.n_template):
            sigma = sigmas[i]
            self.rbf.append(RBF(config, sigma=sigma))

    def fit(self, data, activation=None, feature_channel_last=True):
        """
        data is a list of length (n_pattern)

        :param data: list (n_pattern)(n_data, n_feature, n_feature, n_dim)
        :return:
        """
        print("[PATTERN] Fit pattern")
        # apply mask
        if self.use_mask:
            print("[PATTERN] fit pattern: use mask ON")
            data = self._apply_mask(data)

        # apply zeros
        if self.use_zeros:
            print("[PATTERN] fit pattern: use zeros ON")
            data = self._apply_zeros(data)

        # compute template
        for i in range(len(data)):
            pred = np.array(data[i])

            # compute template
            template = pred[self.config['rbf_template_ref_frame_idx'], self.template[i, 0, 0]:self.template[i, 0, 1],
                               self.template[i, 1, 0]:self.template[i, 1, 1]]
            template = np.expand_dims(template, axis=0)

            # fit rbf template
            self.rbf[i].fit2d(template)

        return self.transform(data, feature_channel_last=feature_channel_last, from_fit=True)

    def transform(self, data, activation=None, feature_channel_last=True, from_fit=False):
        """
        compute activation over the data with the rbf template

        :param data: (n_pattern, n_data, n_feat, n_feat, n_dim)
        :return:
        """
        print("[PATTERN] Transform Pattern")
        if not from_fit:
            # apply mask
            if self.use_mask:
                print("[PATTERN] Transform: use mask")
                data = self._apply_mask(data)

            # apply zeros
            if self.use_zeros:
                print("[PATTERN] Transform: use zeros")
                data = self._apply_zeros(data)

        # transform data
        preds = []
        for i in range(len(data)):
            pred = np.array(data[i])
            preds.append(self.rbf[i].predict2d(pred))

        # transform to (n_data, n_feature, n_feature, n_pattern)
        if feature_channel_last:
            preds = np.moveaxis(preds, 0, -1)

            num_data = len(preds)
            preds = np.squeeze(preds)

            # make sure that preds remains a 4-dimensional array even when there's only one data
            if num_data == 1:
                preds = np.expand_dims(preds, axis=0)

            # add the fourth dimension if there's only one
            if len(np.shape(preds)) == 3:
                preds = np.expand_dims(preds, axis=3)

        print("[PATTERN] prediction transformed!")
        return preds

    def _apply_mask(self, data):
        preds = np.zeros(np.shape(data))
        for i in range(self.n_template):
            preds[i, :, self.mask[i, 0, 0]:self.mask[i, 0, 1], self.mask[i, 1, 0]:self.mask[i, 1, 1]] = \
                data[i, :, self.mask[i, 0, 0]:self.mask[i, 0, 1], self.mask[i, 1, 0]:self.mask[i, 1, 1]]

        print("[PATTERN] apply mask - shape preds", np.shape(preds))
        return preds

    def _apply_zeros(self, data):
        preds = data
        for i in range(len(self.zeros)):
            dict = self.zeros[str(i)]
            idx = dict['idx']
            pos = np.array(dict['pos'])
            preds[idx, :, pos[0, 0]:pos[0, 1], pos[1, 0]:pos[1, 1]] = 0

        print("[PATTERN] apply zeros - shape preds", np.shape(preds))
        return preds

    def save(self, path):
        # save only the rbf patterns
        pickle.dump(self.rbf,
                    open(os.path.join(path, "pattern_rbf.pkl"), 'wb'))
        print("[PATTERN] RBF templates saved")

    def load(self, path):
        self.rbf = pickle.load(open(os.path.join(path, "pattern_rbf.pkl"), 'rb'))
        print("[PATTERN] RBF templates loaded")
