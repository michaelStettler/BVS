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

        # set the "receptieve field" of the pattern
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

        # set sigmas
        sigmas = config['rbf_sigma']
        sigmas = self._set_sigmas(sigmas)

        for i in range(self.n_template):
            self.rbf.append(RBF(config, sigma=sigmas[i]))

    def _set_sigmas(self, sigmas):
        # if only one sigma is provided just repeat it and set the same sigma for all RBF
        if np.isscalar(sigmas):
            sigmas = np.repeat(sigmas, self.n_template)
        elif len(sigmas) == 1:
            sigmas = np.repeat(sigmas[0], self.n_template)

        return sigmas

    def fit(self, data, activation=None, feature_channel_last=True, verbose=False):
        """
        data is a list of length (n_pattern)

        :param data: list (n_pattern)(n_data, n_feature, n_feature, n_dim)
        :return:
        """
        if verbose:
            print("[PATTERN] Fit pattern")
        # apply mask
        if self.use_mask:
            if verbose:
                print("[PATTERN] fit pattern: use mask ON")
            data = self._apply_mask(data)

        # apply zeros
        if self.use_zeros:
            if verbose:
                print("[PATTERN] fit pattern: use zeros ON")
            data = self._apply_zeros(data)

        # compute template
        n_template = len(data)
        for i in range(n_template):
            pred = np.array(data[i])

            # compute template
            template = pred[self.config['rbf_template_ref_frame_idx'], self.template[i, 0, 0]:self.template[i, 0, 1],
                               self.template[i, 1, 0]:self.template[i, 1, 1]]
            template = np.expand_dims(template, axis=0)

            # fit rbf template
            self.rbf[i].fit2d(template)

        return self.transform(data, feature_channel_last=feature_channel_last, from_fit=True, verbose=verbose)

    def transform(self, data, activation=None, feature_channel_last=True, from_fit=False, use_scales=False, face_x_scales=None, verbose=False):
        """
        compute activation over the data with the rbf template

        :param data: (n_pattern, n_data, n_feat, n_feat, n_dim)
        :return:
        """
        if verbose:
            print("[PATTERN] Transform Pattern")

        if use_scales:
            print("[PATTERN] rescale masks to scaled version")
            x_scales = [1, 1, 1, 1, 1, 1, 1, .8, .8, .8, .8, .8, .8, .8, .9, .9, .9, .9, .9, .9, .9, 1.1, 1.1, 1.1, 1.1,
                        1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

        if face_x_scales is not None:
            x_scales = face_x_scales

        if not from_fit:
            # apply mask
            if self.use_mask:
                if verbose:
                    print("[PATTERN] Transform: use mask")
                data = self._apply_mask(data, x_scales, verbose=verbose)

            # apply zeros
            if self.use_zeros:
                if verbose:
                    print("[PATTERN] Transform: use zeros")
                data = self._apply_zeros(data, x_scales)

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

        if verbose:
            print("[PATTERN] prediction transformed!")
        return preds

    def _apply_mask(self, data, x_scales=None, verbose=False):
        """
        x_scales allows to define a single mask and then just use the provided scales to apply it on images

        :param data:
        :param x_scales:
        :return:
        """
        preds = np.zeros(np.shape(data))
        if x_scales is None:
            for i in range(self.n_template):
                preds[i, :, self.mask[i, 0, 0]:self.mask[i, 0, 1], self.mask[i, 1, 0]:self.mask[i, 1, 1]] = \
                    data[i, :, self.mask[i, 0, 0]:self.mask[i, 0, 1], self.mask[i, 1, 0]:self.mask[i, 1, 1]]
        else:
            for j in range(np.shape(data)[1]):
                for i in range(self.n_template):
                    mid_x = np.shape(data)[3] / 2
                    x_start = int(mid_x + (self.mask[i, 1, 0] - mid_x) * x_scales[j])
                    x_end = int(mid_x + (self.mask[i, 1, 1] - mid_x) * x_scales[j])

                    preds[i, j, self.mask[i, 0, 0]:self.mask[i, 0, 1], x_start:x_end] = \
                        data[i, j, self.mask[i, 0, 0]:self.mask[i, 0, 1], x_start:x_end]

        if verbose:
            print("[PATTERN] apply mask - shape preds", np.shape(preds))
        return preds

    def _apply_zeros(self, data, x_scales=None):
        preds = data
        if x_scales is None:
            for i in range(len(self.zeros)):
                dict = self.zeros[str(i)]
                idx = dict['idx']
                pos = np.array(dict['pos'])
                preds[idx, :, pos[0, 0]:pos[0, 1], pos[1, 0]:pos[1, 1]] = 0
        else:
            for j in range(np.shape(data)[1]):
                for i in range(len(self.zeros)):
                    mid_x = np.shape(data)[3] / 2
                    dict = self.zeros[str(i)]
                    idx = dict['idx']
                    pos = np.array(dict['pos'])

                    x_start = int(mid_x + (pos[1, 0] - mid_x) * x_scales[j])
                    x_end = int(mid_x + (pos[1, 1] - mid_x) * x_scales[j])

                    preds[idx, :, pos[0, 0]:pos[0, 1], x_start:x_end] = 0

        return preds

    def update_patterns(self, template, sigmas, mask=None, zeros=None):
        # update template positions
        self.template = np.array(template)

        # clear rbf
        self.rbf = []

        # set mask and zeros if provided
        if mask is not None:
            self.mask = np.array(mask)
            self.use_mask = True
        else:
            self.use_mask = False

        if zeros is not None:
            self.zeros = zeros
            self.use_zeros = True
        else:
            self.use_zeros = False

        # set sigmas
        sigmas = self._set_sigmas(sigmas)

        # create new RBF
        for i in range(self.n_template):
            self.rbf.append(RBF(self.config, sigma=sigmas[i]))

    def save(self, path):
        # save only the rbf patterns
        pickle.dump(self.rbf,
                    open(os.path.join(path, "pattern_rbf.pkl"), 'wb'))
        print("[PATTERN] RBF templates saved")

    def load(self, path):
        self.rbf = pickle.load(open(os.path.join(path, "pattern_rbf.pkl"), 'rb'))
        print("[PATTERN] RBF templates loaded")
