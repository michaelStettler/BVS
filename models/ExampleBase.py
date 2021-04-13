import os
import pickle
import numpy as np

from models.RBF import RBF
from models.Amari import Amari
from utils.extraction_model import load_v4
from utils.feature_reduction import set_feature_selection
from utils.feature_reduction import save_feature_selection
from utils.feature_reduction import load_feature_selection
from utils.feature_reduction import fit_dimensionality_reduction


class ExampleBase:
    """
    The example base model is a physiologically inspired model that comes from recognition of body social interaction in
    the FIND REFERENCE!!!

    It uses the so called snapshot neurons (RBF) followed by a neural field (AMARY) to model the dynamic

    """

    def __init__(self, config, input_shape, load_EB_model=None):
        # declare variable
        self.config = config
        self.input_shape = input_shape

        # set dimensionality reduction
        try:
            self.dim_red = config['dim_red']
        except KeyError:
            self.dim_red = None

        self.normalize = True  # todo add to config
        self.norm = None

        # load front end feature extraction model
        load_v4(self, config, input_shape)  # load extraction model
        print()
        print("[INIT] -- Model loaded --")
        print("[INIT] Extraction Model:", config['extraction_model'])
        print("[INIT] V4 layer:", config['v4_layer'])
        if not (self.dim_red is None):
            print("[INIT] dim_red:", self.dim_red)
        self.shape_v4 = np.shape(self.v4.layers[-1].output)
        print("[INIT] shape_v4", self.shape_v4)
        self.snapshots = RBF(config)

        # initialize n_features based on dimensionality reduction method
        set_feature_selection(self, config)
        print("[INIT] n_features:", self.n_features)

        # initialize Neural Field
        self.neural_field = Amari(config)

        # load norm base model
        if load_EB_model is not None:
            if load_EB_model:
                self.load()

    # ------------------------------------------------------------------------------------------------------------------
    # Save and Load
    def save(self, config=None):
        # modify config if one is given
        if config is None:
            config = self.config

        # create folder if it does not exist
        save_folder = os.path.join("models/saved", config['config_name'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        save_folder = os.path.join(save_folder, "ExampleBase")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # save ExampleBase parameters
        np.save(os.path.join(save_folder, "norm"), self.norm)

        # save feature reduction
        save_feature_selection(self, save_folder)

        # save snapshots
        pickle.dump(self.snapshots, open(os.path.join(save_folder, "snapshots.pkl"), 'wb'))
        print("[SAVE] Snapshot neurons saved")

        print("[SAVE] Example Base Model saved!")
        print()

    def load(self):
        load_folder = os.path.join("models/saved", self.config['config_name'], "ExampleBase")

        if not os.path.exists(load_folder):
            raise ValueError("Loading path does not exists! Please control your path")

        # load ExampleBase parameters
        self.norm = np.load(os.path.join(load_folder, "norm.npy"))

        # load feature reduction
        load_feature_selection(self, load_folder)

        # load snapshots
        self.snapshots = pickle.load(open(os.path.join(load_folder, "snapshots.pkl"), 'rb'))

        print("[LOAD] Example Based model has been loaded from file: {}".format(self.config['config_name']))

    # ------------------------------------------------------------------------------------------------------------------
    # fit / train functions
    def predict_v4(self, data, flatten=True, normalize=True):
        """
        returns prediction of cnn including preprocessing of images
        in ExampleBase, must be used only to train dimensionality reduction and in predict()
        :param data: batch of data
        :param flatten: if True, is flattened
        :return: prediction
        """

        # prediction of v4 layer
        preds = self.v4.predict(data)
        if flatten:
            preds = np.reshape(preds, (np.shape(preds)[0], -1))

        # normalize the data
        if normalize:
            preds /= self.norm

        return preds

    def predict(self, data):
        """
        returns prediction of pipeline before norm-base
        pipeline consists of preprocessing, cnn, and dimensionality reduction
        :param data: batch of data
        :return: prediction
        """
        if self.dim_red is None:
            # get prediction after cnn, before dimensionality reduction
            preds = self.predict_v4(data)
        elif self.dim_red == 'PCA':
            # projection by PCA
            preds = self.pca.transform(self.predict_v4(data))
        else:
            raise KeyError(f'invalid value self.dim_red={self.dim_red}')

        return preds

    def fit(self, data, batch_size=32, fit_normalize=True, fit_dim_red=True, fit_snapshots=True, tune_neural_field=True):
        """
        fit function. We can select what part of the model we want to train.

        Note: the Amary field can be tuned but is not trained!

        :param data:
        :param batch_size:
        :param fit_dim_red:
        :param fit_snapshot:
        :return:
        """
        if fit_normalize:
            print("[FIT] Fitting normalization")
            self._fit_normalize(data)

        if fit_dim_red:
            print("[FIT] Fitting dimensionality reduction")
            self._fit_dim_red(data)

        if fit_snapshots:
            print("[FIT] Fitting Snapshots neurons")
            self._fit_snapshots(data)

        if tune_neural_field:
            print("[FIT] Computing Neural Field")
            self._tune_neural_field(data)

        print("[FIT] Finished training Example Based model!")
        print()

    def _fit_normalize(self, data):
        preds = self.predict_v4(data[0], normalize=False)  # predict without normalizing!
        self.norm = np.amax(preds)

    def _fit_dim_red(self, data):
        """
        fit dimensionality reduction selected by config
        :param data: input data
        :return:
        """
        fit_dimensionality_reduction(self, data)

    def _fit_snapshots(self, data, normalize=True):
        """
        fit the snapshots neurons

        :param data:
        :return:
        """
        preds = self.predict(data[0])
        self.snapshots.fit(preds, verbose=True)

    def _tune_neural_field(self, data):
        """

        :param data:
        :return:
        """
        print("prout :p")

