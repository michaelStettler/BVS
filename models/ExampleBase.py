import os
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

from utils.CNN.extraction_model import load_extraction_model
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
        # -----------------------------------------------------------------
        # limit GPU memory as it appear windows have an issue with this, from:
        # https://forums.developer.nvidia.com/t/could-not-create-cudnn-handle-cudnn-status-alloc-failed/108261/3
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        # ----------------------------------------------------------------

        # declare variable
        self.config = config
        self.input_shape = input_shape

        # set dimensionality reduction
        try:
            self.dim_red = config['dim_red']
        except KeyError:
            self.dim_red = None

        # load front end feature extraction model
        self._load_v4(config, input_shape)  # load extraction model
        print()
        print("[INIT] -- Model loaded --")
        print("[INIT] Extraction Model:", config['extraction_model'])
        print("[INIT] V4 layer:", config['v4_layer'])
        if not (self.dim_red is None):
            print("[INIT] dim_red:", self.dim_red)
        self.shape_v4 = np.shape(self.v4.layers[-1].output)
        print("[INIT] shape_v4", self.shape_v4)

        # initialize n_features based on dimensionality reduction method
        self._set_feature_reduction(config)
        print("[INIT] n_features:", self.n_features)

        # load norm base model
        if load_EB_model is not None:
            self.load()
            print("[INIT] Example Based model has been loaded from file: {}".format(config['config_name']))

    def _load_v4(self, config, input_shape):
        """
        load the v4 pipeline extraction feature

        :param config:
        :param input_shape:
        :return:
        """
        if (config['extraction_model'] == 'VGG19') | (config['extraction_model'] == 'ResNet50V2'):
            self.model = load_extraction_model(config, input_shape)
            self.v4 = tf.keras.Model(inputs=self.model.input,
                                outputs=self.model.get_layer(config['v4_layer']).output)
        else:
            raise ValueError("model: {} does not exists! Please change config file or add the model"
                             .format(config['extraction_model']))
        # define preprocessing for images
        if config['extraction_model'] == 'VGG19':
            self.preprocessing = 'VGG19'
        else:
            self.preprocessing = None
            warnings.warn(f'no preprocessing for images defined for config["model"]={config["extraction_model"]}')

    def _set_feature_reduction(self, config):
        set_feature_selection(self, config)

    # ------------------------------------------------------------------------------------------------------------------
    # Save and Load
    def save(self, config=None):
        if config is None:
            config = self.config
        print("save")
        # create folder if it does not exist
        save_folder = os.path.join("models/saved", config['config_name'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        save_folder = os.path.join(save_folder, "ExampleBase")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # save feature reduction
        save_feature_selection(self, save_folder)

    def load(self):
        load_folder = os.path.join("models/saved", self.config['config_name'], "ExampleBase")

        # load feature reduction
        load_feature_selection(self, load_folder)

    # ------------------------------------------------------------------------------------------------------------------
    # fit / train functions
    def predict_v4(self, data, flatten=True):
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

    def fit(self, data, batch_size=32, fit_dim_red=True, fit_snapshot=True):
        """
        fit function. We can select what part of the model we want to train.

        Note: the Amary field can be tuned but is not trained!

        :param data:
        :param batch_size:
        :param fit_dim_red:
        :param fit_snapshot:
        :return:
        """
        print("fit")
        if fit_dim_red:
            self._fit_dim_red(data)

    def _fit_dim_red(self, data):
        """
        fit dimensionality reduction selected by config
        :param data: input data
        :return:
        """
        fit_dimensionality_reduction(self, data)
