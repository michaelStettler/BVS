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
from utils.feature_reduction import predict_dimensionality_reduction


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
        print()

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
    def predict_v4(self, data, flatten=True):
        """
        returns prediction of cnn including preprocessing of images
        in ExampleBase, must be used only to train dimensionality reduction and in predict()
        :param data: batch of data
        :param flatten: if True, is flattened
        :return: prediction
        """

        # prediction of v4 layer
        preds = self.v4.predict(data, verbose=1)
        if flatten:
            preds = np.reshape(preds, (np.shape(preds)[0], -1))

        return preds

    def predict(self, data, get_snapshots=False, get_nn_field=False):
        """
        Predict expression neurons of the model

        :param data:
        :return:
        """
        # predict v4
        v4_preds = self.predict_v4(data[0])
        print("[PREDS] v4 predicted")

        # normalize predictions
        v4_preds /= self.norm
        print("[PREDS] v4 normalized")

        # apply dimensionality reduction
        v4_preds_red = predict_dimensionality_reduction(self, v4_preds)
        print("[PREDS] dimensionality feature reduced")

        # compute snapshot
        snaps = self.snapshots.predict(v4_preds_red)
        print("[PREDS] Snapshot neurons computed")

        # compute expression neurons
        expr_neurons = self._tune_neural_field(snaps, get_nn_field=get_nn_field)
        if get_nn_field:
            nn_field = expr_neurons[1]
            expr_neurons = expr_neurons[0]
        print("[PREDS] Expression neurons computed")

        if get_snapshots:
            if get_nn_field:
                return expr_neurons, snaps, nn_field
            else:
                return expr_neurons, snaps
        else:
            if get_nn_field:
                return expr_neurons, nn_field
            else:
                return expr_neurons

    def fit(self, data, batch_size=32, fit_normalize=True, fit_dim_red=True, fit_snapshots=True, get_snapshots=False,
            get_nn_field=False):
        """
        fit function. We can select what part of the model we want to train.

        The boolean allows to speed up training by re-training only part of the model

        Note: the Amari field can be tuned but it is not trained! -> check Amari class to see how to fine tune its
        parameters

        :param data:
        :param batch_size:
        :param fit_dim_red:
        :param fit_snapshot:
        :return:
        """

        # predict v4 responses
        print("[FIT] Compute v4")
        v4_preds = self.predict_v4(data[0])
        print("[FIT] Shape v4_preds", np.shape(v4_preds))

        if fit_normalize:
            print("[FIT] - Fitting normalization -")
            v4_preds = self._fit_normalize(v4_preds)
        else:
            v4_preds /= self.norm
        print("[FIT] Data normalized")

        if fit_dim_red:
            print("[FIT] - Fitting dimensionality reduction -")
            v4_preds_red = fit_dimensionality_reduction(self, v4_preds)
        else:
            v4_preds_red = predict_dimensionality_reduction(self, v4_preds)
        print("[FIT] Data reduced")

        if fit_snapshots:
            print("[FIT] - Fitting Snapshots neurons -")
            snaps = self.snapshots.fit(v4_preds_red)
        else:
            snaps = self.snapshots.predict(v4_preds_red)
        print("[FIT] Snapshot neurons computed")
        self.snapshots.get_response_statistics(snaps)

        print("[FIT] - Computing Neural Field -")
        expr_neurons = self._tune_neural_field(snaps, get_nn_field=get_nn_field)
        if get_nn_field:
            nn_field = expr_neurons[1]
            expr_neurons = expr_neurons[0]
        print("[FIT] Expression neurons computed")

        print("[FIT] Finished training Example Based model!")
        print()

        if get_snapshots:
            if get_nn_field:
                return expr_neurons, snaps, nn_field
            else:
                return expr_neurons, snaps
        else:
            if get_nn_field:
                return expr_neurons, nn_field
            else:
                return expr_neurons

    def _fit_normalize(self, data):
        """
        compute the maximum of the prediction to normalize over it

        :param data:
        :return:
        """
        self.norm = np.amax(data)

        return data / self.norm

    def _tune_neural_field(self, data, get_nn_field=False):
        """

        :param data:
        :return:
        """
        # reshape snapshots to train/test format
        # todo transform snapshot space! -> think how to do it better
        data = self.snapshots.reshape_preds(data)

        # feed neural field
        nn_field = self.neural_field.predict_neural_field(data)

        # compute expression neurons
        expression_neurons = self.neural_field.predict_dynamic(nn_field)

        if get_nn_field:
            return expression_neurons, nn_field
        else:
            return expression_neurons

    # ------------------------------------------------------------------------------------------------------------------
    # plots
    def plot_snapshots(self, snaps, title=None):
        self.snapshots.plot_rbf_kernel(snaps, save_folder=os.path.join("models/saved", self.config['config_name']),
                                       title=title)

    def plot_nn_kernels(self, title=None):
        self.neural_field.plot_kernels(save_folder=os.path.join("models/saved", self.config['config_name']),
                                       title=title)

    def plot_neural_field(self, nn_field, title=None):
        self.neural_field.plot_neural_field(nn_field,
                                            save_folder=os.path.join("models/saved", self.config['config_name']),
                                            title=title)

    def plot_expression_neurons(self, expr_neurons, title=None, val=False):
        self.neural_field.plot_dynamic(expr_neurons,
                                       save_folder=os.path.join("models/saved", self.config['config_name']),
                                       title=title,
                                       val=val)
