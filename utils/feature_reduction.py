import os
import pickle
import tensorflow as tf
from sklearn.decomposition import PCA

from utils.CSV_data_generator import CSVDataGen
from utils.Semantic.SemanticFeatureSelection import SemanticFeatureSelection


def set_feature_selection(model, config):
    """
    helper function to set the feature reduction pipeline

    :param model:
    :param config:
    :return:
    """
    if model.dim_red is None:
        # initialize as output of network as default
        if len(model.shape_v4) == 2:  # use flatten... but model.v4.layers[-1].output is a tensorShape object
            model.n_features = model.shape_v4[1]
        elif len(model.shape_v4) == 3:
            model.n_features = model.shape_v4[1] * model.shape_v4[2]
        elif len(model.shape_v4) == 4:
            model.n_features = model.shape_v4[1] * model.shape_v4[2] * model.shape_v4[3]
        else:
            raise NotImplementedError("Dimensionality not implemented")
    elif model.dim_red == "PCA":
        model.pca = PCA(n_components=config['PCA'])
        # initialize n_features as number of components of PCA
        model.n_features = config['PCA']
    elif model.dim_red == "position":
        model.position_method = config['position_method']
        # initialize n_features as number of feature maps*2
        model.n_features = model.shape_v4[-1] * 2
    elif model.dim_red == "semantic":
        model.n_features = len(config["semantic_units"])
        model.semantic_feat_red = SemanticFeatureSelection(config)
    else:
        raise ValueError("Dimensionality reduction {} is not implemented".format(model.dim_red))


def fit_dimensionality_reduction(model, data):
    """
    Helper function to fit the model dimensionality reduction set in the config

    :param data: input data
    :return:
    """
    # in the case of dimensionality reduction set up the pipeline
    if model.dim_red is None:
        print("[FIT] no dimensionality reduction")
        preds = data
    elif model.dim_red == 'PCA':
        # perform PCA on this output
        print("[FIT] Fitting PCA")
        model.pca.fit(data)
        print("[FIT] PCA: explained variance", model.pca.explained_variance_ratio_)
        preds = model.pca.transform(data)

    elif model.dim_red == 'position':
        print(f'[FIT] dimensionality reduction method: position with calculation method {model.position_method}')

    elif model.dim_red == "semantic":
        print("[FIT] Finding semantic units")
        model.semantic_feat_red.fit(data)
        print("[FIT] Finished to find the semantic units")
    else:
        raise KeyError(f'model.dim_red={model.dim_red} is not a valid value')

    return preds


def predict_dimensionality_reduction(model, data):
    """
    Helper function to apply the dimensionality reduction as set in the configuation before the model

    :param model: model instance
    :param data:
    :return: prediction
    """
    if model.dim_red is None:
        # get prediction after cnn, before dimensionality reduction
        preds = data
    elif model.dim_red == 'PCA':
        # projection by PCA
        preds = model.pca.transform(data)
    else:
        raise KeyError(f'invalid value self.dim_red={model.dim_red}')

    return preds


def save_feature_selection(model, save_folder):
    """
    Save feature selection pipeline

    :param model:
    :param save_folder:
    :return:
    """
    # save feature reduction
    if model.dim_red == 'PCA':
        print("[SAVE] Save PCA")
        pickle.dump(model.pca, open(os.path.join(save_folder, "pca.pkl"), 'wb'))
    elif model.dim_red == 'semantic':
        print("[SAVE] Save semantic units dictionary")
        # save only the semantic dictionary
        pickle.dump(model.semantic_feat_red.sem_idx_list,
                    open(os.path.join(save_folder, "semantic_dictionary.pkl"), 'wb'))


def load_feature_selection(model, load_folder):
    """
    load feature selection pipeline

    :return:
    """

    if model.dim_red == 'PCA':
        model.pca = pickle.load(open(os.path.join(load_folder, "pca.pkl"), 'rb'))
    if model.dim_red == 'semantic':
        # load the semantic index dictionary
        sem_idx_list = pickle.load(open(os.path.join(load_folder, "semantic_dictionary.pkl"), 'rb'))
        model.semantic_feat_red.sem_idx_list = sem_idx_list
