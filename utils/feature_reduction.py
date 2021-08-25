import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

from utils.CSV_data_generator import CSVDataGen
from utils.Semantic.SemanticFeatureSelection import SemanticFeatureSelection
from utils.PatternFeatureReduction import PatternFeatureSelection
from utils.SemanticPatternFeatureSelection import SemanticPatternFeatureSelection
from utils.calculate_position import calculate_position
from utils.feat_map_filter_processing import get_feat_map_filt_preds


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
    elif model.dim_red == "semantic" or model.dim_red == "semantic-pattern" or model.dim_red == "pattern":
        # set number of features depending on the way positions are computed
        if config['feat_map_position_return_mode'] == 'raw':
            model.n_features = len(config["semantic_units"]) * model.shape_v4[1] * model.shape_v4[2]
        elif config['feat_map_position_return_mode'] == 'xy_float':
            model.n_features = len(config["semantic_units"]) * 2
        elif config['feat_map_position_return_mode'] == 'weighted_array':
            model.n_features = len(config["semantic_units"]) * model.shape_v4[1] * model.shape_v4[2]
        elif config['feat_map_position_return_mode'] == 'xy float flat':
            model.n_features = 2 * len(config["rbf_template"])
        else:
            raise NotImplementedError("feat_map_position_mode {} not implemented".format(config["feat_map_position_mode"]))

        # declare semanticFeature object
        if model.dim_red == "semantic":
            model.feat_red = SemanticFeatureSelection(config, model.v4)
        elif model.dim_red == "pattern":
            model.feat_red = PatternFeatureSelection(config)
        elif model.dim_red == "semantic-pattern":
            model.feat_red = SemanticPatternFeatureSelection(config, model.v4)
    else:
        raise ValueError("Dimensionality reduction {} is not implemented".format(model.dim_red))


def fit_dimensionality_reduction(model, data, fit_semantic=True):
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

    elif model.dim_red == "semantic" or model.dim_red == "semantic-pattern" or model.dim_red == "pattern":
        if not fit_semantic:
            # take care of the special case of semantic-pattern as it can still needs to fit the pattern
            if model.dim_red == "semantic-pattern":
                preds = model.feat_red.fit(data, activation='mean', fit_semantic=False)
            else:
                preds = model.feat_red.transform(data, activation='mean')
        else:
            preds = model.feat_red.fit(data, activation='mean')

        # apply filter post_processing
        # todo modify this, I'm not really happy with this post-processing yet
        preds = get_feat_map_filt_preds(preds, model.config)

        # allow to further reduce dimensionality by getting a 2 dim vector for each feature maps
        if model.config['feat_map_position_mode'] != 'raw':
            print("[FIT] Using position mode: {}".format(model.config['feat_map_position_mode']))
            preds = calculate_position(preds,
                                       mode=model.config['feat_map_position_mode'],
                                       return_mode=model.config['feat_map_position_return_mode'])

        preds = np.reshape(preds, (len(preds), -1))
        print("[FIT] Finished to find the semantic units", np.shape(preds))
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
    elif model.dim_red == "semantic" or model.dim_red == "semantic-pattern" or model.dim_red == "pattern":
        preds = model.feat_red.transform(data, activation='mean')

        # apply filter post_processing
        preds = get_feat_map_filt_preds(preds, model.config)

        # allow to further reduce dimensionality by getting a 2 dim vector for each feature maps
        if model.config['feat_map_position_mode'] != 'raw':
            print("[PREDS] Using position mode: {}".format(model.config['feat_map_position_mode']))
            preds = calculate_position(preds,
                                       mode=model.config['feat_map_position_mode'],
                                       return_mode='xy float')
        preds = np.reshape(preds, (len(preds), -1))
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
    elif model.dim_red == 'semantic' or model.dim_red == "semantic-pattern" or model.dim_red == "pattern":
        print("[SAVE] Save feature reduction")
        model.feat_red.save(save_folder)


def load_feature_selection(model, load_folder):
    """
    load feature selection pipeline

    :return:
    """

    if model.dim_red == 'PCA':
        print("[LOAD] load PCA")
        model.pca = pickle.load(open(os.path.join(load_folder, "pca.pkl"), 'rb'))
    if model.dim_red == 'semantic' or model.dim_red == "semantic-pattern" or model.dim_red == "pattern":
        print("[LOAD] load feature reduction")
        model.feat_red.load(load_folder)
