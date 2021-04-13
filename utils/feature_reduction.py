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
    print("[FIT] dimensionality reduction")
    # in the case of dimensionality reduction set up the pipeline
    if model.dim_red is None:
        print("[FIT] no dimensionality reduction")
    elif model.dim_red == 'PCA':
        if isinstance(data, CSVDataGen):
            # data = data.getAllData()
            raise ValueError("PCA and DataGenerator has to be implemented first to be usable")
        elif isinstance(data, tf.keras.preprocessing.image.ImageDataGenerator):
            model.v4_predict = model.predict_v4(data)
        else:
            model.v4_predict = model.predict_v4(data[0])
        # old (w/o preprocessing):
        # v4_predict = model.v4.predict(data[0])
        # v4_predict = np.reshape(v4_predict, (data[0].shape[0], -1))
        # model.v4_predict = v4_predict

        # perform PCA on this output
        print("[FIT] Fitting PCA")
        model.pca.fit(model.v4_predict)
        print("[FIT] PCA: explained variance", model.pca.explained_variance_ratio_)

    elif model.dim_red == 'position':
        print(f'[FIT] dimensionality reduction method: position with calculation method {model.position_method}')

    elif model.dim_red == "semantic":
        print("[FIT] Finding semantic units")
        model.semantic_feat_red.fit(model.model)
        print("[FIT] Finished to find the semantic units")
    else:
        raise KeyError(f'model.dim_red={model.dim_red} is not a valid value')


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
