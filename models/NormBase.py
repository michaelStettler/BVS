import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.decomposition import PCA
import warnings

from utils.load_extraction_model import load_extraction_model
from utils.CSV_data_generator import CSVDataGen
from utils.calculate_position import calculate_position
from utils.Semantic.SemanticFeatureSelection import SemanticFeatureSelection


class NormBase:
    """
    NormBase class define the functions to train a norm base mechanism with a front end extracting features
    important functions:
    - NormBase(config, input_shape, save_name)
    - save_model(config, save_name)
    - fit(data, batch_size, fit_dim_red, fit_ref, fit_tun)
    - evaluate(data, batch_size)

    r:= reference vector (n_features, )
    t:= tuning vector (n_category, n_features)

    with
    n_features:= height * width * channels of the input_shape

    """

    def __init__(self, config, input_shape, load_NB_model=None):
        """
        The init function is responsible to declare the front end model of the norm base mechanism declared in the
        config file

        :param config: JSON file
        :param input_shape: Tuple with the input size of the model (height, width, channels)
        :param load_NB_model: if given the fitted model is loaded from config["config_name"] subfolder save_name
        """
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

        # declare parameters
        try:
            self.nu = config['nu']
        except KeyError:
            self.nu = 2.0
        # set tuning function
        try:
            self.tun_func = config['tun_func']
        except KeyError:
            self.tun_func = '2-norm'
        # set dimensionality reduction
        try:
            self.dim_red = config['dim_red']
        except KeyError:
            self.dim_red = None

        self.n_category = config['n_category']
        self.ref_cat = config['ref_category']
        self.ref_cumul = 0  # cumulative count of the number of reference frame passed in the fitting

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

        # declare variables
        self.r = np.zeros(self.n_features)  # reference vectors
        self.t = np.zeros((self.n_category, self.n_features))  # tuning vectors
        self.t_cumul = np.zeros(self.n_category)
        self.t_mean = np.zeros((self.n_category, self.n_features))
        threshold_divided = 50
        self.threshold = self.n_features / threshold_divided
        print("[INIT] Neutral threshold ({:.1f}%):".format(100/threshold_divided), self.threshold)

        # v4 prediction of the training data set - used in external script t05_compare_retained_PCA_index.py
        # TODO: eventually delete to save RAM
        self.v4_predict = None

        # load norm base model
        if load_NB_model is not None:
            self._load_NB_model(config, config["config_name"])
            print("[INIT] saved model is loaded from file")

        # set time constant for dynamic and competitive network
        self._set_dynamic(config)

        # set option to save the v4 raw predictions into a csv file
        self.save_preds = False
        self.preds_saved = False  # boolean to know if the predictions have been already saved since the predictions
        # could be computed multiple times depending of the pipeline
        if config.get('save_preds') is not None:
            if config['save_preds']:
                print("[INIT] Save raw prediction set")
                self.save_preds = True
                self.raw_preds_df = pd.DataFrame()
        print()

    def _set_feature_reduction(self, config):
        if self.dim_red is None:
            # initialize as output of network as default
            if len(self.shape_v4) == 2:  # use flatten... but self.v4.layers[-1].output is a tensorShape object
                self.n_features = self.shape_v4[1]
            elif len(self.shape_v4) == 3:
                self.n_features = self.shape_v4[1] * self.shape_v4[2]
            elif len(self.shape_v4) == 4:
                self.n_features = self.shape_v4[1] * self.shape_v4[2] * self.shape_v4[3]
            else:
                raise NotImplementedError("Dimensionality not implemented")
        elif self.dim_red == "PCA":
            self.pca = PCA(n_components=config['PCA'])
            # initialize n_features as number of components of PCA
            self.n_features = config['PCA']
        elif self.dim_red == "position":
            self.position_method = config['position_method']
            # initialize n_features as number of feature maps*2
            self.n_features = self.shape_v4[-1]*2
        elif self.dim_red == "semantic":
            self.n_features = len(config["semantic_units"])
            self.semantic_feat_red = SemanticFeatureSelection(config)
        else:
            raise ValueError("Dimensionality reduction {} is not implemented".format(self.dim_red ))
        print("[INIT] n_features:", self.n_features)

    def _set_dynamic(self, config):
        if config.get('use_dynamic') is not None:
            if config['use_dynamic']:
                print("[INIT] Model Dynamic Set")
                # set parameters for differentiators and decision networks
                self.tau_u = config['tau_u']  # time constant for pos and negative differentiators
                self.tau_v = config['tau_v']  # time constant for IT resp differentiators
                self.tau_y = config['tau_y']  # time constant for integral differentiators
                self.tau_d = config['tau_d']  # time constant for competitive network
                self.m_inhi_w = config['m_inhi_w']  # weights of mutual inhibition

    ### SAVE AND LOAD ###
    def _load_v4(self, config, input_shape):
        if (config['extraction_model'] == 'VGG19') | (config['extraction_model'] =='ResNet50V2'):
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


    def save_NB_model(self, config):
        """
        This method saves the fitted model
        :param config: saves under the specified path in config
        :param save_name: subfolder name
        can be loaded with load_extraction_model(config, save_name)
        """
        print()
        # control that config folder exists
        save_folder = os.path.join("models/saved", config['config_name'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        save_folder = os.path.join(save_folder, "NormBase")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        #save reference and tuning vector
        print("[SAVE] Save reference and tuning vectors")
        np.save(os.path.join(save_folder, "ref_vector"), self.r)
        np.save(os.path.join(save_folder, "tuning_vector"), self.t)
        np.save(os.path.join(save_folder, "tuning_mean"), self.t_mean)
        #save PCA
        if self.dim_red == 'PCA':
            print("[SAVE] Save PCA")
            #np.save(os.path.join(save_folder, "pca"), self.pca)
            pickle.dump(self.pca, open(os.path.join(save_folder, "pca.pkl"), 'wb'))

        if self.save_preds and self.preds_saved:
            print("[SAVE] Save raw prediction to csv")
            self.raw_preds_df.to_csv(os.path.join(save_folder, "raw_pred.csv"))

        print("[SAVE] Norm Base Model saved")
        print()

    def _load_NB_model(self, config, save_name):
        """
        loads trained model from file, including dimensionality reduction
        this method should only be called by init
        :param config: configuration file
        :param save_name: folder name
        :return:
        """
        load_folder = os.path.join("models/saved", config['config_name'], save_name, "NormBase_saved")
        ref_vector = np.load(os.path.join(load_folder, "ref_vector.npy"))
        tun_vector = np.load(os.path.join(load_folder, "tuning_vector.npy"))
        self.set_ref_vector(ref_vector)
        self.set_tuning_vector(tun_vector)
        self.t_mean = np.load(os.path.join(load_folder, "tuning_mean.npy"))
        if self.dim_red == 'PCA':
            self.pca = pickle.load(open(os.path.join(load_folder, "pca.pkl"), 'rb'))

    ### HELPER FUNCTIONS ###

    def set_ref_vector(self, r):
        self.r = r

    def set_tuning_vector(self, t):
        self.t = t

    def evaluate_v4(self, data, flatten=True):
        """
        returns prediction of cnn including preprocessing of images
        in NormBase, must be used only to train dimensionality reduction and in _get_preds()
        :param data: batch of data
        :param flatten: if True, is flattened
        :return: prediction
        """

        # prediction of v4 layer
        preds = self.v4.predict(data)
        if flatten:
            preds = np.reshape(preds, (np.shape(preds)[0], -1))

        # save raw extraction prediction
        if self.save_preds and not self.preds_saved:
            # ensure that the data are flatten for the csv file
            if not flatten:
                save_preds = np.reshape(preds, (np.shape(preds)[0], -1))
            else:
                save_preds = preds
            # save predictions
            df = pd.DataFrame(save_preds)
            self.raw_preds_df = self.raw_preds_df.append(df, ignore_index=True)

        return preds

    def get_preds(self, data):
        """
        returns prediction of pipeline before norm-base
        pipeline consists of preprocessing, cnn, and dimensionality reduction
        :param data: batch of data
        :return: prediction
        """
        if self.dim_red is None:
            # get prediction after cnn, before dimensionality reduction
            preds = self.evaluate_v4(data)
        elif self.dim_red == 'PCA':
            # projection by PCA
            preds = self.pca.transform(self.evaluate_v4(data))
        elif self.dim_red == 'position':
            # receive unflattened prediction
            # receive xy positions and flatten
            # output shape: (batch_size, n_feature_maps*2)
            preds = np.concatenate(calculate_position(
                    self.evaluate_v4(data, flatten=False),
                    mode=self.position_method, return_mode="xy float"),
                axis=1)
        else:
            raise KeyError(f'invalid value self.dim_red={self.dim_red}')
        return preds

    def _update_ref_vector(self, data):
        """
        updates reference vector wrt to new data
        :param data: input of self.ref_cat
        :return:
        """
        n_ref = np.shape(data)[0]
        if n_ref > 0:
            preds = self.get_preds(data)
            # update ref_vector m
            self.r = (self.ref_cumul * self.r + n_ref * np.mean(preds, axis=0)) / (self.ref_cumul + n_ref)
            self.ref_cumul += n_ref

    def _get_reference_pred(self, data):
        """
        returns difference vector between prediction and reference vector
        :param data: batch data
        :return: difference vector
        """
        len_batch = len(data)  # take care of last epoch if size is not equal as the batch_size
        preds = self.get_preds(data)
        # compute batch diff
        return preds - np.repeat(np.expand_dims(self.r, axis=1), len_batch, axis=1).T

    def _update_dir_tuning(self, data, label):
        """
        updates tuning vector wrt to new data
        :param data: input data
        :param label: corresponding labels
        :return:
        """
        # compute batch diff
        batch_diff = self._get_reference_pred(data)

        # compute direction tuning for each category
        for i in range(self.n_category):
            if i != self.ref_cat:
                # get data for each category
                cat_diff = batch_diff[label == i]

                # get num_of data
                n_cat_diff = np.shape(cat_diff)[0]

                if n_cat_diff > 0:
                    # update cumulative mean for each category
                    self.t_mean[i] = (self.t_cumul[i] * self.t_mean[i] + n_cat_diff * np.mean(cat_diff, axis=0)) / \
                                     (self.t_cumul[i] + n_cat_diff)
                    # update cumulative counts
                    self.t_cumul[i] += n_cat_diff

                    # update tuning vector n
                    self.t[i] = self.t_mean[i] / np.linalg.norm(self.t_mean[i])

    def _get_it_resp(self, data):
        """
        computes the activity of norm-based neurons
        for different tuning functions selected in config
        :param data: input data
        :return: activity for each category
        """
        # compute batch diff
        batch_diff = self._get_reference_pred(data)

        # compute norm-reference neurons
        #v = np.sqrt(np.diag(batch_diff @ batch_diff.T))
        if self.tun_func == '2-norm':
            v = np.linalg.norm(batch_diff, ord=2, axis=1)

            f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
            f[f < 0] = 0 #ReLu activation instead of dividing by 2 and adding 0.5
            #f = self.t @ batch_diff.T @ np.diag(np.power(v * 2, -1))
            #f = f+0.5
            f = np.power(f, self.nu)
            return np.diag(v) @ f.T
        elif self.tun_func == '1-norm':
            v = np.linalg.norm(batch_diff, ord=1, axis=1)
            f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
            f[f < 0] = 0  # ReLu activation instead of dividing by 2 and adding 0.5
            f = np.power(f, self.nu)
            return np.diag(v) @ f.T
        elif self.tun_func == 'simplified':
            # this is the function published in the ICAAN paper with 1-norm and nu=1
            return 0.5 * (np.linalg.norm(batch_diff, ord=1, axis=1) + (self.t @ batch_diff.T)).T
        elif self.tun_func == 'direction-only':
            # return normalized scalar product between actual direction and tuning
            v = np.linalg.norm(batch_diff, ord=2, axis=1)
            f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
            f[f<0] = 0
            return f.T
        elif self.tun_func == 'expressivity-direction':
            # tun_func = exprissivity * (direction^nu)
            # expressivity = norm(d) / norm(tun_mean)
            # direction = f (like above)
            # can be simplified by reducing norm(d), but leave it in for better understanding
            # norm for expressivity can be chosen arbitrarily
            v = np.linalg.norm(batch_diff, ord=2, axis=1)
            f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
            f[f < 0] = 0
            f = np.power(f, self.nu)
            t_mean_norm = np.linalg.norm(self.t_mean, ord=2, axis=1)
            batch_diff_norm = np.linalg.norm(batch_diff, ord=2, axis=1)
            expressivity = np.outer(batch_diff_norm,
                                     np.true_divide(1, t_mean_norm, out=np.zeros_like(t_mean_norm), where=t_mean_norm!=0))
            it_resp = expressivity * f.T
            #set response for reference category, to a sphere around reference vector with linear decay
            # activity=1 in the middle, activity=0 at half the distance to the closest category
            it_resp[:, self.ref_cat] = 1 - (0.5 *batch_diff_norm / np.delete(t_mean_norm, self.ref_cat).min())
            it_resp[:, self.ref_cat][it_resp[:, self.ref_cat]<0] = 0
            return it_resp
        else:
            raise ValueError("{} is no valid choice for tun_func".format(self.tun_func))

    def _get_correct_count(self, x, label):
        one_hot_encoder = np.eye(self.n_category)
        one_hot_cats = one_hot_encoder[x]
        one_hot_label = one_hot_encoder[label]

        return np.count_nonzero(np.multiply(one_hot_cats, one_hot_label))

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # ## FIT FUNCTIONS ###
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def fit(self, data, batch_size=32, fit_dim_red=True, fit_ref=True, fit_tun=True):
        """
        fit model on data
        :param data: input
        :param batch_size:
        :param fit_dim_red: whether to fit dimensionality reduction
        :param fit_ref: whether to fit reference vector
        :param fit_tun: whether to fit tuning vector
        :return:
        """
        if fit_dim_red:
            self._fit_dim_red(data)
        if fit_ref:
            self._fit_reference(data, batch_size)
        if fit_tun:
            self._fit_tuning(data, batch_size)

    def _fit_dim_red(self, data):
        """
        fit dimensionality reduction selected by config
        :param data: input data
        :return:
        """
        print("[FIT] dimensionality reduction")
        # in the case of dimensionality reduction set up the pipeline
        if self.dim_red is None:
            print("[FIT] no dimensionality reduction")
        elif self.dim_red == 'PCA':
            if isinstance(data, CSVDataGen):
                # data = data.getAllData()
                raise ValueError("PCA and DataGenerator has to be implemented first to be usable")
            elif isinstance(data, tf.keras.preprocessing.image.ImageDataGenerator):
                self.v4_predict = self.evaluate_v4(data)
            else:
                self.v4_predict = self.evaluate_v4(data[0])
            # old (w/o preprocessing):
            # v4_predict = self.v4.predict(data[0])
            # v4_predict = np.reshape(v4_predict, (data[0].shape[0], -1))
            # self.v4_predict = v4_predict

            # perform PCA on this output
            print("[FIT] Fitting PCA")
            self.pca.fit(self.v4_predict)
            print("[FIT] PCA: explained variance", self.pca.explained_variance_ratio_)

        elif self.dim_red == 'position':
            print(f'[FIT] dimensionality reduction method: position with calculation method {self.position_method}')

        elif self.dim_red == "semantic":
            print("[FIT] Finding semantic units")
            self.semantic_feat_red.fit(self.model)
            print("[FIT] Finished to find the semantic units")
        else:
            raise KeyError(f'self.dim_red={self.dim_red} is not a valid value')

        # set preds_saved to true so the predictions are saved only once
        if self.save_preds and self.dim_red is not None:
            self.preds_saved = True

    def _fit_reference(self, data, batch_size):
        """
        fits only reference vector without fitting tuning vector
        :param data: [x,y] or DataGen
        :param batch_size:
        :return:
        """
        print("[FIT] Learning reference pose")
        # reset reference
        self.r = np.zeros(self.r.shape)
        self.ref_cumul = 0

        if isinstance(data, list):
            num_data = data[0].shape[0]
            indices = np.arange(num_data)
            for b in tqdm(range(0, num_data, batch_size)):
                # built batch
                end = min(b + batch_size, num_data)
                batch_idx = indices[b:end]
                batch_data = data[0][batch_idx]
                ref_data = batch_data[data[1][batch_idx] == self.ref_cat]
                # update reference vector
                self._update_ref_vector(ref_data)

        elif isinstance(data, CSVDataGen):
            # train using a generator
            data.reset()
            for data_batch in tqdm(data.generate()):
                x = data_batch[0]
                y = data_batch[1]
                ref_data = x[y == self.ref_cat]  # keep only data with ref = 0 (supposedly neutral face)

                self._update_ref_vector(ref_data)
        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

        # set preds_saved to true so the predictions are saved only once
        if self.save_preds:
            self.preds_saved = True

    def _fit_tuning(self, data, batch_size):
        """
        fits only tuning vector with already fitted reference vector
        :param data: [x,y] or CSVDataGen
        :param batch_size:
        :return:
        """
        print("[FIT] Learning tuning vector")
        self.t = np.zeros(self.t.shape)
        self.t_mean = np.zeros(self.t_mean.shape)
        self.t_cumul = np.zeros(self.t_cumul.shape)
        if isinstance(data, list):
            num_data = data[0].shape[0]
            indices = np.arange(num_data)
            for b in tqdm(range(0, num_data, batch_size)):
                # built batch
                end = min(b + batch_size, num_data)
                batch_idx = indices[b:end]
                batch_data = data[0][batch_idx]
                batch_label = data[1][batch_idx]

                # update direction tuning vector
                self._update_dir_tuning(batch_data, batch_label)
        elif isinstance(data, CSVDataGen):
            data.reset()
            for data_batch in tqdm(data.generate()):
                self._update_dir_tuning(data_batch[0], data_batch[1])
        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

        # set preds_saved to true so the predictions are saved only once
        if self.save_preds:
            self.preds_saved = True

    ### EVALUATION / PREDICTION

    def evaluate(self, data, batch_size=32):
        """
        This function evaluates the NormBase model on a test data set.
        :param data: input data
        :param batch_size:
        :return: accuracy, it_resp, labels
        """
        if isinstance(data, list):
            # train using data array
            accuracy, it_resp, labels = self._evaluate_array(data[0], data[1], batch_size)
        elif isinstance(data, CSVDataGen):
            # train using a generator function
            accuracy, it_resp, labels = self._evaluate_generator(data)

        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

        return accuracy, it_resp, labels

    def _evaluate_array(self, x, y, batch_size):
        num_data = np.shape(x)[0]
        indices = np.arange(num_data)

        it_resp = np.zeros((num_data, self.n_category))
        labels = np.zeros(num_data)
        classification = np.zeros(num_data)

        print("[EVALUATE] Evaluating IT responses")
        correct_pred = 0
        # evaluate data
        for b in tqdm(range(0, num_data, batch_size)):
            # built batch
            end = min(b + batch_size, num_data)
            batch_idx = indices[b:end]
            batch_data = x[batch_idx]
            batch_label = np.array(y[batch_idx]).astype(int)
            labels[batch_idx] = batch_label

            # get IT response
            it = self._get_it_resp(batch_data)
            it_resp[batch_idx] = it

            # get classification
            cat = np.argmax(it, axis=1)
            classification[batch_idx] = cat

            # count correct predictions
            correct_pred += self._get_correct_count(cat, batch_label)

        accuracy = correct_pred / num_data
        print("[EVALUATE] accuracy {:.4f}".format(accuracy))
        return accuracy, it_resp, labels

    def _evaluate_generator(self, generator):
        it_resp = []
        classification = []
        labels = []
        num_data = 0

        print("[EVALUATE] Evaluating IT responses")
        correct_pred = 0
        # evaluate data
        for data in tqdm(generator.generate()):
            num_data += len(data[0])

            # get IT response
            it = self._get_it_resp(data[0])
            it_resp.append(it)
            labels.append(data[1])

            # get classification
            it[:, self.ref_cat] = self.threshold
            cat = np.argmax(it, axis=1)
            classification.append(cat)

            # count correct predictions
            correct_pred += self._get_correct_count(cat, np.array(data[1]).astype(np.uint8))

        accuracy = correct_pred / num_data
        print("[EVALUATE] accuracy {:.4f}".format(accuracy))
        return accuracy, np.reshape(it_resp, (-1, self.n_category)), np.concatenate(labels, axis=None)

    ### PLOTTING

    def projection_tuning(self, data, batch_size = 32):
        """
        This function calculates how the data projects onto a plane.
        It returns the projection and correct labels
        keeps constant:
        - 2-norm of difference vector
        - scalar product of difference vector and tuning vector
        :param batch_size:
        :param data: input data
        :return:
        """
        if isinstance(data, CSVDataGen):
            projection, labels = self._projection_generator(data)
        elif isinstance(data, list):
            projection, labels = self._projection_array(data, batch_size=batch_size)
        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))
        return projection, labels

    def _projection_array(self, data, batch_size):
        labels = data[1]
        projection = np.zeros((self.n_category, labels.size, 2))

        num_data = labels.size
        indices = np.arange(num_data)
        for b in tqdm(range(0, num_data, batch_size)):
            # build batch
            end = min(b + batch_size, num_data)
            batch_idx = indices[b:end]
            batch_data = data[0][batch_idx]

            # calculate projection
            batch_diff = self._get_reference_pred(batch_data)
            for category in range(self.n_category):
                if category == self.ref_cat:
                    continue
                scalar_product = batch_diff @ self.t[category]
                projection[category, batch_idx, 0] = scalar_product
                projection[category, batch_idx, 1] = np.sqrt(np.square(np.linalg.norm(batch_diff, axis=1)) -
                                                   np.square(scalar_product))
                projection[category, batch_idx] = projection[category, batch_idx] / np.linalg.norm(self.t_mean[category])

        return projection, labels

    def _projection_generator(self, generator):
        projection = [] #np.zeros((generator.num_data, 2))
        labels = [] #np.zeros(generator.num_data)
        for data in tqdm(generator.generate()):
            # compute batch diff
            batch_diff = self._get_reference_pred(data[0])

        return projection, labels

    def line_constant_activation(self, dx=0.01, x_max=2.0, activations=[0.25,0.5, 0.75,1]):
        """
        calculates the lines of constant activation based on the activation function "expressivity-direction"
        recommended to be used together with projection_tuning()
        :param dx: sample distance
        :param x_max: maximum sample value
        :param activations: values of activation
        :return:
        """
        x = np.arange(dx,x_max,dx)

        lines = np.zeros((x.size, len(activations)))
        for i, activation in enumerate(activations):
            lines[:,i] = np.power(activation/np.power(x,self.nu), 2/(1-self.nu)) - np.square(x)
            lines[:,i][ lines[:,i]<0 ] = 0
            lines[:,i] = np.sqrt(lines[:,i])
        return x, lines

    def compute_dynamic_responses(self, seq_resp):
        """
        Compute the dynamic responses of recognition neurons
        Compute first a differentiation circuit followed bz a competitive network
        :param seq_resp:
        :return:
        """
        seq_length = np.shape(seq_resp)[0]

        # --------------------------------------------------------------------------------------------------------------
        # compute differentitator

        # declare differentiator
        pos_df = np.zeros((seq_length, self.n_category))
        neg_df = np.zeros((seq_length, self.n_category))
        v_df = np.zeros((seq_length, self.n_category))
        y_df = np.zeros((seq_length, self.n_category))

        for f in range(1, seq_length):
            # compute differences
            pos_dif = seq_resp[f - 1] - v_df[f - 1]
            pos_dif[pos_dif < 0] = 0
            neg_dif = v_df[f - 1] - seq_resp[f - 1]
            neg_dif[neg_dif < 0] = 0

            # update differentiator states
            pos_df[f] = ((self.tau_u - 1) * pos_df[f - 1] + pos_dif) / self.tau_u
            neg_df[f] = ((self.tau_u - 1) * neg_df[f - 1] + neg_dif) / self.tau_u
            v_df[f] = ((self.tau_v - 1) * v_df[f - 1] + seq_resp[f - 1]) / self.tau_v
            y_df[f] = ((self.tau_y - 1) * y_df[f - 1] + pos_df[f - 1] + neg_df[f - 1]) / self.tau_y

        # --------------------------------------------------------------------------------------------------------------
        # compute decision network

        # declare inhibition kernel
        inhib_k = (1 - np.eye(self.n_category) * 0.8) * self.m_inhi_w
        # declare decision neurons
        ds_neuron = np.zeros((seq_length, self.n_category))

        for f in range(1, seq_length):
            # update decision neurons
            ds_neur = ((self.tau_d - 1) * ds_neuron[f - 1] + y_df[f - 1] - inhib_k @ ds_neuron[f - 1]) / self.tau_d

            # apply activation to decision neuron
            ds_neur[ds_neur < 0] = 0
            ds_neuron[f] = ds_neur

        return ds_neuron
