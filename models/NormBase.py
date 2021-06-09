import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

from utils.extraction_model import load_v4
from utils.feature_reduction import set_feature_selection
from utils.feature_reduction import save_feature_selection
from utils.feature_reduction import load_feature_selection
from utils.feature_reduction import fit_dimensionality_reduction
from utils.feature_reduction import predict_dimensionality_reduction
from utils.CSV_data_generator import CSVDataGen


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
        # declare parameters
        self.config = config
        self.input_shape = input_shape

        try:
            self.nu = config['nu']
        except KeyError:
            self.nu = 2.0
            print("nu missing from json configuration file! Set to '2.0'")
        # set tuning function
        try:
            self.tun_func = config['tun_func']
        except KeyError:
            self.tun_func = '2-norm'
            print("tuning_function missing from json configuration file! Set to '2-norm'")
        # set dimensionality reduction
        try:
            self.dim_red = config['dim_red']
        except KeyError:
            self.dim_red = None

        self.n_category = config['n_category']
        self.ref_cat = config['ref_category']
        self.ref_cumul = 0  # cumulative count of the number of reference frame passed in the fitting

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

        # initialize n_features based on dimensionality reduction method
        set_feature_selection(self, config)
        print("[INIT] n_features:", self.n_features)

        # declare Model variables
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
            if load_NB_model:
                self.load()

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

    def _set_dynamic(self, config):
        self.is_dynamic = False
        if config.get('use_dynamic') is not None:
            if config['use_dynamic']:
                self.is_dynamic = True
                print("[INIT] Model Dynamic Set")
                # set parameters for differentiators and decision networks
                self.tau_u = config['tau_u']  # time constant for pos and negative differentiators
                self.tau_v = config['tau_v']  # time constant for IT resp differentiators
                self.tau_y = config['tau_y']  # time constant for integral differentiators
                self.tau_d = config['tau_d']  # time constant for competitive network
                self.m_inhi_w = config['m_inhi_w']  # weights of mutual inhibition
                print("[INIT] Model fit and predict will return first the dynamic decision neurons")

    ### SAVE AND LOAD ###
    def save_NB_model(self, config=None):
        print("Deprecated, please change to 'save'")
        self.save(config)

    def save(self, config=None):
        """
        This method saves the fitted model
        :param config: saves under the specified path in config
        :param save_name: subfolder name
        can be loaded with load_extraction_model(config, save_name)
        """
        # modify config if one is given
        if config is None:
            config = self.config

        # control that config folder exists
        save_folder = os.path.join("models/saved", config['config_name'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        save_folder = os.path.join(save_folder, "NormBase")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # save reference and tuning vector
        print("[SAVE] Save reference and tuning vectors")
        np.save(os.path.join(save_folder, "ref_vector"), self.r)
        np.save(os.path.join(save_folder, "tuning_vector"), self.t)
        np.save(os.path.join(save_folder, "tuning_mean"), self.t_mean)

        # save feature reduction
        save_feature_selection(self, save_folder)

        # save raw predictions
        if self.save_preds and self.preds_saved:
            print("[SAVE] Save raw prediction to csv")
            self.raw_preds_df.to_csv(os.path.join(save_folder, "raw_pred.csv"))

        print("[SAVE] Norm-Based Model saved!")
        print()

    def load(self):
        """
        loads trained model from file, including dimensionality reduction
        this method should only be called by init

        :return:
        """
        load_folder = os.path.join("models/saved", self.config['config_name'], "NormBase")

        if not os.path.exists(load_folder):
            raise ValueError("Loading path does not exists! Please control your path")

        self.r = np.load(os.path.join(load_folder, "ref_vector.npy"))
        self.t = np.load(os.path.join(load_folder, "tuning_vector.npy"))
        self.t_mean = np.load(os.path.join(load_folder, "tuning_mean.npy"))

        # load feature reduction
        load_feature_selection(self, load_folder)

        print("[LOAD] Norm Based model has been loaded from file: {}".format(self.config['config_name']))

    ### HELPER FUNCTIONS ###

    def set_ref_vector(self, r):
        self.r = r

    def set_tuning_vector(self, t):
        self.t = t

    def _update_ref_vector(self, ref):
        """
        updates reference vector wrt to new data
        :param ref:  input reference features
        :return:
        """
        # control if there's some reference features' input
        n_ref = np.shape(ref)[0]
        if n_ref > 0:
            # update ref_vector m
            self.r = (self.ref_cumul * self.r + n_ref * np.mean(ref, axis=0)) / (self.ref_cumul + n_ref)
            self.ref_cumul += n_ref

    def _get_reference_pred(self, preds):
        """
        returns difference vector between prediction and reference vector
        :param data: batch data
        :return: difference vector
        """
        len_batch = len(preds)  # take care of last epoch if size is not equal as the batch_size
        # compute batch diff
        return preds - np.repeat(np.expand_dims(self.r, axis=0), len_batch, axis=0)

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

    def _get_it_resp(self, preds):
        """
        computes the activity of norm-based neurons
        for different tuning functions selected in config
        :param data: input data
        :return: activity for each category
        """
        # compute batch diff
        batch_diff = self._get_reference_pred(preds)

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
            # this is the function published in the ICANN paper with 1-norm and nu=1
            return 0.5 * (np.linalg.norm(batch_diff, ord=1, axis=1) + (self.t @ batch_diff.T)).T
        elif self.tun_func == 'direction-only':
            # return normalized scalar product between actual direction and tuning
            v = np.linalg.norm(batch_diff, ord=2, axis=1)
            f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
            f[f < 0] = 0
            return f.T
        elif self.tun_func == 'expressivity-direction':
            # tun_func = expressivity * (direction^nu)
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
            # set response for reference category, to a sphere around reference vector with linear decay
            # activity=1 in the middle, activity=0 at half the distance to the closest category
            it_resp[:, self.ref_cat] = 1 - (0.5 *batch_diff_norm / np.delete(t_mean_norm, self.ref_cat).min())
            it_resp[:, self.ref_cat][it_resp[:, self.ref_cat] < 0] = 0
            return it_resp
        else:
            raise ValueError("{} is no valid choice for tun_func".format(self.tun_func))

    def _get_decisions_neurons(self, it_resp, seq_length, get_differentiator=False):
        decisions_neurons = []
        differentiators = []

        num_data = np.shape(it_resp)[0]
        indices = np.arange(num_data)
        for b in tqdm(range(0, num_data, seq_length)):
            # build batch
            end = min(b + seq_length, num_data)
            batch_idx = indices[b:end]
            batch_data = it_resp[batch_idx]

            # calculate decision neuron
            if get_differentiator:
                ds_neurons, diff = self.compute_dynamic_responses(batch_data, get_differentiator)
                differentiators.append(diff)
            else:
                ds_neurons = self.compute_dynamic_responses(batch_data)
            decisions_neurons.append(ds_neurons)

        if get_differentiator:
            return np.array(decisions_neurons), np.array(differentiators)
        else:
            return np.array(decisions_neurons)

    def compute_dynamic_responses(self, seq_resp, get_differentiator=False):
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
        v_df = np.zeros((seq_length, self.n_category))      # raw difference
        pos_df = np.zeros((seq_length, self.n_category))    # positive flanks
        neg_df = np.zeros((seq_length, self.n_category))    # negative flanks
        y_df = np.zeros((seq_length, self.n_category))      # integrator

        for f in range(1, seq_length):
            # compute differences
            pos_dif = seq_resp[f - 1] - v_df[f - 1]
            pos_dif[pos_dif < 0] = 0
            neg_dif = v_df[f - 1] - seq_resp[f - 1]
            neg_dif[neg_dif < 0] = 0

            # update differentiator states
            v_df[f] = ((self.tau_v - 1) * v_df[f - 1] + seq_resp[f - 1]) / self.tau_v
            pos_df[f] = ((self.tau_u - 1) * pos_df[f - 1] + pos_dif) / self.tau_u
            neg_df[f] = ((self.tau_u - 1) * neg_df[f - 1] + neg_dif) / self.tau_u
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

        if get_differentiator:
            return ds_neuron, np.array([pos_df, neg_df])
        else:
            return ds_neuron

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # ## FIT FUNCTIONS ###
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    def fit(self, data, batch_size=32, fit_dim_red=True, fit_ref=True, fit_tun=True, get_it_resp=False, get_differentiator=False):
        """
        fit model on data

        :param data: input
        :param batch_size:
        :param fit_dim_red: whether to fit dimensionality reduction
        :param fit_ref: whether to fit reference vector
        :param fit_tun: whether to fit tuning vector
        :param get_it_resp: allows to get the NormBase it neurons if model is dynamic, otherwise it is the normal output
        :param get_differentiator: allows to get the intermediate differentiators output
        :return:
        """

        # predict v4 responses
        print("[FIT] Compute v4")

        flatten = True
        if self.dim_red == 'semantic' or self.dim_red == "semantic-pattern" or self.dim_red == "pattern":
            flatten = False

        v4_preds = self.predict_v4(data[0], flatten=flatten)
        print("[FIT] Shape v4_preds", np.shape(v4_preds))

        # set preds_saved to true so the predictions are saved only once
        if self.save_preds and self.dim_red is not None:
            self.preds_saved = True

        if fit_dim_red:
            print("[FIT] - Fitting dimensionality reduction -")
            v4_preds_red = fit_dimensionality_reduction(self, v4_preds)
        else:
            v4_preds_red = predict_dimensionality_reduction(self, v4_preds)
        print("[FIT] Data reduced")

        if fit_ref:
            print("[FIT] - Fitting Reference Pattern -")
            self._fit_reference([v4_preds_red, data[1]], batch_size)
            print("[FIT] Reference pattern learned")

        if fit_tun:
            print("[FIT] - Fitting Tuning Vector -")
            self._fit_tuning([v4_preds_red, data[1]], batch_size)
            print("[FIT] Tuning pattern learned")

        # compute projections
        print("[FIT] compute IT responses")
        it_resp = self._get_it_resp(v4_preds_red)

        # compute differentiator if dynamic is set
        if self.is_dynamic:
            print("[FIT] compute dynamic")
            if get_differentiator:
                ds_neurons, differentiators = self._get_decisions_neurons(it_resp,
                                                                          self.config['seq_length'],
                                                                          get_differentiator=get_differentiator)
            else:
                ds_neurons = self._get_decisions_neurons(it_resp, self.config['seq_length'])

        print()
        if self.is_dynamic:
            if get_it_resp and get_differentiator:
                return ds_neurons, it_resp, differentiators
            elif get_it_resp:
                return ds_neurons, it_resp
            elif get_differentiator:
                return ds_neurons, differentiators
            else:
                return ds_neurons
        else:
            return it_resp

    def _fit_reference(self, data, batch_size):
        """
        fits only reference vector without fitting tuning vector
        :param data: [x,y] or DataGen
        :param batch_size:
        :return:
        """
        print("[FIT] Learning reference pattern")
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
            raise ValueError("Type {} of data is not recognize!".format(type(data)))

        return self.r

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
        print("shape self.t", np.shape(self.t))
        print("shape self.t_mean", np.shape(self.t_mean))
        print("shape self.t_cumul", np.shape(self.t_cumul))

        if isinstance(data, list) or type(data).__module__ == np.__name__:
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
            raise ValueError("Type {} of data is not recognize!".format(type(data)))

    ### PREDICTION / EVALUATION

    def evaluate_v4(self, data, flatten=True):
        # deprecated, but keep it
        print("[WARNING] This function is deprecated, please change it to 'predict_v4'")
        return self.predict_v4(data, flatten)

    def predict_v4(self, data, flatten=True):
        """
        returns prediction of cnn including preprocessing of images
        in NormBase, must be used only to train dimensionality reduction and in predict()
        :param data: batch of data
        :param flatten: if True, is flattened
        :return: prediction
        """

        # prediction of v4 layer
        preds = self.v4.predict(data, verbose=1)
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

    def predict(self, data, get_it_resp=False, get_differentiator=False):
        """
        predict expression neurons of Norm base Mechanism
        pipeline consists of preprocessing, cnn, and dimensionality reduction
        :param data: batch of data
        :return: prediction
        """
        print("[PREDICT] Compute v4")
        flatten = True
        if self.dim_red == 'semantic' or self.dim_red == "semantic-pattern" or self.dim_red == "pattern":
            flatten = False

        v4_preds = self.predict_v4(data[0], flatten=flatten)

        print("[PREDICT] - reduce data dimensionality -")
        v4_preds_red = predict_dimensionality_reduction(self, v4_preds)
        print("shape v4_preds_red", np.shape(v4_preds_red))

        print("[PREDICT] compute IT responses")
        it_resp = self._get_it_resp(v4_preds_red)

        if self.is_dynamic:
            print("[PREDICT] compute dynamic")
            if get_differentiator:
                ds_neurons, differentiators = self._get_decisions_neurons(it_resp,
                                                                          self.config['seq_length'],
                                                                          get_differentiator=get_differentiator)
            else:
                ds_neurons = self._get_decisions_neurons(it_resp, self.config['seq_length'])

        if self.is_dynamic:
            if get_it_resp and get_differentiator:
                return ds_neurons, it_resp, differentiators
            elif get_it_resp:
                return ds_neurons, it_resp
            elif get_differentiator:
                return ds_neurons, differentiators
            else:
                return ds_neurons
        else:
            return it_resp

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

    def _get_correct_count(self, x, label):
        one_hot_encoder = np.eye(self.n_category)
        one_hot_cats = one_hot_encoder[x]
        one_hot_label = one_hot_encoder[label]

        return np.count_nonzero(np.multiply(one_hot_cats, one_hot_label))

    ### PLOTTING

    def projection_tuning(self, data, batch_size=32):
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

    def plot_it_neurons(self, it_neurons, title=None, save_folder=None, normalize=False):
        plt.figure()

        if normalize:
            norm = np.amax(it_neurons)
        for i in range(self.config['n_category']):
            if normalize:
                plt.plot(it_neurons[:, i] / norm)
            else:
                plt.plot(it_neurons[:, i])

        # set figure title
        fig_title = 'IT_responses.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)

    def plot_it_neurons_per_sequence(self, it_neurons, title=None, save_folder=None, normalize=False):
        # compute the number of sequence depending on the number of frames and seuence length
        seq_length = self.config['seq_length']
        n_sequence = np.shape(it_neurons)[0] // seq_length

        plt.figure()
        for s in range(n_sequence):
            # create subplot for each sequence
            plt.subplot(n_sequence, 1, s + 1)
            start = s * seq_length
            stop = start + seq_length

            # normalize activity
            if normalize:
                norm = np.amax(it_neurons[start:stop])

            # plot for each category
            for i in range(self.config['n_category']):
                # select color
                color = self.config['colors'][i]

                if normalize:
                    plt.plot(it_neurons[start:stop, i] / norm, color=color)
                else:
                    plt.plot(it_neurons[start:stop, i], color=color)

        # set figure title
        fig_title = 'IT_responses.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)

    def plot_differentiators(self, differentiators, title=None, save_folder=None, normalize=False):
        # get the number of sequence and number of differentiators
        n_sequence = np.shape(differentiators)[0]
        n_differentiators = np.shape(differentiators)[1]

        plt.figure()
        for s in range(n_sequence):
            # create subplot for each sequence
            plt.subplot(n_sequence, 1, s + 1)

            # normalize activity
            if normalize:
                norm = np.amax(differentiators[s])

            # plot for each category
            for i in range(self.config['n_category']):
                #  change color depending on the category
                color = self.config['colors'][i]
                for d in range(n_differentiators):

                    # changed linestyle depending on the differentiator
                    if d % 2 == 0:
                        linestyle = 'solid'
                    else:
                        linestyle = 'dashed'

                    # plot normalized differentiator
                    if normalize:
                        plt.plot(differentiators[s, d, :, i] / norm, color=color, linestyle=linestyle)
                    else:
                        plt.plot(differentiators[s, d, :, i], color=color, linestyle=linestyle)

        # set figure title
        fig_title = 'differentiators_responses.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)

    def plot_decision_neurons(self, ds_neurons, title=None, save_folder=None, normalize=False):
        # get the number of sequence
        n_sequence = np.shape(ds_neurons)[0]

        plt.figure()
        for s in range(n_sequence):
            # create subplot for each sequence
            plt.subplot(n_sequence, 1, s + 1)

            # normalize activity
            if normalize:
                norm = np.amax(ds_neurons[s])

            # plot for each category
            for i in range(self.config['n_category']):
                color = self.config['colors'][i]

                if normalize:
                    plt.plot(ds_neurons[s, :, i] / norm, color=color)
                else:
                    plt.plot(ds_neurons[s, :, i], color=color)

        # set figure title
        fig_title = 'decision_neurons_responses.png'
        if title is not None:
            fig_title = title + '_' + fig_title

        if save_folder is not None:
            plt.savefig(os.path.join(save_folder, fig_title))
        else:
            plt.savefig(fig_title)