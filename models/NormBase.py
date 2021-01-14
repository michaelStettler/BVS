import types
import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.decomposition import PCA

from utils.load_model import load_model
from utils.data_generator import DataGen


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

    def __init__(self, config, input_shape, save_name=None):
        """
        The init function is responsible to declare the front end model of the norm base mechanism declared in the
        config file

        :param config: JSON file
        :param input_shape: Tuple with the input size of the model (height, width, channels)
        :param save_name: if given the fitted model is loaded from config['save_name'] subfolder save_name
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
        # -----------------------------------------------------------------

        # declare parameters
        try:
            self.nu = config['nu']
        except KeyError:
            self.nu = 2.0
        try:
            self.tun_func = config['tun_func']
        except KeyError:
            self.tun_func = '2-norm'
        try:
            #set dim_red if available in config (only option 'PCA')
            self.dim_red = config['dim_red']
            self.pca = PCA(n_components=config['PCA'])
        except KeyError:
            self.dim_red = None
        self.n_category = config['n_category']
        self.ref_cat = config['ref_category']
        self.ref_cumul = 0  # cumulative count of the number of reference frame passed in the fitting

        # load front end feature extraction model
        self._load_v4(config, input_shape)
        print("[INIT] -- Model loaded --")
        print("[INIT] Model:", config['model'])
        print("[INIT] V4 layer:", config['v4_layer'])
        if not (self.dim_red is None):
            print("[INIT] dim_red:", self.dim_red)
        shape_v4 = np.shape(self.v4.layers[-1].output)
        print("shape_v4", shape_v4)
        try:
            # initialize n_features as number of components of PCA
            self.n_features = config['PCA']
        except KeyError:
            # initialize as output of network as default
            shape_v4 = np.shape(self.v4.layers[-1].output)
            if len(shape_v4) == 2:  # use flatten... but self.v4.layers[-1].output is a tensorShape object
                self.n_features = shape_v4[1]
            elif len(shape_v4) == 3:
                self.n_features = shape_v4[1] * shape_v4[2]
            elif len(shape_v4) == 4:
                self.n_features = shape_v4[1] * shape_v4[2] * shape_v4[3]
            else:
                raise NotImplementedError("Dimensionality not implemented")

        self.r = np.zeros(self.n_features)
        self.t = np.zeros((self.n_category, self.n_features))
        self.t_cumul = np.zeros(self.n_category)
        self.t_mean = np.zeros((self.n_category, self.n_features))
        threshold_divided = 50
        self.threshold = self.n_features / threshold_divided
        print("[INIT] n_features:", self.n_features)
        print("[INIT] threshold ({:.1f}%):".format(100/threshold_divided), self.threshold)

        # load model
        if not save_name is None:
            self._load_model(config, save_name)
            print("[INIT] saved model is loaded from file")

        print()

    def _load_v4(self, config, input_shape):
        if (config['model'] == 'VGG19') | (config['model'] =='ResNet50V2'):
            model = load_model(config, input_shape)
            self.v4 = tf.keras.Model(inputs=model.input,
                                     outputs=model.get_layer(config['v4_layer']).output)
        else:
            raise ValueError("model: {} does not exists! Please change config file or add the model"
                             .format(config['model']))

    ### SAVE AND LOAD ###

    def save_model(self, config, save_name):
        """
        This method saves the fitted model
        :param config: saves under the specified path in config
        :param save_name: subfolder name
        can be loaded with load_model(config, save_name)
        """
        if not os.path.exists(os.path.join("models/saved", config['save_name'])):
            os.mkdir(os.path.join("models/saved", config['save_name']))
        save_folder = os.path.join("models/saved", config['save_name'], save_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_folder = os.path.join(save_folder, "NormBase_saved")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        #save reference and tuning vector
        np.save(os.path.join(save_folder, "ref_vector"), self.r)
        np.save(os.path.join(save_folder, "tuning_vector"), self.t)
        np.save(os.path.join(save_folder, "tuning_mean"), self.t_mean)
        #save PCA
        if self.dim_red == 'PCA':
            #np.save(os.path.join(save_folder, "pca"), self.pca)
            pickle.dump(self.pca, open(os.path.join(save_folder, "pca.pkl"), 'wb'))

    def _load_model(self, config, save_name):
        """
        loads trained model from file, including dimensionality reduction
        this method should only be called by init
        :param config: configuration file
        :param save_name: folder name
        :return:
        """
        load_folder = os.path.join("models/saved", config['save_name'], save_name, "NormBase_saved")
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

    def _get_preds(self,data):
        """
        returns prediction of pipeline before norm-base
        :param data: batch of data
        :return: prediction
        """
        # prediction of v4 layer
        preds = self.v4.predict(data)
        preds = np.reshape(preds, (data.shape[0], -1))
        if self.dim_red == 'PCA':
            # projection by PCA
            preds = self.pca.transform(preds)
        return preds

    def _update_ref_vector(self, data):
        """
        updates reference vector wrt to new data
        :param data: input of self.ref_cat
        :return:
        """
        n_ref = np.shape(data)[0]
        if n_ref > 0:
            preds = self._get_preds(data)
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
        preds = self._get_preds(data)
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

    ### FIT FUNCTIONS ###

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
        if self.dim_red == 'PCA':
            print("[FIT] Fitting PCA")
            # get output on all data from self.v4
            if isinstance(data, DataGen):
                # data = data.getAllData()
                raise ValueError("PCA and DataGenerator has to be implemented first to be usable")
            v4_predict = self.v4.predict(data[0])
            v4_predict = np.reshape(v4_predict, (data[0].shape[0], -1))
            print("[FIT] v4 prediction generated", v4_predict.shape)
            # perform PCA on this output
            self.pca.fit(v4_predict)
            print("explained variance", self.pca.explained_variance_ratio_)

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
        elif isinstance(data, DataGen):
            # train using a generator
            data.reset()
            for data_batch in tqdm(data.generate()):
                x = data_batch[0]
                y = data_batch[1]
                ref_data = x[y == self.ref_cat]  # keep only data with ref = 0 (supposedly neutral face)

                self._update_ref_vector(ref_data)
        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

    def _fit_tuning(self, data, batch_size):
        """
        fits only tuning vector with already fitted reference vector
        :param data: [x,y] or DataGen
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
        elif isinstance(data, DataGen):
            data.reset()
            for data_batch in tqdm(data.generate()):
                self._update_dir_tuning(data_batch[0], data_batch[1])
        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

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
        elif isinstance(data, DataGen):
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
        if isinstance(data, DataGen):
            projection, labels = self._projection_generator(data)
        if isinstance(data, list):
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
