import types
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.load_model import load_model


class NormBase:
    """
    NormBase class define the functions to train a norm base mechanism with a front end extracting features

    r:= reference vector (n_features, )
    t:= tuning vector (n_category, n_features)

    with
    n_features:= height * width * channels of the input_shape

    """

    def __init__(self, config, input_shape, nu=2):
        """
        The init function is responsible to declare the front end model of the norm base mechanism declared in the
        config file

        :param config: JSON file
        :param input_shape: Tuple with the input size of the model (height, width, channels)
        :param nu: tuning width
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
        self.nu = nu
        self.n_category = config['n_category']
        self.ref_cat = config['ref_category']
        self.ref_cumul = 0  # cumulative count of the number of reference frame passed in the fitting

        # load front end feature extraction model
        self.v4 = self._load_model(config, input_shape)
        print("[INIT] -- Model loaded --")
        print("[INIT] Model:", config['model'])
        print("[INIT] V4 layer:", config['v4_layer'])

        shape_v4 = np.shape(self.v4.layers[-1].output)
        self.n_features = shape_v4[1] * shape_v4[2] * shape_v4[3]
        self.r = np.zeros(self.n_features)
        self.t = np.zeros((self.n_category, self.n_features))
        self.t_cumul = np.zeros(self.n_category)
        self.t_mean = np.zeros((self.n_category, self.n_features))
        print("[INIT] n_features:", self.n_features)
        print()

    def _load_model(self, config, input_shape):
        if config['model'] == 'VGG19':
            model = load_model(config, input_shape)
            v4 = tf.keras.Model(inputs=model.input,
                                     outputs=model.get_layer(config['v4_layer']).output)
        else:
            raise ValueError("model: {} does not exists! Please change config file or add the model"
                             .format(config['model']))
        return v4

    def print_v4_summary(self):
        print(self.v4.summary())

    def set_ref_vector(self, r):
        self.r = r

    def set_tuning_vector(self, t):
        self.t = t

    def _update_ref_vector(self, data):
        n_ref = np.shape(data)[0]

        if n_ref > 0:
            # predict images
            preds = self.v4.predict(data)
            preds = np.reshape(preds, (n_ref, -1))

            # update ref_vector m
            self.r = (self.ref_cumul * self.r + n_ref * np.mean(preds, axis=0)) / (self.ref_cumul + n_ref)
            self.ref_cumul += n_ref

    def _update_dir_tuning(self, data, label):
        len_batch = len(data)  # take care of last epoch if size is not equal as the batch_size

        # predict images
        preds = self.v4.predict(data)
        preds = np.reshape(preds, (len_batch, -1))

        # compute batch diff
        batch_diff = preds - np.repeat(np.expand_dims(self.r, axis=1), len_batch, axis=1).T

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

    def fit(self, data, batch_size=32, shuffle=False):
        """
        The fit function allows to learn both the reference vector (m) and the tuning vector (n).
        The training is complete in two different loops, one loop for the reference vector and one for the tuning
        vector.

        The loops allows to compute the reference and tuning vector in a cumulative way over the complete dataset
        by leveraging the creation of batch from the data

        n_features:= height * width * channels

        :param x: training data (n_samples, height, width, channels)
        :param y: label (n_samples, )
        :param batch_size:
        :param shuffle:
        :return: r, t
        """

        if isinstance(data, list):
            # train using data array
            self._fit_array(data[0], data[1], batch_size, shuffle)
        elif isinstance(data, types.GeneratorType):
            # train using a generator function
            self._fit_generator(data)

        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

        return self.r, self.t

    def _fit_array(self, x, y, batch_size, shuffle):
        num_data = np.shape(x)[0]
        indices = np.arange(num_data)

        if shuffle:
            np.random.shuffle(indices)

        # learn reference vector
        print("[FIT] Learning reference pose")
        for b in tqdm(range(0, num_data, batch_size)):
            # built batch
            end = min(b + batch_size, num_data)
            batch_idx = indices[b:end]
            batch_data = x[batch_idx]
            ref_data = batch_data[y[batch_idx] == self.ref_cat]  # keep only data with ref = 0 (supposedly neutral face)

            # update reference vector
            self._update_ref_vector(ref_data)

        # learn tuning direction
        print("[FIT] Learning tuning direction")
        for b in tqdm(range(0, num_data, batch_size)):
            # built batch
            end = min(b + batch_size, num_data)
            batch_idx = indices[b:end]
            batch_data = x[batch_idx]
            batch_label = y[batch_idx]

            # update direction tuning vector
            self._update_dir_tuning(batch_data, batch_label)

    def _fit_generator(self, generator):
        # learn reference vector
        print("[FIT] Learning reference pose")
        for data in tqdm(generator):
            x = data[0]
            y = data[1]

            ref_data = x[y == self.ref_cat]  # keep only data with ref = 0 (supposedly neutral face)

            self._update_ref_vector(ref_data)

        # learn tuning direction
        print("[FIT] Learning tuning direction")
        for data in tqdm(generator):
            x = data[0]
            y = data[1]

            self._update_dir_tuning(x, y)

    def predict(self, data, batch_size=32):
        x = data[0]
        num_data = np.shape(x)[0]
        indices = np.arange(num_data)

        it_resp = np.zeros((num_data, self.n_category))

        # predict data
        for b in tqdm(range(0, num_data, batch_size)):
            # built batch
            end = min(b + batch_size, num_data)
            batch_idx = indices[b:end]
            batch_data = x[batch_idx]
            len_batch = len(batch_idx)  # take care of last epoch if size is not equal as the batch_size

            # predict images
            preds = self.v4.predict(batch_data)
            preds = np.reshape(preds, (len_batch, -1))

            # compute batch diff
            batch_diff = preds - np.repeat(np.expand_dims(self.r, axis=1), len_batch, axis=1).T

            # compute norm-reference neurons
            v = np.sqrt(np.diag(batch_diff @ batch_diff.T))
            f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
            f[f < 0] = 0
            f = np.power(f, self.nu)
            it_resp[batch_idx] = np.diag(v)@ f.T

        return it_resp

    def evaluate(self, data, batch_size=32):
        x = data[0]
        y = data[1]

        num_data = np.shape(x)[0]
        indices = np.arange(num_data)

        it_resp = np.zeros((num_data, self.n_category))
        classification = np.zeros(num_data)

        correct_pred = 0
        # predict data
        for b in tqdm(range(0, num_data, batch_size)):
            # built batch
            end = min(b + batch_size, num_data)
            batch_idx = indices[b:end]
            batch_data = x[batch_idx]
            batch_label = np.array(y[batch_idx]).astype(int)
            len_batch = len(batch_idx)

            # predict images
            preds = self.v4.predict(batch_data)
            preds = np.reshape(preds, (len_batch, -1))

            # compute batch diff
            batch_diff = preds - np.repeat(np.expand_dims(self.r, axis=1), len_batch, axis=1).T

            # compute norm-reference neurons
            v = np.sqrt(np.diag(batch_diff @ batch_diff.T))
            f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
            f[f < 0] = 0
            f = np.power(f, self.nu)
            it = np.diag(v)@ f.T
            it_resp[batch_idx] = it

            # get classification
            cat = np.argmax(it, axis=1)
            classification[batch_idx] = np.argmax(it, axis=1)

            # transform into one hot
            _one_hot_encoder = np.eye(self.n_category)
            one_hot_cats = _one_hot_encoder[cat]
            one_hot_label = _one_hot_encoder[batch_label]
            correct_pred += np.count_nonzero(np.multiply(one_hot_cats, one_hot_label))

        accuracy = correct_pred / num_data
        print("[EVALUATE] accuracy {:.4f}".format(accuracy))
        return it_resp

