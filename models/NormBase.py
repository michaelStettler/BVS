import types
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.load_model import load_model
from utils.data_generator import DataGen


class NormBase:
    """
    NormBase class define the functions to train a norm base mechanism with a front end extracting features

    r:= reference vector (n_features, )
    t:= tuning vector (n_category, n_features)

    with
    n_features:= height * width * channels of the input_shape

    """

    def __init__(self, config, input_shape):
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
        try:
            self.nu = config['nu']
        except KeyError:
            self.nu = 2.0
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
        threshold_divided = 50
        self.threshold = self.n_features / threshold_divided
        print("[INIT] n_features:", self.n_features)
        print("[INIT] threshold ({:.1f}%):".format(100/threshold_divided), self.threshold)
        print()

    def _load_model(self, config, input_shape):
        if (config['model'] == 'VGG19') | (config['model'] =='ResNet50V2'):
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

    def _get_reference_pred(self, data):
        len_batch = len(data)  # take care of last epoch if size is not equal as the batch_size

        # predict images
        preds = self.v4.predict(data)
        preds = np.reshape(preds, (len_batch, -1))

        # compute batch diff
        return preds - np.repeat(np.expand_dims(self.r, axis=1), len_batch, axis=1).T

    def _update_dir_tuning(self, data, label):
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
        # compute batch diff
        batch_diff = self._get_reference_pred(data)

        # compute norm-reference neurons
        v = np.sqrt(np.diag(batch_diff @ batch_diff.T))
        f = self.t @ batch_diff.T @ np.diag(np.power(v, -1))
        f[f < 0] = 0
        f = np.power(f, self.nu)
        return np.diag(v) @ f.T

    def get_correct_count(self, x, label):
        one_hot_encoder = np.eye(self.n_category)
        one_hot_cats = one_hot_encoder[x]
        one_hot_label = one_hot_encoder[label]

        return np.count_nonzero(np.multiply(one_hot_cats, one_hot_label))

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
        elif isinstance(data, DataGen):
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
        for data in tqdm(generator.generate()):
            x = data[0]
            y = data[1]
            ref_data = x[y == self.ref_cat]  # keep only data with ref = 0 (supposedly neutral face)

            self._update_ref_vector(ref_data)

        # reset count of the generator for second pass
        generator.reset()

        # learn tuning direction
        print("[FIT] Learning tuning direction")
        for data in tqdm(generator.generate()):
            self._update_dir_tuning(data[0], data[1])

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

            # get it response
            it = self._get_it_resp(batch_data)
            it_resp[batch_idx] = it

        return it_resp

    def evaluate(self, data, batch_size=32):

        if isinstance(data, list):
            # train using data array
            it_resp = self._evaluate_array(data[0], data[1], batch_size)
        elif isinstance(data, DataGen):
            # train using a generator function
            it_resp = self._evaluate_generator(data)

        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

        return it_resp

    def _evaluate_array(self, x, y, batch_size):
        num_data = np.shape(x)[0]
        indices = np.arange(num_data)

        it_resp = np.zeros((num_data, self.n_category))
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

            # get IT response
            it = self._get_it_resp(batch_data)
            it_resp[batch_idx] = it

            # get classification
            cat = np.argmax(it, axis=1)
            classification[batch_idx] = cat

            # count correct predictions
            correct_pred += self.get_correct_count(cat, batch_label)

        accuracy = correct_pred / num_data
        print("[EVALUATE] accuracy {:.4f}".format(accuracy))
        return it_resp

    def _evaluate_generator(self, generator):
        it_resp = []
        classification = []
        num_data = 0

        print("[EVALUATE] Evaluating IT responses")
        correct_pred = 0
        # evaluate data
        for data in tqdm(generator.generate()):
            num_data += len(data[0])

            # get IT response
            it = self._get_it_resp(data[0])
            it_resp.append(it)

            # get classification
            it[:, self.ref_cat] = self.threshold
            cat = np.argmax(it, axis=1)
            classification.append(cat)

            # count correct predictions
            correct_pred += self.get_correct_count(cat, np.array(data[1]).astype(np.uint8))

        accuracy = correct_pred / num_data
        print("[EVALUATE] accuracy {:.4f}".format(accuracy))
        return np.reshape(it_resp, (-1, self.n_category))

    '''
    This function evaluates the NormBase model on a test data set.
    This function is similar to evaluate but returns additional results.
    returns 
    accuracy: fraction correct
    it_resp: response activity
    labels: correct labels
    '''
    def evaluate_accuracy(self, data, batch_size=32):
        if isinstance(data, list):
            # train using data array
            accuracy, it_resp, labels = self._evaluate_accuracy_array(data[0], data[1], batch_size)
        elif isinstance(data, DataGen):
            # train using a generator function
            accuracy, it_resp, labels = self._evaluate_accuracy_generator(data)

        else:
            raise ValueError("Type {} od data is not recognize!".format(type(data)))

        return accuracy, it_resp, labels

    def _evaluate_accuracy_array(self, x, y, batch_size):
        # implement and return similar to function above
        return None, None, None

    def _evaluate_accuracy_generator(self, generator):
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
            correct_pred += self.get_correct_count(cat, np.array(data[1]).astype(np.uint8))

        accuracy = correct_pred / num_data
        print("[EVALUATE] accuracy {:.4f}".format(accuracy))
        return accuracy, np.reshape(it_resp, (-1, self.n_category)), np.concatenate(labels, axis=None)