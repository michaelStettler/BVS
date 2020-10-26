import numpy as np
import tensorflow as tf
from tqdm import tqdm


class NormBase:

    def __init__(self, config, input_shape):
        self.ref_idx = int(config['ref_idx'])
        self.ref_cumul = 0  # cumulative count of the number of reference frame passed in the fitting

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

        # load model
        if config['model'] == 'VGG16':
            self.model = tf.keras.applications.VGG16(include_top=False,
                                                     weights=config["weights"],
                                                     input_shape=input_shape)
            self.v4 = tf.keras.Model(inputs=self.model.input,
                                     outputs=self.model.get_layer(config['v4_layer']).output)
        else:
            raise ValueError("model: {} does not exists! Please change config file or add the model"
                             .format(config['model']))

        print("[Model] -- Model loaded --")
        print("[Model] V4 layer:", config['v4_layer'])
        print(self.v4.summary())

        shape_v4 = np.shape(self.v4.layers[-1].output)
        self.n_features = shape_v4[1] * shape_v4[2] * shape_v4[3]
        self.m = np.zeros(self.n_features)

    def fit(self, x, y, batch_size=32):
        num_data = np.shape(x)[0]
        indices = np.arange(num_data)
        print("[fit] num indices", len(indices))

        # todo shuffle
        # if shuffle
        # np.random.shuffle(indices)

        # learn reference vector
        for b in tqdm(range(0, num_data, batch_size)):
            # built batch
            end = min(b + batch_size, num_data)
            batch_idx = indices[b:end]

            batch_data = x[batch_idx]
            ref_data = batch_data[y[batch_idx] == self.ref_idx]  # keep only data with ref = 0 (supposedly neutral face)
            n_ref = np.shape(ref_data)[0]

            if n_ref > 0:
                # predict images
                preds = self.v4.predict(ref_data)
                preds = np.reshape(preds, (n_ref, -1))

                # update ref_vector m
                self.m = (self.ref_cumul * self.m + n_ref * np.mean(preds, axis=0)) / (self.ref_cumul + n_ref)
                self.ref_cumul += n_ref


        # learn tuning direction