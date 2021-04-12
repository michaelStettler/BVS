import os
import tensorflow as tf


class ExampleBase:

    def __init__(self, config, input_shape, load_EB_model=None):
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

        # declare variable
        self.config = config
        self.input_shape = input_shape

        # load norm base model
        if load_EB_model is not None:
            self.load()
            print("[INIT] Example Based model has been loaded from file: {}".format(config['config_name']))

    def fit(self, data):
        print("fit")

    def save(self, config=None):
        if config is None:
            config = self.config
        print("save")
        # create folder if it does not exist
        save_folder = os.path.join("models/saved", config['config_name'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        save_folder = os.path.join(save_folder, "ExampleBase")
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    def load(self):
        print("load")
