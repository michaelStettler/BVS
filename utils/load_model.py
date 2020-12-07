import tensorflow as tf


def load_model(config, input_shape=None):
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
    if config['model'] == "VGG19":
        if config["include_top"]:
            model = tf.keras.applications.VGG19(include_top=True, weights=config['weights'])
        elif not config["include_top"] and input_shape is not None:
            model = tf.keras.applications.VGG19(include_top=False, weights=config['weights'], input_shape=input_shape)
        else:
            raise ValueError("Include top is false but input_shape is not given!")
    elif config['model'] == "ResNet50V2":
        if config["include_top"]:
            model = tf.keras.applications.ResNet50V2(include_top=True, weights=config['weights'])
        elif not config["include_top"] and input_shape is not None:
            if config['weights'] == 'imagenet':
                model = tf.keras.applications.ResNet50V2(include_top=False, weights=config['weights'], input_shape=input_shape)
            else:
                model = tf.keras.models.load_model(config['weights'])
        else:
            raise ValueError("Include top is false but input_shape is not given!")
        # fix names in layers, unnamed layers get system names which causes layers upstream
        model.layers[37]._name = "max_pooling2d"
        model.layers[83]._name = "max_pooling2d_1"
        model.layers[150]._name = "max_pooling2d_2"
    else:
        raise ValueError("Model {} not found!".format(config['model']))

    return model
