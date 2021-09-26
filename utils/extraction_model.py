import tensorflow as tf
import warnings


def load_extraction_model(config, input_shape=None, verbose=False):
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
    if config['extraction_model'] == "VGG19":
        if config["include_top"]:
            model = tf.keras.applications.VGG19(include_top=True, weights=config['weights'])
        elif not config["include_top"] and input_shape is not None:
            model = tf.keras.applications.VGG19(include_top=False, weights=config['weights'], input_shape=input_shape)
        else:
            raise ValueError("Include top is false but input_shape is not given!")
    elif config['extraction_model'] == "ResNet50V2":
        if config["include_top"]:
            model = tf.keras.applications.ResNet50V2(include_top=True, weights=config['weights'])
        elif not config["include_top"] and input_shape is not None:
            if config['weights'] == 'imagenet':
                model = tf.keras.applications.ResNet50V2(include_top=False, weights=config['weights'], input_shape=input_shape)
                # fix names in layers, unnamed layers get system names which causes layers upstream
                model.layers[37]._name = "max_pooling2d"
                model.layers[83]._name = "max_pooling2d_1"
                model.layers[150]._name = "max_pooling2d_2"
            else:
                model = tf.keras.models.load_model(config['weights'])
        else:
            raise ValueError("Include top is false but input_shape is not given!")
    else:
        raise ValueError("Model {} not found!".format(config['model']))

    if verbose:
        print(model.summary())

    return model


def load_v4(model, config, input_shape):
    """
    load the v4 pipeline extraction feature
    When using a CNN, it is essentially cutting of the CNN to a selected layer

    :param config:
    :param input_shape:
    :return:
    """
    if (config['extraction_model'] == 'VGG19') | (config['extraction_model'] == 'ResNet50V2'):
        model.model = load_extraction_model(config, input_shape)
        model.v4 = tf.keras.Model(inputs=model.model.input,
                                 outputs=model.model.get_layer(config['v4_layer']).output)
    elif config['extraction_model'] == 'FLAME':
        print("TODO TAKE CARE OF NO EXTRACTION")
    else:
        raise ValueError("model: {} does not exists! Please change config file or add the model"
                         .format(config['extraction_model']))
    # define preprocessing for images
    if config['extraction_model'] == 'VGG19':
        model.preprocessing = 'VGG19'
    else:
        model.preprocessing = None
        warnings.warn(f'no preprocessing for images defined for config["model"]={config["extraction_model"]}')