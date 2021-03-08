import os
import json


def load_config(config_name, path=None):
    """
    small helper function to load the json config
    :param config_name:
    :param path:
    :return:
    """

    # use given path if given
    if path is not None:
        config_file_path = os.path.join(path, config_name)

    # load config
    with open(config_file_path) as json_file:
        config = json.load(json_file)
    return config
