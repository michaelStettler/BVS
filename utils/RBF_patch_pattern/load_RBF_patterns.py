import os
import numpy as np


def load_files(config, condition, file_name, avatar_name=None):
    files = []
    for lmk_name in config["{}_lmk_name".format(condition)]:
        if avatar_name is not None:
            pattern_name = "{}_{}_{}_{}.npy".format(condition, avatar_name, file_name, lmk_name)
        else:
            pattern_name = "{}_{}_{}.npy".format(condition, file_name, lmk_name)
        pattern_path = os.path.join(config["directory"], config["LMK_data_directory"], pattern_name)
        files.append(np.load(pattern_path))

    return files


def load_LMK_patterns_and_sigma(config, avatar_name=None, load_FR=True, load_FER=True):
    """
    return all the patterns and sigma saved for the LMK detector

    :param config:
    :param avatar_name:
    :param load_FR:
    :param load_FER:
    :return:
    """

    if avatar_name is None and load_FR:
        raise ReferenceError("Please set the avatar name for the FR pipeline")

    if avatar_name is not None:
        load_FR = True

    if load_FR:
        FR_patterns_list = []
        FR_sigma_list = []

        for avatar_name in avatar_name:
            patterns = load_files(config, "FR", "patterns", avatar_name=avatar_name)
            sigma = load_files(config, "FR", "sigma", avatar_name=avatar_name)

            FR_patterns_list.append(patterns)
            FR_sigma_list.append(sigma)

    if load_FER:
        FER_patterns_list = load_files(config, "FER", "patterns")
        FER_sigma_list = load_files(config, "FER", "sigma")

    if load_FR and load_FER:
        return FR_patterns_list, FR_sigma_list, FER_patterns_list, FER_sigma_list
    elif load_FR:
        return FR_patterns_list, FR_sigma_list
    elif load_FER:
        return FER_patterns_list, FER_sigma_list
    else:
        raise NotImplemented("No loading!")