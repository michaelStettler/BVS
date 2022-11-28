import os
import numpy as np

from utils.load_config import load_config

"""
run: python -m datasets_utils.merge_LMK_pos
"""


def merge_LMK_pos(config, from_all_lmk=True):
    FER_pos = None

    if from_all_lmk:
        lmk_names = ["left_eyebrow_ext", "left_eyebrow_int", "right_eyebrow_int", "right_eyebrow_ext",
                     "left_mouth", "top_mouth", "right_mouth", "down_mouth",
                     "left_eyelid", "right_eyelid"]
    else:
        lmk_names = config["FER_lmk_name"]

    for fer_name in lmk_names:
        fer_path = os.path.join(config["directory"], config["LMK_data_directory"], config["condition"], "FER_LMK_pos" + "_" + fer_name + ".npy")
        FER_name_pos = np.load(fer_path)

        if len(np.shape(FER_pos)) == 0:
            FER_pos = FER_name_pos
        else:
            FER_pos = np.concatenate((FER_pos, FER_name_pos), axis=1)

    print("shape FER_pos", np.shape(FER_pos))
    np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FER_LMK_pos"), FER_pos)

    return FER_pos


if __name__ == "__main__":
    # %% import config
    config_path = 'BH_01_morph_space_with_NRE_m0001.json'
    # load config
    config = load_config(config_path, path='configs/behavourial')
    print("-- Config loaded --")
    print()

    merge_LMK_pos(config)
