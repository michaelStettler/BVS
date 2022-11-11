import os
import numpy as np

from utils.load_config import load_config

"""
run: python -m datasets_utils.merge_LMK_pos
"""

#%% import config
config_path = 'BH_01_morph_space_with_NRE_m0001.json'
# load config
config = load_config(config_path, path='configs/behavourial')
print("-- Config loaded --")
print()

FER_pos = None
for fer_name in config["FER_lmk_name"]:
    fer_path = os.path.join(config["directory"], config["LMK_data_directory"], "FER_LMK_pos" + "_" + fer_name + ".npy")
    FER_name_pos = np.load(fer_path)

    if len(np.shape(FER_pos)) == 0:
        FER_pos = FER_name_pos
    else:
        FER_pos = np.concatenate((FER_pos, FER_name_pos), axis=1)

print("shape FER_pos", np.shape(FER_pos))
np.save(os.path.join(config["directory"], config["LMK_data_directory"], "FER_LMK_pos"), FER_pos)
