#%%
import numpy as np
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_data import load_data
from plots_utils.plot_BVS import display_images

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

#%%
# define configuration
config_path = 'NR_02_FERG_m0001.json'

# load config
config = load_config(config_path, path='configs/norm_reference')

#%% Load training data
train_data = load_data(config)
print("shape train_data[0]", np.shape(train_data[0]))

"""
Labeled images

| Avatar   |                                                       Ray |
|:---------|----------------------------------------------------------:|
| Neutral  | 21,  37,  49,  73,  84, 145, 148, 150, 222, 238, 278, 306 |
| Happy    | 43, 124, 166, 212, 294, 328, 504, 574, 637, 672, 703, 761 |
| Angry    | 38,  67,  76,  89,  99, 123, 151, 158, 162, 232, 320, 345 |
| Sad      | 15,  22,  47, 146, 227, 420, 468, 559, 579, 626, 646, 721 |
| Surprise | 55, 130, 139, 160, 189, 219, 248, 401, 446, 540, 557, 570 |
| Fear     |  1,   8,  51,  62,  72, 105, 164, 186, 260, 267, 380, 381 |
| Disgust  |  0,  26,  71,  75,  92, 142, 147, 185, 194, 221, 225, 233 |

| Avatar   |                                                     Bonnie |
|:---------|-----------------------------------------------------------:|
| Neutral  |   6,  12,  24,  29, 106, 177, 192, 224, 231, 271, 275, 289 |
| Happy    |  30, 182, 340, 352, 357, 362, 383, 390, 429, 439, 475, 478 |
| Angry    |   2,   4,  44,  85, 107, 109, 181, 195, 255, 259, 272, 313 |
| Sad      | 126, 159, 178, 200, 206, 207, 246, 290, 301, 317, 428, 454 |
| Surprise |  34,  59,  97, 129, 157, 165, 197, 208, 251, 397, 411, 449 |
| Fear     |   7,  16, 127, 175, 179, 282, 285, 293, 314, 346, 355, 389 |
| Disgust  |  19,  32,  33,  41,  58,  64,  69,  86, 102, 104, 149, 174 |

| Avatar   |   Jules |
|:---------|--------:|
| Neutral  | 20, 163 |
| Happy    |  3,  98 |
| Angry    | 81, 100 |
| Sad      | 17,  35 |
| Surprise | 14,  27 |
| Fear     | 52,  74 |
| Disgust  | 80, 114 |

| Avatar   |  Malcolm |
|:---------|---------:|
| Neutral  |  77, 135 |
| Happy    |  68,  96 |
| Angry    |  65, 136 |
| Sad      |  39,  53 |
| Surprise | 125, 138 |
| Fear     |   5,  28 |
| Disgust  |  90, 180 |

| Avatar   |           Aia |
|:---------|--------------:|
| Neutral  |    13, 25, 45 |
| Happy    |        50, 61 |
| Angry    |        54, 66 |
| Sad      |        36, 56 |
| Surprise |        48, 70 |
| Fear     | 9, 11, 18, 46 |
| Disgust  |        60, 94 |

| Avatar   |     Mery |
|:---------|---------:|
| Neutral  |  10,  42 |
| Happy    |  40,  83 |
| Angry    |  23, 287 |
| Sad      |  31, 203 |
| Surprise |  79, 121 |
| Fear     |  91, 261 |
| Disgust  | 270, 461 |

"""

#%%
# create labeling
# l_eye, r_eye, nose, l_eyebrow_ext, l_eyebrow_int, r_eyebrow_int, r_eyebrow_ext,
# l_corner_mouth, up_mouth, r_corner_mouth, lower_mouth,
# l_lower_eyelid, r_lower_eyelid

# [x, y] -> [horizontal, vertical]

save_name = "/Users/michaelstettler/PycharmProjects/BVS/data/FERG_DB_256/lmk_pos.npy"
lmk_pos = np.load(save_name, allow_pickle=True).item()

from_idx = 69
img_idx = 174
print("lmk_pos[{}]: {}".format(from_idx, lmk_pos[from_idx]))  # print existing pos of same avatar/expression
lmk_pos[img_idx] = [[[91, 117], [137, 117], [114, 142], [75, 92], [100, 96], [128, 90], [154, 82], [92, 172], [114, 154], [135, 174], [114, 186], [83, 117], [145, 117]]]

print("len lmk_pos", len(lmk_pos))
print("idx:", img_idx)
display_images([train_data[0][img_idx]], lmks=lmk_pos[img_idx], lmk_size=3, size_img=8, pre_processing='VGG19')
np.save(save_name, lmk_pos)