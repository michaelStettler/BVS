import os
import numpy as np
import face_alignment
from skimage import io
import matplotlib.pyplot as plt
import tqdm

from utils.load_config import load_config
from utils.load_data import load_data


"""
run: python -m projects.memory_efficiency.05a_reduce_SOTA_LMK
"""
#%%
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

# declare FAN network
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

#%%
# define configuration
config_file = 'NR_03_FERG_from_LMK_m0001.json'
# load config
config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')
print("-- Config loaded --")
print()

#%%
# Load data
train_data = load_data(config, get_raw=True)
train_label = train_data[1]
test_data = load_data(config, train=False, get_raw=True)
test_label = test_data[1]
print("shape train_data[0]", np.shape(train_data[0]))
print("shape test_data[0]", np.shape(test_data[0]))

#%%
# predict LMK
def predict_lmk(data):
    lmks_pos = []
    for i, img in enumerate(tqdm.tqdm(data)):
        lmk_preds = np.array(fa.get_landmarks(img))

        if len(np.shape(lmk_preds)) == 0:
            print(f"wrong size for lmk {i}: {np.shape(lmk_preds)}")
            lmk_preds = np.zeros((1, 68, 2))

        if len(lmk_preds) > 1:
            lmk_preds = np.expand_dims(lmk_preds[0], axis=0)

        if lmk_preds.shape != (1, 68, 2):
            print(f"Issue at {i}", np.shape(lmk_preds))
            break
        lmks_pos.append(lmk_preds)

    return np.array(lmks_pos, dtype=np.float16)


# predict lmk
train_lmk = predict_lmk(train_data[0])
test_lmk = predict_lmk(test_data[0])

#%%
# save lmk
np.save(os.path.join(config['directory'], "FAN_train_LMK.npy"), train_lmk)
np.save(os.path.join(config['directory'], "FAN_test_LMK.npy"), test_lmk)
