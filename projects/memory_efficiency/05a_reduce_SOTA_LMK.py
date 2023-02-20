import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from utils.load_config import load_config


"""
run: python -m projects.memory_efficiency.05a_reduce_SOTA_LMK
"""
#%%
np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

cond = 1
draw = False
conditions = ["FAN", "MediaPipe"]
red_idx = [
    [17, 19, 21, 22, 24, 26, 37, 38, 40, 41, 43, 44, 46, 47, 48, 50, 51, 52, 54, 56, 57, 58],  # FAN
    [163, 145, 154, 161, 159, 157,  # right eye
     390, 374, 381, 388, 386, 384,  # left eye
     46, 52, 55,  # right eyebrow
     276, 282, 285,  # left eyebrow
     61, 181, 17, 405, 39, 0, 269  # mouth
     ]  # MediaPipe
]
n_lmk = len(red_idx[cond])
print("Condition:", conditions[cond])
print("n_lmk:", n_lmk)

#%%
# define configuration
config_file = 'NR_03_FERG_from_LMK_m0001.json'
# load config
config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')
print("-- Config loaded --")
print()

#%%
# Load data
config['train_lmk_pos'] = f"{config['directory']}/{conditions[cond]}_train_LMK.npy"
config['test_lmk_pos'] = f"{config['directory']}/{conditions[cond]}_test_LMK.npy"

# load lmk pos
train_data = np.load(config['train_lmk_pos'])
test_data = np.load(config['test_lmk_pos'])

if conditions[cond] == "FAN":
    train_data = np.squeeze(train_data)
    test_data = np.squeeze(test_data)
    train_data = train_data.astype(int)
    test_data = test_data.astype(int)
elif conditions[cond] == "MediaPipe":
    train_data = np.round(train_data * 224).astype(int)
    test_data = np.round(test_data * 224).astype(int)

print("shape train_data", np.shape(train_data))
print("shape test_data", np.shape(test_data))

#%%
# draw landmark
if draw:
    lmk0 = train_data[0]
    print("shape lmk0", lmk0.shape)
    print("min max 0", np.min(lmk0[:, 0]), np.max(lmk0[:, 0]))
    print("min max 1", np.min(lmk0[:, 1]), np.max(lmk0[:, 1]))
    for i in tqdm.tqdm(range(len(lmk0))):
        img = np.zeros((224, 224, 3))

        # draw each idx
        for l, lmk in enumerate(lmk0):
            # if idx of interest, draw in red
            if i == l:
                img[lmk[1]-1:lmk[1]+2, lmk[0]-1:lmk[0]+2] = [1, 0, 0]
            else:
                img[lmk[1]-1:lmk[1]+2, lmk[0]-1:lmk[0]+2] = [1, 1, 1]

        # convert to matplotlib and save
        plt.figure()
        plt.imshow(img)
        plt.title(f"idx_{i}")
        plt.savefig(f"img_{conditions[cond]}_{i}.jpeg")
        plt.close()

#%% filter lmk
train_lmk = train_data[:, red_idx[cond]]
test_lmk = test_data[:, red_idx[cond]]
print("shape train_lmk", np.shape(train_lmk))
print("shape test_lmk", np.shape(test_lmk))

#%% save lmk
np.save(os.path.join(config['directory'], f"{conditions[cond]}_train_LMK_{n_lmk}.npy"), train_lmk)
np.save(os.path.join(config['directory'], f"{conditions[cond]}_test_LMK_{n_lmk}.npy"), test_lmk)
