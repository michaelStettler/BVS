import os
import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
run: python -m datasets_utils.create_FERG_DB_LMK_merge
"""

# declare variable
db_path = "/Users/michaelstettler/PycharmProjects/BVS/data/FERG_DB_256"
save_name = 'FERG_DB_LMK_pos'
avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
conditions = ['train', 'test']

for condition in conditions:
    # load csv file
    csv_file = 'FERG_DB_' + condition + '.csv'
    df = pd.read_csv(os.path.join(db_path, csv_file))
    print(df.head())

    # load lmk files
    lmk_files = []
    for a in range(len(avatar_names)):
        if condition == 'train':
            lmk_name = avatar_names[a] + '_lmk_pos.npy'
        elif condition == 'test':
            lmk_name = 'test_' + avatar_names[a] + '_lmk_pos.npy'
        lmks = np.load(os.path.join(db_path, 'saved_lmks_pos', lmk_name)).astype(np.float32)
        lmk_files.append(np.load(os.path.join(db_path, 'saved_lmks_pos', lmk_name)).astype(np.float32))
    print("len lmk_files", len(lmk_files))

    # merge lmk files
    lmk_pos = []
    avatar_types = []
    counters = np.zeros(len(avatar_names))
    print("counters", counters)
    for i, row in df.iterrows():
        img_name = row['images']
        for a in range(len(avatar_names)):
            if avatar_names[a] in img_name:
                lmk_pos.append(lmk_files[a][int(counters[a])])
                counters[a] += 1
                avatar_types.append(a)

    print("counters", counters)
    print("total images:", np.sum(counters))
    print("len lmk_pos", len(lmk_pos))
    lmk_pos = np.array(lmk_pos).astype(np.float16)
    print("shape lmk_pos", np.shape(lmk_pos))

    if condition == 'train':
        np.save(os.path.join(db_path, save_name), lmk_pos)
        np.save(os.path.join(db_path, 'FERG_DB_avatar_type'), avatar_types)
    elif condition == 'test':
        np.save(os.path.join(db_path, 'test_' + save_name), lmk_pos)
        np.save(os.path.join(db_path, 'test_FERG_DB_avatar_type'), avatar_types)
