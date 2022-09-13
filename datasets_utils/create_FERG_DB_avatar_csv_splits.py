import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# set random seed
np.random.seed(0)

"""
run: python -m datasets_utils.create_FERG_DB_avatar_csv_splits

Please run first: create_FERG_DB_csv_splits to be able to load the csv files
"""


def filter_df_by_avatar(csv_name, avatar_name, verbose=False):
    # load df
    df = pd.read_csv(csv_name)

    # create new df
    df_avatar = pd.DataFrame(columns=["images", "category", "image_path"])

    # loop over each row, should learn how to do a regex on column with filter or something...
    for index, row in tqdm(df.iterrows()):
        if avatar_name in row['images']:
            new_entry_df = pd.DataFrame([[row['images'], row['category'], row['image_path']]],
                                        columns=["images", "category", "image_path"])
            df_avatar = pd.concat([df_avatar, new_entry_df], ignore_index=True)

    if verbose:
        print("loaded df")
        print(df.head())
        print()
        print("new df")
        print(df_avatar.head())

    return df_avatar


if __name__ == '__main__':
    db_path = "/Users/michaelstettler/PycharmProjects/BVS/data/FERG_DB_256"
    avatar_names = ['jules', 'malcolm', 'ray', 'aia', 'bonnie', 'mery']
    avatar_name = avatar_names[4]
    conditions = ['train', 'val', 'test']

    for condition in conditions:
        csv_name = 'FERG_DB_' + condition + '_' + avatar_name + '.csv'

        load_csv_name = 'FERG_DB_' + condition + '.csv'

        # filter DataFrame by avatar
        new_df = filter_df_by_avatar(os.path.join(db_path, load_csv_name), avatar_name)

        # save new df
        new_df.to_csv(os.path.join(db_path, csv_name))


