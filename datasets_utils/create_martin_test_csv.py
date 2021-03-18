import os
import pandas as pd
import numpy as np

np.random.seed(0)

"""
create csv for martin test

run: python -m datasets_utils.create_martin_test_csv
"""
path = 'D:/Dataset/martin_test'
csv_name = 'martin_test.csv'

df = pd.DataFrame(columns=["image_path", "category"])
print(list(df.columns.values))

def splitall(path):
    """
    helper function to split all the folders within a path

    used to remove the data folder and create a csv that is relative to where the csv file is saved
    :param path:
    :return:
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_files(path):
    # get number of folder of the path
    path_length = len(path.split('/'))

    # find all images
    files_name = []
    for root, dirs, files in sorted(os.walk(path)):
        for file in files:
            if isinstance(file, str):
                if file.endswith((".jpg", ".jpeg", ".png", "JPEG")):
                    # get full path
                    image_path = os.path.join(root, file)
                    image_path = image_path.replace("\\", "/")

                    # create relative path name
                    splits = splitall(image_path)
                    relat_img_path = os.path.join(*splits[path_length:])
                    relat_img_path = relat_img_path.replace('\\', '/')

                    files_name.append(relat_img_path)
    return np.array(files_name)

# get all files
files = get_files(path)
print("found {} images".format(len(files)))

# add face images to the dataframe
for f, file in enumerate(files):
    category = 0
    if "HumanAvatar_c2" in file:
        category = 1
    elif "HumanAvatar_c3" in file:
        category = 2
    elif "MonkeyAvatar_c2" in file:
        category = 3
    elif "MonkeyAvatar_c3" in file:
        category = 4
    df = df.append({'image_path': file, 'category': category}, ignore_index=True)

print(df.head())
print("datafrane create for {} images".format(len(df.index)))

df.to_csv(os.path.join(path, csv_name), index=True)
