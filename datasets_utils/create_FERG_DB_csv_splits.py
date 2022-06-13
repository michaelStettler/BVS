import os
import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

db_path = "/Users/michaelstettler/PycharmProjects/BVS/data/FERG_DB_256"
csv_name = 'FERG_DB'
categories_name = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

avatars_path = os.listdir(db_path)
print("avatars:")
print(avatars_path)


# function to retrieve all files
def get_files(path, folder):
    # find all face images
    face_files = []
    path_files = []
    for root, dirs, files in os.walk(os.path.join(path, folder)):
        for file in files:
            if isinstance(file, str):
                if file.endswith((".jpg", ".jpeg", ".png", ".JPEG")):
                    face_files.append(file)
                    path_files.append(folder.replace("\\", "/"))
    return np.array(face_files), np.array(path_files)


# get all files
images = np.array([])
category = np.array([])
paths = np.array([])

# define arrays to sort each category individually
anger_idx = np.array([])
disgust_idx = np.array([])
fear_idx = np.array([])
joy_idx = np.array([])
neutral_idx = np.array([])
sadness_idx = np.array([])
surprise_idx = np.array([])
n_images = 0

# sort all images for each avatars
for avatar in avatars_path:
    if 'README' not in avatar and '.DS' not in avatar:
        for cat in categories_name:
            # get all images from sub folders
            files, file_path = get_files(db_path, os.path.join(avatar, avatar + '_' + cat))

            # append new images
            images = np.concatenate((images, files))
            paths = np.concatenate((paths, file_path))

            # build category vector
            cat_v = [cat for x in range(len(files))]
            category = np.concatenate((category, cat_v))

            # store idx
            idx = np.arange(n_images, len(files) + n_images)
            if cat == 'anger':
                anger_idx = np.concatenate((anger_idx, idx))
            elif cat == 'disgust':
                disgust_idx = np.concatenate((disgust_idx, idx))
            elif cat == 'fear':
                fear_idx = np.concatenate((fear_idx, idx))
            elif cat == 'joy':
                joy_idx = np.concatenate((joy_idx, idx))
            elif cat == 'neutral':
                neutral_idx = np.concatenate((neutral_idx, idx))
            elif cat == 'sadness':
                sadness_idx = np.concatenate((sadness_idx, idx))
            elif cat == 'surprise':
                surprise_idx = np.concatenate((surprise_idx, idx))
            else:
                print("Issue with category:", cat)
            n_images += len(files)


print("shape images", np.shape(images))
print(images[:5])
print("shape paths", np.shape(paths))
print(paths[:5])
print("shape category", np.shape(category))
print(category[:5])
print()
print("n_anger:", len(anger_idx))
print("n_disgust:", len(disgust_idx))
print("n_fear:", len(fear_idx))
print("n_joy:", len(joy_idx))
print("n_neutral:", len(neutral_idx))
print("n_sadness:", len(sadness_idx))
print("n_surprise:", len(surprise_idx))

# split from all images
idx = np.arange(len(images))
np.random.shuffle(idx)
print("idx[:10]", idx[:10])
test_idx = idx[:6000]
val_idx = idx[6000:12000]
train_idx = idx[12000:]

test_array = np.array([images[test_idx], category[test_idx], paths[test_idx]]).transpose()
val_array = np.array([images[val_idx], category[val_idx], paths[val_idx]]).transpose()
train_array = np.array([images[train_idx], category[train_idx], paths[train_idx]]).transpose()
print("shape test_array", np.shape(test_array))
print("shape val_array", np.shape(val_array))
print("shape train_array", np.shape(train_array))

# build dataframe
test_df = pd.DataFrame(data=test_array, columns=['images', "category", "path"])
val_df = pd.DataFrame(data=val_array, columns=['images', "category", "path"])
train_df = pd.DataFrame(data=train_array, columns=['images', "category", "path"])

# save dataframe as csv
test_df.to_csv(os.path.join(db_path, csv_name + "_test.csv"))
val_df.to_csv(os.path.join(db_path, csv_name + "_val.csv"))
train_df.to_csv(os.path.join(db_path, csv_name + "_train.csv"))
