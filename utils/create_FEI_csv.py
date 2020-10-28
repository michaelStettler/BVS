import os
import pandas as pd
import numpy as np

np.random.seed(0)

path = 'D:/Dataset/FEI'
csv_name = 'FEI_face_units.csv'

df = pd.DataFrame(columns=("image", "category", "path"))


def get_files(path, folder):
    # find all face images
    face_files = []
    path_files = []
    for root, dirs, files in os.walk(os.path.join(path, folder)):
        for file in files:
            if isinstance(file, str):
                if file.endswith((".jpg", ".jpeg", ".png", "JPEG")):
                    face_files.append(file)
                    path_files.append(root.replace("\\", "/"))
    return np.array(face_files), np.array(path_files)


face_files, face_path_files = get_files(path, "face_images")
# create indices and shuffle indices
indices = np.arange(len(face_files))
np.random.shuffle(indices)

# shuffle the files
face_files = face_files[indices]
face_path_files = face_path_files[indices]

# keep first 50 images
face_files = face_files[:50]
face_path_files = face_path_files[:50]

# add face images to the dataframe
category = "face"
for f, file in enumerate(face_files):
    df = df.append({'image': file, 'category': category, 'path': face_path_files[f]}, ignore_index=True)

# find all non-face images
non_face_files, non_face_path_files = get_files(path, "non_face_images")

# add non_face images to the dataframe
category = "non_face"
for f, file in enumerate(non_face_files):
    df = df.append({'image': file, 'category': category, 'path': non_face_path_files[f]}, ignore_index=True)

print(df.head())
print("datafrane create for {} images".format(len(df.index)))

df.to_csv(os.path.join(path, csv_name), index=False)
