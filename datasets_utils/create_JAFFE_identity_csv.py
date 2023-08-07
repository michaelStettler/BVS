import os
import pandas as pd
import numpy as np

np.random.seed(0)

# path = 'D:/Dataset/FEI'  # windows
path = r'C:\Users\Alex\Documents\Uni_Data\NRE\jaffedbase'
csv_name = 'JAFFE_face_units.csv'

df = pd.DataFrame(columns=("image", "category", "image_path"))


def get_files(path, folder):
    # find all face images
    face_files = []
    path_files = []
    for root, dirs, files in os.walk(os.path.join(path, folder)):
        for file in files:
            if isinstance(file, str):
                if file.endswith((".tiff")):
                    face_files.append(file)
                    path_files.append(f"{folder}/{file}")
    return np.array(face_files), np.array(path_files)


face_files, face_path_files = get_files(path, "jaffedbase")

# add face images to the dataframe
category = 0
#TODO Put correct categories (expressions)
for f, file in enumerate(face_files):
    df = df.append({'image': file, 'category': category, 'image_path': face_path_files[f]}, ignore_index=True)

for identity in ['KA', 'KL', 'KM', 'KR', 'MK', 'NA', 'NM', 'TM', 'UY', 'YM']:
    id_df = df[df['image'].str.contains(identity)]
# print("datafrane create for {} images".format(len(df.index)))

    id_df.to_csv(os.path.join(path, f"JAFFE_{identity}_id.csv"), index=False)
