import os
import pandas as pd
import numpy as np

np.random.seed(0)

# declare variables
path = '/Users/michaelstettler/PycharmProjects/BVS/data/KDEF_and_AKDEF'  # personal mac
csv_train_name = 'KDEF_facial_expr_train.csv'
csv_test_name = 'KDEF_facial_expr_test.csv'

# declare dataframes
df_train = pd.DataFrame(columns=["image", "category", "image_path"])
df_test = pd.DataFrame(columns=("image", "category", "image_path"))


def get_files(path, folder):
    # find all face images
    face_list = []
    path_list = []
    categories_list = []
    for root, dirs, files in os.walk(os.path.join(path, folder)):
        for file in files:
            if isinstance(file, str):
                if file.endswith((".JPG", ".jpg", ".jpeg", ".png", "JPEG")):
                    # keep only the image that has front view (straigth)
                    if 'S.' in file:
                        face_list.append(file)

                        img_path = root.replace("\\", "/")
                        img_path = img_path.split("/")[-2:]
                        img_path = '/'.join(img_path)
                        path_list.append(os.path.join(img_path, file))


                        # sort category
                        if 'NES.' in file:   # Neutral
                            categories_list.append(0)
                        elif 'HAS.' in file:  # happy
                            categories_list.append(1)
                        elif 'ANS.' in file:  # angry
                            categories_list.append(2)
                        elif 'SAS.' in file:  # sad
                            categories_list.append(3)
                        elif 'SUS.' in file:  # surprise
                            categories_list.append(4)
                        elif 'AFS.' in file:  # afraid -> fear
                            categories_list.append(5)
                        elif 'DIS.' in file:  # disgust
                            categories_list.append(6)

    return np.array(face_list), np.array(path_list), np.array(categories_list)

face_list, path_list, category_list = get_files(path, "KDEF")
print("shape face_list", np.shape(face_list))
print("shape path_list", np.shape(path_list))
print("shape category_list", np.shape(category_list))

# for file_name, path_name, cat in zip(face_list, path_list, category_list):
#     print("img_name: {}, path: {}, category: {}".format(file_name, path_name, cat))

# create indices and shuffle indices
indices = np.arange(len(face_list))
np.random.shuffle(indices)

# shuffle the files
face_files = face_list[indices]
face_path_files = face_list[indices]

train_split = .7
test_split = 1 - train_split
n_train_img = int(len(face_files) * train_split)
n_test_img = int(len(face_files) * test_split)
print("n_train_img: {}, n_test_img: {}, total: {} ({})".format(n_train_img,
                                                               n_test_img,
                                                               n_train_img + n_test_img,
                                                               len(face_files)))

# create training set
training_face_list = face_list[:n_train_img]
training_path_list = path_list[:n_train_img]
training_cat_list = category_list[:n_train_img]

# create testing set
testing_face_list = face_list[n_train_img:]
testing_path_list = path_list[n_train_img:]
testing_cat_list = category_list[n_train_img:]

print("len training_face_list: {}, len testing_face_list: {}".format(len(training_face_list), len(testing_face_list)))

# create training csv
for file, img_path, category in zip(training_face_list, training_path_list, training_cat_list):
    new_entry_df = pd.DataFrame([[file, category, img_path]], columns=["image", "category", "image_path"])
    df_train = pd.concat([df_train, new_entry_df], ignore_index=True)

print("train df:")
print(df_train.head())
print()

# create testing csv
for file, img_path, category in zip(testing_face_list, testing_path_list, testing_cat_list):
    new_entry_df = pd.DataFrame([[file, category, img_path]], columns=["image", "category", "image_path"])
    df_test = pd.concat([df_test, new_entry_df], ignore_index=True)

print("test df")
print(df_test.head())

# save dataFrame
df_train.to_csv(os.path.join(path, csv_train_name), index=False)
df_test.to_csv(os.path.join(path, csv_test_name), index=False)
