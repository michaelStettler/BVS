import os
import pandas as pd

path = '/Users/michaelstettler/PycharmProjects/BVS/data/ExpressivityLevels'  # mac
# path = 'D:/Dataset/ExpressivityLevels/'
neutral_frame = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009',
                 '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019',
                 '0020']
avatar_type = "monkey"
csv_train_name = "monkey_train.csv"
csv_val_name = "monkey_val.csv"

df_train = pd.DataFrame(columns=("image_path", "image_name", "category", "avatar", "strength"))
df_val = pd.DataFrame(columns=("image_path", "image_name", "category", "avatar", "strength", "path"))
for root, dirs, files in os.walk(os.path.join(path, "images")):
    print("num files:", len(files))
    print("root", root)
    print("dirs", dirs)

    for file in sorted(files):
        if isinstance(file, str):
            if file.endswith((".jpg", ".jpeg", ".png")):
                words = file.split('_')
                args = words[4].split('.')

                # get local path
                local_path = os.path.relpath(root, start=path)
                local_path = local_path.replace('\\', '/')

                # get image_path
                image_path = os.path.join(local_path, file)
                image_path = image_path.replace('\\', '/')

                # set some frame to be the Neutral category
                category = words[1]
                if args[1] in neutral_frame:
                    category = "Neutral"

                # add entry to the data frame
                if words[2] == "1.0":
                    df_train = df_train.append({'image_path': image_path, 'image_name': file, 'category': category,
                                                'avatar': avatar_type, 'strength': words[2]}, ignore_index=True)
                else:
                    df_val = df_val.append({'image_path': image_path, 'image_name': file, 'category': category,
                                            'avatar': avatar_type, 'strength': words[2]}, ignore_index=True)

print(df_train.head())
print(df_val.head())
# save to csv
df_train.to_csv(os.path.join(path, "csv", csv_train_name), index=False)
df_val.to_csv(os.path.join(path, "csv", csv_val_name), index=False)
