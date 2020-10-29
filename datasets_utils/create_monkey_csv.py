import os
import pandas as pd

# path = '/Users/michaelstettler/PycharmProjects/BVS/data/expressivity_level'  # mac
path = 'D:/Stimuli/MonkeyExpressivityLevel/v3'  # windows
neutral_frame = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009',
                 '00010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019',
                 '0020']
avatar_type = "monkey"
csv_train_name = "monkey_train.csv"
csv_val_name = "monkey_val.csv"

df_train = pd.DataFrame(columns=("image", "category", "avatar", "strength", "path"))
df_val = pd.DataFrame(columns=("image", "category", "avatar", "strength", "path"))
for root, dirs, files in os.walk(os.path.join(path, "images")):
    print("num files:", len(files))
    print("dirs", root)

    for file in files:
        if isinstance(file, str):
            if file.endswith((".jpg", ".jpeg", ".png")):
                words = file.split('_')
                args = words[4].split('.')

                # set some frame to be the Neutral category
                category = words[1]
                if args[1] in neutral_frame:
                    category = "Neutral"

                # add entry to the data frame
                if words[2] == "1.0":
                    df_train = df_train.append({'image': file, 'category': category, 'avatar': avatar_type,
                                                'strength': words[2], 'path': root}, ignore_index=True)
                else:
                    df_val = df_val.append({'image': file, 'category': category, 'avatar': avatar_type,
                                            'strength': words[2], 'path': root}, ignore_index=True)

print(df_train.head())
print(df_val.head())
# save to csv
df_train.to_csv(os.path.join(path, "csv", csv_train_name), index=False)
df_val.to_csv(os.path.join(path, "csv", csv_val_name), index=False)
