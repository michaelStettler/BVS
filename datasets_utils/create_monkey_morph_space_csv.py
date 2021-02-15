import os
import pandas as pd

path = '/app/Data/Dataset/MonkeyMorph'
avatar_type = "monkey"
csv_train_name = "monkey_morph_train.csv"
csv_val_name = "monkey_morph_val.csv"

df_train = pd.DataFrame(columns=("image", "category", "avatar", "strength", "path"))
df_val = pd.DataFrame(columns=("image", "category", "avatar", "strength", "path"))
for root, dirs, files in os.walk(os.path.join(path, "images")):
    print("num files:", len(files))
    print("dirs", root)

    for file in sorted(files):
        if isinstance(file, str):
            if file.endswith((".jpg", ".jpeg", ".png")):
                words = file.split('_')
                args = words[-1].split('.')

                # set some frame to be the Neutral category
                if float(words[2]) >= float(words[4]):
                    category = 'Threat' # angry
                else:
                    category = 'Fear'  # fear

                if int(args[2]) < 20 or int(args[2]) >= 80:
                    category = "Neutral"

                # add entry to the data frame
                if float(words[2]) == 1.0 or float(words[4]) == 1.0:
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
