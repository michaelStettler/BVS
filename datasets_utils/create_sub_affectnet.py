import pandas as pd
import os

"""
Create a new subset as a .csv file of the whole AffectNet dataset
"""

# declare variables
path = 'D:/Dataset/AffectNet'  # windows
full_data = 'training_modified.csv'
# full_data = 'validation_modified.csv'
expression_dict = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain',
                   'Non-Face']

# parameter for the sub datasets
classes = [0, 1, 2, 3, 4, 5, 6, 7]
num_per_class = 500
csv_name = 'train_sub8_4000.csv'
# csv_name = 'val_sub8.csv'

# print info of the full dataset
df = pd.read_csv(os.path.join(path, full_data))
print("num entry", len(df.index))
print(df.info())
print()

# create sub dataset
sub_df = pd.DataFrame()
for c in classes:
    # keep only the class of interest
    tmp_df = df[df.expression == c]
    print(expression_dict[c], ":", len(tmp_df.index))
    # keep only the num_per_class entry
    tmp_df = tmp_df.iloc[:num_per_class]
    # add data to the new sub dataset
    sub_df = sub_df.append(tmp_df)

# save sub dataset
print()
print("new len", len(sub_df.index))
sub_df.to_csv(os.path.join(path, csv_name), index=False)
