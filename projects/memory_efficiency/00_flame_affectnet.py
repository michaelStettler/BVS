import os
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
run: python -m projects.memory_efficiency.00_flame_affectnet
"""
path = "/Users/michaelstettler/PycharmProjects/BVS/data/AffectNet_FLAME"
train_csv = "flame_training_params.csv"
test_csv = "flame_validation_params.csv"

df_train = pd.read_csv(os.path.join(path, train_csv), index_col=0)
print(df_train.head())

train_x = df_train['params']
print("shape train_x", np.shape(train_x))
print(train_x[0])
test = train_x[0]
test = test[1:-1]  # remove '[]'
test = test.split('\n')
print(test)

# process data
# read csv columns that are stored as a string of '[xxx yyy zzz]' and set it to a numpy array
train_data = [[float(t) for i in range(len(train_x[j][1:-1].split('\n'))) for t in train_x[j][1:-1].split('\n')[i].split(' ') if t != ""] for j in range(10)]
train_x = np.array(train_data)

# process label
train_y = []
for y in tqdm(df_train['expression']):
    if y == 'Neutral':
        train_y.append(0)
    elif y == 'Happy':
        train_y.append(1)
    elif y == 'Anger':
        train_y.append(2)
    elif y == 'Sad':
        train_y.append(3)
    elif y == 'Surprise':
        train_y.append(4)
    elif y == 'Fear':
        train_y.append(5)
    elif y == 'Disgust':
        train_y.append(6)
    elif y == 'Contempt':
        train_y.append(7)
    else:
        raise ValueError("{} is not a valid argument".format(y))
train_y = np.array(train_y)

print("train_x", np.shape(train_x))
print(train_x[0, :5])
print("shape train_y", np.shape(train_y))
print(train_y[:5])
