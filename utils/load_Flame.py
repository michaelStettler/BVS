import pandas as pd
import numpy as np
from tqdm import tqdm


def load_FLAME_csv_params(df):
    params = df['params']
    print("shape params", np.shape(params))
    # process data
    # read csv columns that are stored as a string of '[xxx yyy zzz]' and set it to a numpy array
    params_data = [
        [float(t) for i in range(len(params[j][1:-1].split('\n'))) for t in params[j][1:-1].split('\n')[i].split(' ')
         if t != ""] for j in range(500)]
    # params_data = [
    #     [float(t) for i in range(len(params[j][1:-1].split('\n'))) for t in params[j][1:-1].split('\n')[i].split(' ')
    #      if t != ""] for j in range(len(params))]
    params_data = np.array(params_data)

    # process label
    label = []
    for y in tqdm(df['expression']):
        if y == 'Neutral':
            label.append(0)
        elif y == 'Happy':
            label.append(1)
        elif y == 'Anger':
            label.append(2)
        elif y == 'Sad':
            label.append(3)
        elif y == 'Surprise':
            label.append(4)
        elif y == 'Fear':
            label.append(5)
        elif y == 'Disgust':
            label.append(6)
        elif y == 'Contempt':
            label.append(7)
        else:
            raise ValueError("{} is not a valid argument".format(y))

    label = np.array(label)
    label = label[:500]

    return [params_data, label]