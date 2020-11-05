import numpy as np
import pandas as pd
import cv2
import os
import warnings
from tqdm import tqdm
import tensorflow as tf

from utils.data_generator import DataGen


def load_data(config, train=True, sort_by=None):
    if config['train_data'] == 'test':
        print("[DATA] Generate random training data")
        np.random.seed(0)
        n_data = 36
        n_classes = config['n_category']
        x = np.random.rand(n_data, 224, 224, 3)
        y = np.random.randint(n_classes, size=n_data)
        # y = np.eye(n_classes)[dataY.reshape(-1)]   # transform to one hot encoding
        data = [x, y]

    elif config['train_data'] == 'monkey_test':
        data = _loaf_monkey(config, train, sort_by)

    elif config['train_data'] == 'FEI':
        data = _load_FEI(config)

    elif config['train_data'] == 'affectnet':
        data = _load_affectnet(config, train)
    else:
        raise ValueError("training data: '{}' does not exists! Please change norm_base_config file or add the training data"
                         .format(config['train_data']))

    return data


def _loaf_monkey(config, train, sort_by):
    if train:
        df = pd.read_csv(config['csv_train'])
    else:
        df = pd.read_csv(config['csv_val'])

    if sort_by is not None:
        df = df.sort_values(by=sort_by)

    # keep only the 100% conditions
    num_data = len(df.index)
    print("[DATA] Found {} images".format(num_data))
    # print(df.head())

    #  declare x and y
    x = np.zeros((num_data, 224, 224, 3))
    y = np.zeros(num_data)

    # create training and label data
    idx = 0
    for index, row in tqdm(df.iterrows()):
        # load img
        im = cv2.imread(os.path.join(row['path'], row['image']))
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # crop image
        im_crop = im_rgb[:, 280:1000, :]
        # resize image
        im_rgb = cv2.resize(im_crop, (224, 224))
        x[idx, :, :, :] = im_rgb

        category = -1
        if row['category'] == "Neutral":
            category = 0
        elif row['category'] == "Threat":
            category = 1
        elif row['category'] == "Fear":
            category = 2
        elif row['category'] == "LipSmacking":
            category = 3
        else:
            warnings.warn(
                "WARNING! image: '{}' category '{}' seems not to exists!".format(row['image'], row['category']))
        y[idx] = category

        idx += 1
    return [x, y]


def _load_FEI(config):
    # get csv
    df = pd.read_csv(config['csv'])
    num_data = len(df.index)
    print("[DATA] Found {} images".format(num_data))
    # print(df.head())

    #  declare x and y
    x = np.zeros((num_data, 224, 224, 3))
    y = np.zeros(num_data)

    # create training and label data
    idx = 0
    for index, row in tqdm(df.iterrows()):
        # load img
        im = cv2.imread(os.path.join(row['path'], row['image']))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # resize image
        height, width, depths = im.shape
        if height > width:
            ratio = height / 224
            dim = (int(width / ratio), 224)
        else:
            ratio = width / 224
            dim = (224, int(height / ratio))
        im = cv2.resize(im, dim)
        # pad image
        height, width, depths = im.shape
        img = np.zeros((224, 224, 3))
        if height > width:
            pad = int((224 - width) / 2)
            img[:, pad:pad + width, :] = im
        else:
            pad = int((224 - height) / 2)
            img[pad:pad + height, :, :] = im
        x[idx, :, :, :] = img

        category = -1
        if row['category'] == "face":
            category = 0
        elif row['category'] == "non_face":
            category = 1
        else:
            warnings.warn(
                "WARNING! image: '{}' category '{}' seems not to exists!".format(row['image'], row['category']))
        y[idx] = category

        idx += 1

    return [x, y]


def _load_affectnet(config, train):
    # get csv
    if train:
        df = pd.read_csv(config['csv_train'])
        directory = config['train_directory']
    else:
        df = pd.read_csv(config['csv_val'])
        directory = config['val_directory']
    num_data = len(df.index)
    print(df.head)
    print("[DATA] Found {} images".format(num_data))
    print(df.info())
    # print(df.head())

    #  declare generator
    dataGen = DataGen(config, df, directory)

    return dataGen
