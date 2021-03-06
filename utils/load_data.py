import numpy as np
import pandas as pd
import cv2
import os
import warnings
from tqdm import tqdm
import tensorflow as tf

from utils.data_generator import DataGen
from utils.image_utils import resize_image
from utils.image_utils import pad_image


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

    elif config['train_data'] == 'FEI_SA':
        data = _load_FEI_SA(config)

    elif config['train_data'] == 'FEI_semantic':
        data = _load_FEI_semantic(config)

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

        # resize and pad image
        img = resize_image(im)
        img = pad_image(img)

        # add image to data
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


def _load_FEI_SA(config):
    # get csv
    df = pd.read_csv(config['csv'])
    num_data = len(df.index)
    print("[DATA] Found {} images".format(num_data))
    # print(df.head())

    #  declare x and y
    x = np.zeros((num_data, 224, 224, 3))
    y = np.zeros((num_data, 50))

    # create training and label data
    idx = 0
    for index, row in tqdm(df.iterrows()):
        # load img
        im = cv2.imread(os.path.join(config['SA_img_path'], row['img_name']))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # resize and pad image
        img = resize_image(im)
        img = pad_image(img)

        # add image to data
        x[idx, :, :, :] = img

        # transform string back to array
        S = row['S']
        A = row['A']
        # remove the [] and split with space
        S = S[1:-1].split(' ')
        A = A[1:-1].split(' ')
        # transform to array and remove the empty space
        S = [float(s) for s in S if s != '']
        A = [float(a) for a in A if a != '']
        # set shape and appearance index to the label
        y[idx, :25] = S
        y[idx, 25:] = A

        # increase idx
        idx += 1

    return [x, y]


def _load_FEI_semantic(config):
    # get csv
    df = pd.read_csv(config['csv'])
    num_data = len(df.index)
    print("[DATA] Found {} images".format(num_data))

    #  declare x and y
    x = np.zeros((num_data, 224, 224, 3))
    y = np.zeros((num_data, 224, 224, config['num_cat']))

    for idx, row in df.iterrows():
        # load image
        im = cv2.imread(os.path.join(config['img_path'], row['img']))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # resize and pad image
        img = resize_image(im)
        img = pad_image(img)
        # add image to data
        x[idx] = img

        # load label
        label = np.load(os.path.join(config['label_path'], row['label']))
        # resize and pad label
        label = resize_image(label)
        label = pad_image(label, dim=(224, 224, config['num_cat']))
        # add label to data
        y[idx] = label

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
