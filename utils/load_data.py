import numpy as np
import pandas as pd
import cv2
import os
import warnings
from tqdm import tqdm
import tensorflow as tf

from utils.CSV_data_generator import CSVDataGen
from utils.image_utils import resize_image
from utils.image_utils import pad_image
from utils.load_from_csv import load_from_csv


def load_data(config, train=True, sort_by=None, get_raw=False):
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
        data = _load_monkey(config, train, sort_by)

    elif config['train_data'] == 'FEI':
        data = _load_FEI(config)

    elif config['train_data'] == 'FEI_SA':
        data = _load_FEI_SA(config)

    elif config['train_data'] == 'affectnet':
        data = _load_affectnet(config, train)

    elif config['train_data'] == 'morphing_space':
        data = _load_morphing_space(config, train, sort_by, get_raw=get_raw)

    elif config['train_data'] == 'basic_shapes':
        data = _load_basic_shapes(config, train)

    elif config['train_data'] == 'bfs_space':  # bfs = basic face shape
        data = _load_bfs(config, train, get_raw=get_raw)

    else:
        raise ValueError("training data: '{}' does not exists! Please change norm_base_config file or add the training data"
                         .format(config['train_data']))

    return data


def _load_monkey(config, train, sort_by):
    if train:
        df = pd.read_csv(config['csv_train'])
    else:
        df = pd.read_csv(config['csv_val'])

    try:
        directory = config['directory']
    except KeyError:
        directory = None

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
    for index, row in tqdm(df.iterrows(), total=num_data):
        # load img
        if directory is None:
            im = cv2.imread(os.path.join(row['image_path']))
        else:
            im = cv2.imread(os.path.join(directory, row['image_path']))
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # crop image
        im_crop = im_rgb[:, 280:1000, :]
        # resize image
        im_rgb = cv2.resize(im_crop, (224, 224))
        x[idx, :, :, :] = im_rgb

        # todo change csv or find a way to implement this within "load_from_csv"
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


def _load_morphing_space(config, train, sort_by, get_raw=False):
    # load csv
    df = pd.read_csv(config['csv'], index_col=0)

    # sort if argument is present
    if sort_by is not None:
        df = df.sort_values(by=sort_by)

    # select avatar
    if train:
        config_avatar = config['train_avatar']
        expressions = config['train_expression']
    else:
        config_avatar = config['val_avatar']
        expressions = config['val_expression']

    # build filter depending on avatar
    if config_avatar == 'all_orig':
        avatar = ['Monkey_orig', 'Human_orig']
    elif config_avatar == 'human_orig':
        avatar = ['Human_orig']
    elif config_avatar == 'monkey_orig':
        avatar = ['Monkey_orig']
    else:
        raise ValueError('Avatar {} does not exists in morphing_space dataset!'.format(config_avatar))
    # apply filter
    df = df[df['avatar'].isin(avatar)]

    # read out the expressions for training
    new_df = pd.DataFrame()
    for expression in expressions:
        sub_df = df.copy()
        if 'c1' in expression:  # 100% human - 100% angry
            h_exp = [1.0]
            anger = [1.0]
        elif 'c2' in expression:  # 100% human - 100% fear
            h_exp = [1.0]
            anger = [0.0]
        elif 'c3' in expression:  # 100% monkey - 100% angry
            h_exp = [0.0]
            anger = [1.0]
        elif 'c4' in expression:  # 100% monkey - 100% fear
            h_exp = [0.0]
            anger = [0.0]
        elif 'full' in expression:  # full morphing space
            h_exp = [1.0, 0.75, 0.5, 0.25, 0.0]
            anger = [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            raise NotImplementedError('Expression "{}" is not yet implemented'.format(expression))

        # select anger expression
        sub_df = sub_df[sub_df['anger'].isin(anger)]
        # select species expression
        sub_df = sub_df[sub_df['h_exp'].isin(h_exp)]

        # append to new df
        new_df = new_df.append(sub_df, ignore_index=True)

    num_data = len(new_df.index)
    print("[DATA] Found {} images".format(num_data))

    # set use_generator to false if non existant
    if config.get('use_data_generator') is None:
        config['use_data_generator'] = False

    if config['use_data_generator']:
        # get image target size
        input_shape = config['input_shape']
        target_size = tuple(input_shape[:-1])

        # make sure the categopry is a string
        new_df[config["y_col"]] = new_df[config["y_col"]].astype(str)

        # use tensorflow generator

        if get_raw:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        elif config["extraction_model"] == "VGG19":
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=tf.keras.applications.vgg19.preprocess_input)

        else:
            print("[LOAD DATA] Warning the preprocessing funtion may be wrong for {}".format(config["extraction_model"]))
        data = datagen.flow_from_dataframe(new_df, directory=config['directory'],
                                           x_col=config['x_col'],
                                           y_col=config["y_col"],
                                           target_size=target_size,
                                           batch_size=config['batch_size'])
    else:
        # load each image from the csv file
        data = load_from_csv(new_df, config)

        if get_raw:
            data[0] = data[0]
        elif config["extraction_model"] == "VGG19":
            data[0] = tf.keras.applications.vgg19.preprocess_input(np.copy(data[0]))
        else:
            print("[LOAD DATA] Warning the preprocessing funtion may be wrong for {}".format(config["extraction_model"]))

    return data


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


def _load_affectnet(config, train):
    # get csv
    if train:
        df = pd.read_csv(config['csv_train'])
        directory = config['train_directory']
    else:
        df = pd.read_csv(config['csv_val'])
        directory = config['val_directory']
    num_data = len(df.index)
    #print(df.head)
    print("[DATA] Found {} images".format(num_data))
    #print(df.info())
    # print(df.head())

    #  declare generator
    dataGen = CSVDataGen(config, df, directory)

    return dataGen


def _load_basic_shapes(config, train):
    from PIL import Image, ImageDraw
    color_background = (128,128,128)
    images = []
    if train == "black_circle_displacement":
        for displacement in [0,4,8,16,32]:
            im = Image.new('RGB', (224, 224), color_background)
            draw = ImageDraw.Draw(im)
            draw.ellipse([70,70+displacement,100,100+displacement], fill=(0,0,0))
            im = np.array(im)
            images.append(im)
    elif train == "eye_shape":
        im = Image.new('RGB', (224, 224), color_background)
        draw = ImageDraw.Draw(im)
        draw.ellipse((70, 80, 100, 100), fill=(255, 255, 255), outline=(0, 0, 0), width=3)
        draw.ellipse([80, 83, 90, 93], fill=(0, 0, 0))
        im = np.array(im)
        images.append(im)
    else:
        raise ValueError(
            f"basic shape with train: '{train}' does not exists! Please change norm_base_config file or add the training data")
    #images = np.array(images)
    return images


def _load_bfs(config, train, get_raw=False):
    """
    helper function to load the basic face shape dataset
    """
    # load csv
    df = pd.read_csv(config['csv'], index_col=0)

    # select avatar
    if train:
        config_avatar = config['train_avatar']
        expressions = config['train_expression']
    else:
        config_avatar = config['val_avatar']
        expressions = config['val_expression']

    x_scale = [1.0]
    y_scale = [1.0]
    # build filter depending on avatar
    if config_avatar == 'Louise':
        avatar = ['Louise']
    elif config_avatar == 'Louise_all_identities':
        avatar = ['Louise']
        x_scale = [0.8, 0.9, 1.0, 1.1, 1.2]
        y_scale = [1.0]
    elif config_avatar == 'Monkey':
        avatar = ['Monkey']
    elif config_avatar == 'Monkey_all_identities':
        avatar = ['Monkey']
        x_scale = [0.8, 0.9, 1.0, 1.1, 1.2]
        y_scale = [1.0]
    elif config_avatar == 'Merry':
        avatar = ['Mery']
    elif config_avatar == 'Merry_all_identities':
        avatar = ['Mery']
        x_scale = [0.8, 0.9, 1.0, 1.1, 1.2]
        y_scale = [1.0]
    elif config_avatar == 'all_test':
        avatar = ['Monkey', 'Mery']
    elif config_avatar == 'ref_shapes':
        avatar = ['Louise', 'Monkey', 'Mery']
    elif config_avatar == 'ref_shapes':
        avatar = ['Louise', 'Monkey', 'Mery']
        x_scale = [0.8, 0.9, 1.1, 1.2]
    else:
        raise ValueError('Avatar {} does not exists in morphing_space dataset!'.format(config_avatar))

    # apply filter
    df = df[df['avatar'].isin(avatar)]
    df = df[df['x_scale'].isin(x_scale)]
    df = df[df['y_scale'].isin(y_scale)]

    # read out the expressions for training
    new_df = pd.DataFrame()
    for expression in expressions:
        sub_df = df.copy()

        if 'Neutral' in expression:
            exp = [0]
        elif 'Happy' in expression:
            exp = [1]
        elif 'Angry' in expression:
            exp = [2]
        elif 'Sad' in expression:
            exp = [3]
        elif 'Surprise' in expression:
            exp = [4]
        elif 'Fear' in expression:
            exp = [5]
        elif 'Disgust' in expression:
            exp = [6]
        elif 'full' in expression:  # full set
            exp = [0, 1, 2, 3, 4, 5, 6]
        elif 'all_expressions' in expression:  # all expressions (no Neutral)
            exp = [1, 2, 3, 4, 5, 6]
        else:
            raise NotImplementedError('Expression "{}" is not yet implemented'.format(expression))

        # select anger expression
        sub_df = sub_df[sub_df['category'].isin(exp)]

        # append to new df
        new_df = new_df.append(sub_df, ignore_index=True)

    # load each image from the csv file
    data = load_from_csv(new_df, config)

    if get_raw:
        data[0] = data[0]
    elif config["extraction_model"] == "VGG19":
        data[0] = tf.keras.applications.vgg19.preprocess_input(np.copy(data[0]))
    else:
        print("[LOAD DATA] Warning the preprocessing funtion may be wrong for {}".format(config["extraction_model"]))

    return data

