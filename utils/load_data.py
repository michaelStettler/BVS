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

    elif config['train_data'] == 'FEI_semantic':
        data = _load_FEI_semantic(config)

    elif config['train_data'] == 'affectnet':
        data = _load_affectnet(config, train)
    elif config['train_data'] == 'ExpressionMorphing':
        print("[WARNING] 'ExpressionMorphing' dataset should be removed in the future by 'morphing_space'")
        # todo remove _load_expression_morphing()
        data = _load_expression_morphing(config, train, sort_by)
    elif config['train_data'] == 'morphing_space':
        data = _load_morphing_space(config, train, sort_by, get_raw=get_raw)
    elif config['train_data'] == 'basic_shapes':
        data = _load_basic_shapes(config, train)
    else:
        raise ValueError("training data: '{}' does not exists! Please change norm_base_config file or add the training data"
                         .format(config['train_data']))

    return data


def _load_monkey(config, train, sort_by):
    if train:
        df = pd.read_csv(config['csv_train'])
        try:
            directory = config['train_directory']
        except KeyError:
            directory = None
    else:
        df = pd.read_csv(config['csv_val'])
        try:
            directory = config['val_directory']
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
    for index, row in tqdm(df.iterrows()):
        # load img
        if os.path.exists(os.path.join(directory, row['path'], row['image'])):
            # Michael's dataset and csv file
            im = cv2.imread(os.path.join(directory, row['path'], row['image']))
        else:
            # Tim's dataset and csv file
            im = cv2.imread(os.path.join(directory, row['image']))
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


def _load_expression_morphing(config, train, sort_by):
    if not isinstance(train, bool):         # if train is an integer
        df = pd.read_csv(config['csv{}'.format(train)])
        try:
            directory = config['directory{}'.format(train)]
        except KeyError:
            directory = None
        try:
            avatar = config['avatar{}'.format(train)]
        except KeyError:
            avatar = 'all'
        try:
            human_expression_config = config['human_expression{}'.format(train)]
        except KeyError:
            human_expression_config = 'all'
        try:
            anger_config = config['anger{}'.format(train)]
        except KeyError:
            anger_config = 'all'
    else:
        if train:
            df = pd.read_csv(config['csv_train'])
            try:
                avatar = config['train_avatar']
            except KeyError:
                avatar ='all'
            try:
                human_expression_config = config['train_human_expression']
            except KeyError:
                human_expression_config = 'all'
            try:
                anger_config = config['train_anger']
            except KeyError:
                anger_config = 'all'
        else:
            df = pd.read_csv(config['csv_val'])
            try:
                avatar = config['val_avatar']
            except KeyError:
                avatar ='all'
            try:
                human_expression_config = config['val_human_expression']
            except KeyError:
                human_expression_config = 'all'
            try:
                anger_config = config['val_anger']
            except KeyError:
                anger_config = 'all'

    if sort_by is not None:
        df = df.sort_values(by=sort_by)

    # select avatar
    if avatar == 'all':
        monkey_avatar = [False, True]
    elif avatar == 'human':
        monkey_avatar = [False]
    else:
        monkey_avatar = [True]
    df = df[df['monkey_avatar'].isin(monkey_avatar)]
    # select human/monkey expression
    if human_expression_config == 'all':
        human_expression = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        human_expression = human_expression_config
    df = df[df['human_expression'].isin(human_expression)]
    # select anger/fear blending
    if anger_config == 'all':
        anger = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        anger = anger_config
    df = df[df['anger'].isin(anger)]

    num_data = len(df.index)
    print("[DATA] Found {} images".format(num_data))


    # load each image from the csv file
    data = load_from_csv(new_df, config)

    return data


def _load_morphing_space(config, train, sort_by, get_raw=False):
    # load csv
    df = pd.read_csv(config['csv'], index_col=0)

    # sort if argument is present
    if sort_by is not None:
        df = df.sort_values(by=sort_by)

    # select avatar
    if train:
        config_avatar = config['train_avatar']
    else:
        config_avatar = config['val_avatar']

    # build filter depending on avatar
    if config_avatar == 'all_orig':
        avatar = ['Monkey_orig', 'Human_orig']
    elif config_avatar == 'human_orig':
        avatar = ['Human_orig']
    elif config_avatar == 'monkey_orig':
        avatar = ['Monkey_orig']
    else:
        raise ValueError('Avatar {} does not exists in morphing_space dataset!'.format(train))
    # apply filter
    df = df[df['avatar'].isin(avatar)]

    # read out the expressions for training
    new_df = pd.DataFrame()
    for expression in config['train_expression']:
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
        else:
            raise NotImplementedError('Expression {} is not yet implemented')

        # select anger expression
        sub_df = sub_df[sub_df['anger'].isin(anger)]
        # select species expression
        sub_df = sub_df[sub_df['h_exp'].isin(h_exp)]

        # append to new df
        new_df = new_df.append(sub_df, ignore_index=True)

    num_data = len(new_df.index)
    print("[DATA] Found {} images".format(num_data))

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
    #print(df.head)
    print("[DATA] Found {} images".format(num_data))
    #print(df.info())
    # print(df.head())

    #  declare generator
    dataGen = DataGen(config, df, directory)

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