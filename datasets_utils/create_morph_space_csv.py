import os, sys
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def splitall(path):
    """
    helper function to split all the folders within a path

    used to remove the data folder and create a csv that is relative to where the csv file is saved
    :param path:
    :return:
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_params_from_img_name(params):
    """
    Use the split from the image name to gather all the information within the image_name

    The function retrieves the following info:
        - avatar (only human/monkey here, for equilibrated and 30deg, this is added after using the folder name)
        - neutral expression
        - angry
        - fear
        - monkey expression
        - human expression
        - frame

    :param params: list of all the splitted string from the image name
    :return:
    """
    # define idx in function of equilibrating 0(Neutral is added) or not
    is_equi = False
    if 'Neutral' in params[1]:
        # is equilibrated
        is_equi = True
        n_exp_idx = 2
        anger_idx = 4
        fear_idx = 6
        m_exp_idx = 8
    else:
        # normal condition
        n_exp_idx = None
        anger_idx = 2
        fear_idx = 4
        m_exp_idx = 6

    # get parameters from image name
    n_exp = 0.0
    if n_exp_idx is not None:
        n_exp = float(params[n_exp_idx])
    anger = float(params[anger_idx])
    fear = float(params[fear_idx])
    m_exp = float(params[m_exp_idx])
    h_exp = 1.0 - m_exp

    # get avatar
    avatar = 'Monkey'
    if 'Human' in params[0]:
        avatar = 'Human'

    # add equilibrated state if Neutral is present
    if is_equi:
        avatar = avatar + '_equi'

    # get image frame
    frame = int(params[-1].split('.')[-2])

    return avatar, n_exp, anger, fear, m_exp, h_exp, frame


def get_img_category(anger, fear, m_exp, h_exp, frame, img_pos, threshold=50):
    """
    Get the category of the image

    It uses the mesh deformation to set the category to neutral (0) when the positions are given

    0 - Neutral
    1 - Human Angry
    2 - Human Fear
    3 - Monkey Angry
    4 - Monkey Fear

    :param anger:
    :param fear:
    :param m_exp:
    :param h_exp:
    :param frame:
    :param img_pos:
    :param threshold:
    :return:
    """
    # categorize image to the four main categories
    c = 0
    if anger >= fear and h_exp >= m_exp:
        c = 1
    elif anger < fear and h_exp >= m_exp:
        c = 2
    elif anger >= fear and h_exp < m_exp:
        c = 3
    elif anger < fear and h_exp < m_exp:
        c = 4
    else:
        raise ValueError("Condition issues with: anger {}, fear {}, monk_exp {},  hum_exp {}"
                         .format(anger, fear, m_exp, h_exp))

    # set image to neutral if below deformation threshold
    dist = threshold  # set so to keep the former category if no distance file is present
    if img_pos is not None:
        # compute mesh deformation of the current frame
        dist = np.sqrt(np.sum(np.power(img_pos[frame] - img_pos[0], 2)))
    else:
        print("[WARNING] No image position found! Keep category: {}".format(c))

    if dist < threshold:
        c = 0

    return c


def get_image_params(img_name, img_pos):
    """
    Function to get all the parameters from one image
    It calls the get_params_from_img_name and get_img_category functions

    Parameters are:
        - category
        - avatar
        - neutral expression
        - angry
        - fear
        - monkey expression
        - human expression
        - frame

    :param img_name:
    :param img_pos:
    :return:
    """
    params = img_name.split('_')

    # get params from image name
    avatar, n_exp, anger, fear, m_exp, h_exp, frame = get_params_from_img_name(params)

    # get category of the image
    cat = get_img_category(anger, fear, m_exp, h_exp, frame, img_pos)

    return cat, avatar, anger, fear, m_exp, h_exp, n_exp


def create_morphe_space_csv(img_path, dist_file_list):
    """
    Create a dataframe with the columns:
    'image_path', 'image_name', 'category', 'avatar', 'anger', 'fear', 'm_exp', 'h_exp', 'n_exp'

    The function will add one entry to each image it founds within the "img_path" argument.
    The function coorelates each image to the corresponding mesh deformation from the list given within the
    "dist_file_list" arguement

    :param img_path:
    :param dist_file_list:
    :return:
    """
    columns_name = ['image_path', 'image_name', 'category', 'avatar', 'anger', 'fear', 'm_exp', 'h_exp', 'n_exp']
    df = pd.DataFrame(columns=columns_name)

    # find each images
    image_list = list()
    for (dirpath, dirnames, filenames) in os.walk(img_path):
        image_list += [os.path.join(dirpath, file) for file in filenames if 'csv' not in file]
    print("found {} images".format(len(image_list)))

    # get the length of the path before the csv root file to enable the image path to start from there
    img_path_length = len(img_path.split('/'))

    # add each image into the dataframe
    for image_path in tqdm(image_list):
        splits = splitall(image_path)
        relat_img_path = os.path.join(*splits[img_path_length:-1])  # add * 'splat' to take a list as argument
        relat_img_path = relat_img_path.replace('\\', '/')
        image_name = splits[-1]

        # get deformation file
        img_pos = None
        img_cond_dir = relat_img_path.split('/')[-2]
        img_last_dir = relat_img_path.split('/')[-1]
        for file in dist_file_list:
            if img_last_dir in file and img_cond_dir in file:
                img_pos = np.load(file)

        # get params from the image
        cat, avatar, anger, fear, m_exp, h_exp, n_exp = get_image_params(image_name, img_pos)

        # add 30 deg condition from the folder name
        if '30deg' in relat_img_path:
            avatar = avatar + '_30deg'

        # add into dataframe
        df = df.append({'image_path': relat_img_path, 'image_name': image_name, 'category': cat, 'avatar': avatar,
                        'anger': anger, 'fear': fear, 'm_exp': m_exp, 'h_exp': h_exp, 'n_exp': n_exp},
                       ignore_index=True)

    return df


if __name__ == "__main__":
    """
    The script will go over all the images contain within the morphing space and sort the category according
    to the mesh displacement
    
    run: python -m datasets_utils.create_morph_space_csv  
    """
    dist_path = 'D:/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_orig'
    img_path = 'D:/Dataset/MorphingSpace'

    # todo add folders from other morphing space
    dist_file_list = glob.glob(dist_path + '/*.npy')

    # create morphe space
    df = create_morphe_space_csv(img_path, dist_file_list)
    print(df.head())

    # save csv
    df.to_csv(os.path.join(img_path, 'morphing_space.csv'))
