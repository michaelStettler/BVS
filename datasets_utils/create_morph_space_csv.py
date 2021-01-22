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
        elif parts[1] == path:  # sentinel for relative paths
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
    # retrieve neutral parameter
    n_exp = 0.0
    if n_exp_idx is not None:
        n_exp = float(params[n_exp_idx])
    # retrieve other parameters
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
    else:
        # add simply the "orig" sord to remove confusion between the condition and the avatar
        avatar = avatar + '_orig'

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
        try:
            dist = np.sqrt(np.sum(np.power(img_pos[frame] - img_pos[0], 2)))
        except:
            dist = 0
            # print("Frame {} out of bounds".format(frame))
    else:
        print("[WARNING] No image position! Keep category: {}".format(c))

    # set the category to neutral if the mesh displacement is smaller than the threshold
    if dist < threshold:
        c = 0

    return c


def get_image_params(img_name, img_pos, threshold=50):
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
    cat = get_img_category(anger, fear, m_exp, h_exp, frame, img_pos, threshold=threshold)

    return cat, avatar, anger, fear, m_exp, h_exp, n_exp


def create_morphe_space_csv(img_path, dist_file_list, m_threshold=50, h_threshold=8):
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
    print("Adding images to dataframe:")
    for image_path in tqdm(image_list):
        splits = splitall(image_path)
        relat_img_path = os.path.join(*splits[img_path_length:-1])  # add * 'splat' to take a list as argument
        relat_img_path = relat_img_path.replace('\\', '/')
        image_name = splits[-1]

        # get information from directory
        img_pos = None
        img_cond_dir = relat_img_path.split('/')[-2]
        img_last_dir = relat_img_path.split('/')[-1]

        pos_found = False
        for file in dist_file_list:
            if img_last_dir in file and img_cond_dir in file:
                img_pos = np.load(file)
                pos_found = True
        if not pos_found:
            print("[WARNING] Position for {} not found".format(image_path))

        # select threshold
        if 'HumanAvatar' in image_name:
            threshold = h_threshold
        else:
            threshold = m_threshold

        # get params from the image
        cat, avatar, anger, fear, m_exp, h_exp, n_exp = get_image_params(image_name, img_pos, threshold=threshold)

        # add 30 deg condition from the folder name
        if '30deg' in relat_img_path:
            avatar = avatar + '_30deg'

        # add into dataframe
        df = df.append({'image_path': relat_img_path, 'image_name': image_name, 'category': cat, 'avatar': avatar,
                        'anger': anger, 'fear': fear, 'm_exp': m_exp, 'h_exp': h_exp, 'n_exp': n_exp},
                       ignore_index=True)

    return df


def control_threshold(dist_file_list, m_threshold, h_threshold):
    count = 0
    list_small = {"file": [], "value": []}
    smaller = 1e6
    print("Control threshold in file:")
    for file in tqdm(dist_file_list):
        pos = np.load(file)
        dist = np.sqrt(np.sum(np.power(pos[:] - pos[0], 2), axis=(1, 2, 3)))

        if 'HumanAvatar' in file:
            threshold = h_threshold
        else:
            threshold = m_threshold

        max_val = np.amax(dist)

        if max_val < threshold:
            count += 1
            list_small["file"].append(file)
            list_small["value"].append(max_val)

        if max_val < smaller:
            smaller = max_val

    print("Found {} sequence that does not reach the threshold value (monkey:{}, human:{})"
          .format(count, m_threshold, h_threshold))
    if count > 0:
        print("Smaller value found is {}".format(smaller))
        print("File that does not reach the threshold value:")
        for i in range(count):
            print("File: {}, value {}".format(list_small['file'][i], list_small['value'][i]))
    print()


if __name__ == "__main__":
    """
    The script will go over all the images contain within the morphing space dataset and sort them by category according
    to the mesh displacement and the image name as follow:
        c0 = Neutral expression -> mesh displacement smaller than the threshold
        c1 = Human Angry
        c2 = Human Fear
        c3 = Monkey Angry
        c4 = Monkey Fear
    
    The csv have the following output:
    image_path, image_name, category, avatar, anger, fear, monkey_expression (m_exp), human_expression (h_exp), 
    neutral_expression (n_exp)
        
    run: python -m datasets_utils.create_morph_space_csv  
    """

    m_threshold = 50
    h_threshold = 8
    # windows
    # dist_path = ['D:/Maya projects/DigitalLuise/data/vtx_distance/human_orig',
    #              'D:/Maya projects/DigitalLuise/data/vtx_distance/human_30deg',
    #              'D:/Maya projects/DigitalLuise/data/vtx_distance/human_equ',
    #              'D:/Maya projects/DigitalLuise/data/vtx_distance/human_equ_30deg',
    #              'D:/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_orig',
    #              'D:/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_30deg',
    #              'D:/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_equ',
    #              'D:/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_equ_30deg']
    # img_path = 'D:/Dataset/MorphingSpace'

    # docker
    dist_path = ['/app/Data/Maya projects/DigitalLuise/data/vtx_distance/human_orig',
                 '/app/Data/Maya projects/DigitalLuise/data/vtx_distance/human_30deg',
                 '/app/Data/Maya projects/DigitalLuise/data/vtx_distance/human_equ',
                 '/app/Data/Maya projects/DigitalLuise/data/vtx_distance/human_equ_30deg',
                 '/app/Data/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_orig',
                 '/app/Data/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_30deg',
                 '/app/Data/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_equ',
                 '/app/Data/Maya projects/MonkeyHead_MayaProject/data/vtx_distance/monkey_equ_30deg']
    img_path = '/app/Data/Dataset/MorphingSpace'

    dist_file_list = []
    for path in dist_path:
        dist_file_list.append(glob.glob(path + '/*.npy'))
    # flatten the list
    dist_file_list = [item for sublist in dist_file_list for item in sublist]

    # control if threshold will yield to a at least one frame in all sequence
    control_threshold(dist_file_list, m_threshold, h_threshold)

    # create morph space csv dataset
    df = create_morphe_space_csv(img_path, dist_file_list, m_threshold=m_threshold, h_threshold=h_threshold)
    print(df.head())

    # save csv
    df.to_csv(os.path.join(img_path, 'morphing_space.csv'))
