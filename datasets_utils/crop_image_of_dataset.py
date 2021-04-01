import os
import cv2
from tqdm import tqdm


def crop(img_path, size=(720, 720), alignment='center'):
    """
    helper function to crop an image to the given size
    :param img:
    :param size: (width, height)
    :param center:
    :return:
    """
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("img is None! Path: {}".format(img_path))

    h, w, c = img.shape

    # crop height
    if h > size[1]:
        # compute difference
        diff_h = h - size[1]

        if alignment == 'center':
            half_diff = int(diff_h / 2)
            img = img[half_diff:size[1] + half_diff]
        elif alignment == 'left':
            img = img[:size[1]]
        elif alignment == 'right':
            start = h - size[1]
            img = img[start:]
        else:
            raise NotImplementedError("condition {} is not implemented".format(alignment))

    # crop width
    if w > size[0]:
        # compute difference
        diff_w = w - size[0]

        if alignment == 'center':
            half_diff = int(diff_w / 2)
            img = img[:, half_diff:size[0] + half_diff, :]
        elif alignment == 'left':
            img = img[:, size[0], :]
        elif alignment == 'right':
            start = w - size[0]
            img = img[:, start:, :]
        else:
            raise NotImplementedError("condition {} is not implemented".format(alignment))

    return img


def get_list_of_files(dirName):
    """
    helper function to get all the files within a folder and its subdirectories
    :param dirName:
    :return:
    """
    # create a list of file and sub directories 
    # names in the given directory 
    file_list = os.listdir(dirName)
    all_files = list()
    # Iterate over all the entries
    for entry in file_list:
        if 'csv' in entry or entry[0] == '.':
            print("removed:", entry)
        else:
            # Create full path
            full_path = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(full_path):
                all_files = all_files + get_list_of_files(full_path)
            else:
                if 'jpeg' or 'jpg' in full_path:
                    all_files.append(full_path)

    return all_files


def crop_folder(path, size=(720, 720), alignment='center'):
    """
    crop all images within the path given to the dimensions given and alignment
        - center: will crop around the center
        - left: will crop from the left/top border
        - right: will crop from the right/bottom border
    :param path:
    :return:
    """
    # get all images
    img_list = get_list_of_files(path)

    # replace \\ to /
    img_list = [path.replace('\\', '/') for path in img_list]

    print("Number of images found:", len(img_list))

    # crop all images
    for im_name in tqdm(img_list):
        cropped_img = crop(im_name, size, alignment)
        cv2.imwrite(im_name, cropped_img)


if __name__ == "__main__":
    """
    Script to crop all images within a folder. 
    We use it to crop the monkey avatars from 1280x720 to 720x720 so all morphing space images have the same size
    
    usage: python -m datasets_utils.crop_image_of_dataset

    """

    path = 'D:/Dataset/MorphingSpace'

    crop_folder(path)