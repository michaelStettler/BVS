import numpy as np
import os
import cv2


def load_sequence(list, path):
    sequence = []
    for file_name in list:
        im = cv2.imread(os.path.join(path, file_name))
        sequence.append(im)

    return np.array(sequence)


def filter_sequence(sequence, kernel_size=3, stride=1):
    # compute kernel padding
    ker_padding = kernel_size // 2

    # copy sequence
    filt_seq = np.copy(sequence)

    # loop over each image
    for i in range(ker_padding, len(sequence) - ker_padding, stride):
        # get the kernel sequence
        ker_seq = sequence[(i-ker_padding):(i + ker_padding + 1)]

        # get mean of the kernel sequence
        ker_mean = np.mean(ker_seq, axis=0)

        # set new seq with mean
        filt_seq[i] = ker_mean

    return filt_seq


def save_sequence(sequence, path, list_file):
    # loop over each image
    for i, image in enumerate(sequence):
        cv2.imwrite(os.path.join(path, list_file[i]), image)


if __name__ == "__main__":
    """
    This script removes some noise created by maya rendering
    It filters for each pixel their values trough the sequence length
    
    run: python -m datasets_utils.filter_rendering_noise
    """

    # # -----------------------------------------------------------------------------------------------------------------
    # # single folder
    # # path = 'D:/Dataset/MorphingSpace/human_orig_filt3/HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0'
    # # path = 'D:/Dataset/MorphingSpace/human_orig_filt3/HumanAvatar_Anger_1.0_Fear_0.0_Monkey_1.0_Human_0.0'
    # path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/human_orig/HumanAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0'
    # path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/monkey_orig/MonkeyAvatar_Anger_0.0_Fear_1.0_Monkey_0.0_Human_1.0'
    #
    # list_file = sorted(os.listdir(path))
    # print("length list_file", len(list_file))
    #
    # sequence = load_sequence(list_file, path)
    # print("shape sequence", np.shape(sequence))
    #
    # filt_seq = filter_sequence(sequence, kernel_size=3)
    # print("shape filt_seq", np.shape(filt_seq))
    #
    # save_sequence(filt_seq, path, list_file)

    # -----------------------------------------------------------------------------------------------------------------
    # complete directory

    # # Human orig
    # path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/human_orig_raw'
    # new_path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/human_orig_filt3'
    # monkey orig
    path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/monkey_orig_raw'
    new_path = '/Users/michaelstettler/PycharmProjects/BVS/data/MorphingSpace/monkey_orig_filt3'

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    # get all sub directories
    list_dir = os.listdir(path)

    for directory in list_dir:
        if directory[0] != ".":
            print("directory", directory)
            list_file = sorted(os.listdir(os.path.join(path, directory)))
            print("length list_file", len(list_file))

            sequence = load_sequence(list_file, os.path.join(path, directory))
            print("Sequence loaded:", np.shape(sequence))

            filt_seq = filter_sequence(sequence, kernel_size=3)
            print("Sequence filtered:", np.shape(filt_seq))

            # create new directory
            if not os.path.exists(os.path.join(new_path, directory)):
                os.mkdir(os.path.join(new_path, directory))

            save_sequence(filt_seq, os.path.join(new_path, directory), list_file)
            print("Sequence saved!")
            print()
