import numpy as np


def segment_sequence(data, seq_start_idx, seq_length):
    """
    helper function to segment sequence based on the first index
    :param data:
    :param seq_start_idx:
    :param seq_length:
    :return:
    """
    # segment sequence based on config
    seg_data = []
    if seq_start_idx is not None:
        for start in seq_start_idx:
            seg_data.append(data[start:start + seq_length])
    seg_data = np.array(seg_data)

    # reshape array to get back to previous dimension
    if len(np.shape(seg_data)) == 2:
        seg_data = np.reshape(seg_data, (-1))
    elif len(np.shape(seg_data)) == 3:
        seg_data = np.reshape(seg_data, (-1, seg_data.shape[2]))
    elif len(np.shape(seg_data)) == 4:
        seg_data = np.reshape(seg_data, (-1, seg_data.shape[2], seg_data.shape[3]))
    elif len(np.shape(seg_data)) == 5:
        seg_data = np.reshape(seg_data, (-1, seg_data.shape[2], seg_data.shape[3], seg_data.shape[4]))
    else:
        raise NotImplementedError("Reshape for array dimensions of size {} not implement yet!".format(len(np.shape(seg_data))))

    return seg_data




def remove_neutral_frames(data, neutral_idx):
    print("todo remove neutral frames")
