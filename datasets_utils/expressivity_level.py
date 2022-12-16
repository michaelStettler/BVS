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


def get_expr_extreme_idx():
    """
    fear 03_Fear_1.0_4_lightSamples3.0050.jpeg: 50
    lipsmack 03_LipSmacking_1.0_4_lightSamples3.0073.jpeg: 193
    threat 03_Threat_1.0_4_lightSamples3.0067.jpeg: 307

    :return:
    """
    return [50, 193, 307]


def get_extreme_frame_from_expr_strength(data):
    extremes_idx = get_expr_extreme_idx()
    filtered_data = data[0][extremes_idx]
    filtered_label = data[1][extremes_idx]

    return [filtered_data, filtered_label]

