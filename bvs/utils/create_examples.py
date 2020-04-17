import numpy as np


def get_simple_vertical_bar():
    img = np.zeros((50, 50, 1))
    img[10:40, 25, :] = 255
    return img


def get_double_simple_vertical_bar():
    img = np.zeros((256, 256, 1))
    img[78:125, 128, :] = 255
    img[131:179,128, :] = 255
    return img


def get_half_vert_hori_pattern():
    img = np.zeros((250, 250, 1))
    hori_pos = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230]
    vert_pos = [135, 155, 175, 195, 215, 235]

    img[hori_pos, 15:25, :] = 255
    img[hori_pos, 35:45, :] = 255
    img[hori_pos, 55:65, :] = 255
    img[hori_pos, 75:85, :] = 255
    img[hori_pos, 95:105, :] = 255
    img[hori_pos, 115:125, :] = 255
    img[5:15, vert_pos, :] = 255
    img[25:35, vert_pos, :] = 255
    img[45:55, vert_pos, :] = 255
    img[65:75, vert_pos, :] = 255
    img[85:95, vert_pos, :] = 255
    img[105:115, vert_pos, :] = 255
    img[125:135, vert_pos, :] = 255
    img[145:155, vert_pos, :] = 255
    img[165:175, vert_pos, :] = 255
    img[185:195, vert_pos, :] = 255
    img[205:215, vert_pos, :] = 255
    img[225:235, vert_pos, :] = 255
    return img


def get_half_vert_hori_pattern_small():
    img = np.zeros((75, 75, 1))
    hori_pos = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]
    vert_pos = [40, 44, 48, 52, 56, 60]

    img[hori_pos, 14:17, :] = 255
    img[hori_pos, 19:22, :] = 255
    img[hori_pos, 24:27, :] = 255
    img[hori_pos, 29:32, :] = 255
    img[hori_pos, 34:37, :] = 255
    img[9:12, vert_pos, :] = 255
    img[14:17, vert_pos, :] = 255
    img[19:22, vert_pos, :] = 255
    img[24:27, vert_pos, :] = 255
    img[29:32, vert_pos, :] = 255
    img[34:37, vert_pos, :] = 255
    img[39:42, vert_pos, :] = 255
    img[44:47, vert_pos, :] = 255
    img[49:52, vert_pos, :] = 255
    img[54:57, vert_pos, :] = 255
    img[59:62, vert_pos, :] = 255
    return img