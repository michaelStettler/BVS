import cv2
import numpy as np


def resize_image(im, dim=(224, 224)):
    # resize image
    height, width, depths = im.shape
    if height > width:
        ratio = height / dim[0]
        dim = (int(width / ratio), dim[0])
    else:
        ratio = width / dim[1]
        dim = (dim[1], int(height / ratio))
    im = cv2.resize(im, dim)

    return im


def pad_image(im, dim=(224, 224, 3)):
    # pad image
    height, width, depths = im.shape
    img = np.zeros(dim)
    if height > width:
        pad = int((dim[1] - width) / 2)
        img[:, pad:pad + width, :] = im
    else:
        pad = int((dim[0] - height) / 2)
        img[pad:pad + height, :, :] = im

    return img