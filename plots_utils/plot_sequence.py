import numpy as np
import cv2
import os
import matplotlib.cm as cm


def plot_sequence(data, rgb=False, pre_processing=None, video_name=None, path=None, lmks=None, ref_lmks=None, lmk_size=5):
    print("shape data", np.shape(data))

    # transform to bgr
    if rgb:
        data = data[..., ::-1]

    if pre_processing == 'VGG19':
        data_copy = []
        for image in data:
            # (un-)process image from VGG19 pre-processing
            img = np.array(image + 128).astype(np.uint8)
            # img = img[..., ::-1]  # rgb
            img[img > 255] = 255
            img[img < 0] = 0
            data_copy.append(img)
        data = np.array(data_copy)

    # retrieve parameters
    width = np.shape(data)[1]
    height = np.shape(data)[2]

    # set name
    if video_name is not None:
        video_name = video_name
    else:
        video_name = 'video.mp4'

    # set path
    if path is not None:
        path = path
    else:
        path = ''

    # compute padding
    lmk_padding = int(lmk_size / 2)

    # create colors
    if lmks is not None:
        colors = cm.rainbow(np.linspace(0, 1, np.shape(lmks)[1]))

    # set video recorder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(path, video_name), fourcc, 30, (width, height))

    for i, image in enumerate(data):
        if ref_lmks is not None:
            # add lmk on image
            ref_lmk_pos = np.array(ref_lmks[0]).astype(int)
            for lmk in ref_lmk_pos:
                image[lmk[1] - lmk_padding:lmk[1] + lmk_padding, lmk[0] - lmk_padding:lmk[0] + lmk_padding] = [255,
                                                                                                               0,
                                                                                                               0]
        if lmks is not None:
            # add lmk on image
            lmk_pos = np.array(lmks[i]).astype(int)
            for l, lmk in enumerate(lmk_pos):
                if ref_lmks is not None:
                    color = np.array(colors[l])[:3] * 255
                    color = color[::-1]
                    image = cv2.arrowedLine(image, (ref_lmk_pos[l, 0], ref_lmk_pos[l, 1]), (lmk[0], lmk[1]), color, 1)

                image[lmk[1] - lmk_padding:lmk[1] + lmk_padding, lmk[0] - lmk_padding:lmk[0] + lmk_padding] = [0, 255, 0]
        # write image
        video.write(image)

    cv2.destroyAllWindows()
    video.release()