import numpy as np
import cv2
import os


def plot_sequence(data, video_name=None, path=None):
    print("shape data", np.shape(data))

    # transform to bgr
    data = data[..., ::-1]

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

    # set video recorder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(path, video_name), fourcc, 30, (width, height))

    for image in data:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()