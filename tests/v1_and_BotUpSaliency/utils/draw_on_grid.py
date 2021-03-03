import numpy as np


def draw_on_grid(img):
    """
    draw_on_grid will replicate what would be the input image look like from the gabor filter activations (input).
    It will draw a line corresponding to each orientation into a small box for each x, y coordinate of the input image.

    This function is not intended to be use! Very inneficient!
    It is just a small function to help visualize the output of the saliency map.

    :param img:
    :return:
    """
    # define variable
    box_size = 7  #px
    gap = 2  #px

    # get input size
    img_size = np.shape(img)
    print("[draw on grid] img_size", img_size)

    # normalize img
    img -= np.min(img)
    if np.max(img != 0):
        img /= np.max(img)

    # define image grid
    grid_img = np.zeros((img_size[0]*(box_size+gap) + gap, img_size[1]*(box_size+gap) + gap))

    for y in range(img_size[0]):
        y_gap = gap + y * (box_size + gap)
        for x in range(img_size[1]):
            x_gap = gap + x * (box_size + gap)

            # get values
            pix_value = np.max(img[y, x, :])
            n = np.argmax(img[y, x, :])
            if pix_value > 0:
                line = get_line(n, pix_value)
                grid_img[y_gap:y_gap+box_size, x_gap:x_gap+box_size] = line

    grid_img = np.expand_dims(grid_img, axis=2)
    return grid_img.astype(np.uint8)


def get_line(n, value):
    if n == 0:
        box = get_0_line(value)
    elif n == 3:
       box = get_45_line(value)
    elif n == 6:
       box = get_90_line(value)
    elif n == 9:
       box = get_125_line(value)
    else:
       raise NotImplementedError("Need to implement in draw_on_grid for other angles!")

    return box * 255


def get_0_line(x):
    return np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [x, x, x, x, x, x, x],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]])


def get_45_line(x):
    return np.array([[0, 0, 0, 0, 0, 0, x],
                     [0, 0, 0, 0, 0, x, 0],
                     [0, 0, 0, 0, x, 0, 0],
                     [0, 0, 0, x, 0, 0, 0],
                     [0, 0, x, 0, 0, 0, 0],
                     [0, x, 0, 0, 0, 0, 0],
                     [x, 0, 0, 0, 0, 0, 0]])


def get_90_line(x):
    return np.array([[0, 0, 0, x, 0, 0, 0],
                     [0, 0, 0, x, 0, 0, 0],
                     [0, 0, 0, x, 0, 0, 0],
                     [0, 0, 0, x, 0, 0, 0],
                     [0, 0, 0, x, 0, 0, 0],
                     [0, 0, 0, x, 0, 0, 0],
                     [0, 0, 0, x, 0, 0, 0]])


def get_125_line(x):
    return np.array([[x, 0, 0, 0, 0, 0, 0],
                     [0, x, 0, 0, 0, 0, 0],
                     [0, 0, x, 0, 0, 0, 0],
                     [0, 0, 0, x, 0, 0, 0],
                     [0, 0, 0, 0, x, 0, 0],
                     [0, 0, 0, 0, 0, x, 0],
                     [0, 0, 0, 0, 0, 0, x]])
