import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_ft_map_pos(pos, fig_name=None, path=None, titles=None, color_seq=None, arrows=None, arrows_color=None):

    # create color sequence
    if color_seq is None:
        color_seq = np.arange(len(pos))
    else:
        color_seq = color_seq

    # retrieve parameters
    num_subplot = np.shape(pos)[1]
    n_column = int(np.sqrt(num_subplot))
    n_rows = np.ceil(num_subplot / n_column).astype(np.int)

    # set video name
    fig_name = _set_fig_name(fig_name, 'feature_map_positions.png')

    # set save folder
    path = _set_save_folder(path, '')

    # create figure
    plt.figure(figsize=(n_column*4, n_rows*4))

    for i in range(num_subplot):
        plt.subplot(n_rows, n_column, i + 1, aspect='equal')

        plt.scatter(pos[:, i, 1], pos[:, i, 0], c=color_seq)
        # plt.xlim(13.5, 14.0)
        # plt.ylim(11.7, 12.2)
        plt.colorbar()

        # set axis dimension to plot in a square so the angle are correct
        xlim = plt.xlim()
        ylim = plt.ylim()
        xlim_diff = xlim[1] - xlim[0]
        ylim_diff = ylim[1] - ylim[0]

        # set smaller axis to equal the bigger one
        if ylim_diff > xlim_diff:
            diff = ylim_diff
            mid = xlim[0] + xlim_diff / 2
            plt.xlim(mid - diff / 2, mid + diff / 2)
        elif ylim_diff < xlim_diff:
            diff = xlim_diff
            mid = ylim[0] + ylim_diff / 2
            plt.ylim(mid - diff / 2, mid + diff / 2)

        if arrows is not None:
            arrows_tail = arrows[0]
            arrows_head = arrows[1]
            for j in range(len(arrows_tail)):
                arr_t = arrows_tail[j, i]
                arr_h = arrows_head[j, i]
                if arrows_color is not None:
                    plt.arrow(arr_t[1], arr_t[0], arr_h[1], arr_h[0], color=arrows_color[j], linewidth=2)
                else:
                    plt.arrow(arr_t[1], arr_t[0], arr_h[1], arr_h[0], linewidth=2)

        # add sub title if given
        if titles is not None:
            plt.title(titles[i])

    plt.savefig(os.path.join(path, fig_name))


def plot_ft_pos_on_sequence(pos, seq, vid_name=None, save_folder=None, ft_size=(28, 28), pre_proc='VGG',
                            lmk_size=1):
    # apply pre_processing
    seq = _pre_proc(seq, pre_proc)

    # set video name
    vid_name = _set_fig_name(vid_name, 'feature_map_positions_on_sequence.mp4')

    # set save folder
    save_folder = _set_save_folder(save_folder, '')

    # get number of landmarks
    num_lmk = np.shape(pos)[1]

    # reshape pos for xy-coordinates for each landmarks
    if len(np.shape(pos)) < 3:
        pos = np.reshape(pos, (len(pos), -1, 2))

    # retrieve width and height
    width = np.shape(seq)[1]
    height = np.shape(seq)[2]

    # compute ratio
    x_ratio = width / ft_size[0]
    y_ratio = height / ft_size[1]

    # set padding for the landmark
    lmk_padding = int(lmk_size)

    # set video recorder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(save_folder, vid_name), fourcc, 30, (width, height))

    # add each image to the video
    for i, image in enumerate(seq):
        img = np.copy(image)

        # plot each landmarks
        for k in range(num_lmk):
            # draw initial landmarks
            x_init = int(np.round(pos[0, k, 0] * x_ratio))  # horizontal
            y_init = int(np.round(pos[0, k, 1] * y_ratio))  # vertical
            img[(x_init-lmk_padding):(x_init+lmk_padding), (y_init-lmk_padding):(y_init+lmk_padding)] = [0, 255, 0]

            # draw current landmarks
            x_ = int(np.round(pos[i, k, 0] * x_ratio))  # horizontal
            y_ = int(np.round(pos[i, k, 1] * y_ratio))  # vertical
            img[(x_-lmk_padding):(x_+lmk_padding), (y_-lmk_padding):(y_+lmk_padding)] = [0, 0, 255]
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def plot_pos_on_images(pos, images, fig_name=None, save_folder=None, ref_pos=None, ft_size=(28, 28), pre_proc='VGG',
                       lmk_size=1):
    # apply pre_processing
    images = _pre_proc(images, pre_proc)

    # set images name
    images_name = _set_fig_name(fig_name, 'pos_on_images.png')

    # set save folder
    save_folder = _set_save_folder(save_folder, '')

    # get number of landmarks
    num_lmk = np.shape(pos)[1]

    # set padding for the landmark
    lmk_padding = int(lmk_size)

    # retrieve width and height
    width = np.shape(images)[1]
    height = np.shape(images)[2]

    # compute ratio
    x_ratio = width / ft_size[0]
    y_ratio = height / ft_size[1]

    # add each image to the video
    for i, image in enumerate(images):
        img = np.copy(image)

        # plot each landmarks
        for k in range(num_lmk):
            if ref_pos is not None:
                # draw initial landmarks
                x_init = int(np.round(ref_pos[i, k, 0] * x_ratio))  # horizontal
                y_init = int(np.round(ref_pos[i, k, 1] * y_ratio))  # vertical
                img[(x_init - lmk_padding):(x_init + lmk_padding), (y_init - lmk_padding):(y_init + lmk_padding)] = \
                    [0, 255, 0]

            # draw current landmarks
            x_ = int(np.round(pos[i, k, 0] * x_ratio))  # horizontal
            y_ = int(np.round(pos[i, k, 1] * y_ratio))  # vertical
            img[(x_ - lmk_padding):(x_ + lmk_padding), (y_ - lmk_padding):(y_ + lmk_padding)] = [0, 0, 255]

        # get image name
        image_name_list = images_name.split('.')
        # save image
        im_name = os.path.join(save_folder, "{}_{}.{}".format(image_name_list[0], i, image_name_list[1]))
        cv2.imwrite(im_name, img)


def _pre_proc(seq, pre_proc):
    seq = np.copy(seq)

    # modify seq for cv2
    if 'VGG' in pre_proc:
        seq += 255/2
    elif 'raw' in pre_proc:
        seq = seq
    elif 'rgb':
        seq = seq[..., ::-1]
    else:
        raise NotImplementedError

    # clip values
    seq[seq > 255] = 255
    seq[seq < 0] = 0

    # ensure that encoding is in uint8 for cv2
    seq = np.array(seq).astype(np.uint8)

    # check if img is black and white
    if len(np.shape(seq[0])) < 3:
        seq = np.expand_dims(seq, axis=3)
    if np.shape(seq)[-1] == 1:
        seq = np.repeat(seq, 3, axis=3)

    return seq


def _set_fig_name(fig_name, def_fig_name):
    if fig_name is not None:
        fig_name = fig_name
    else:
        fig_name = def_fig_name

    return fig_name


def _set_save_folder(path, def_path):
    if path is not None:
        path = path
    else:
        path = def_path

    return path
