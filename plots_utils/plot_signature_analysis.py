import os
import numpy as np
import cv2
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_tuning_plot(ref_vectors, tun_vectors, lmks, colors, proj=None, lmk_proj=None, fig_title=None, fig_size=(2, 2), dpi=200):
    # print("shape ref_vectors", np.shape(ref_vectors))
    # print("shape projections", np.shape(projections))
    # print("shape tun_vectors", np.shape(tun_vectors))
    # create figure
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    # center ref_vectors
    centers = np.mean(ref_vectors, axis=0)
    ref_vect = ref_vectors - centers
    lmks_vect = lmks - centers

    for i, ref_v, tun_v, lmk, color in zip(np.arange(len(ref_vectors)), ref_vect, tun_vectors, lmks_vect, colors):
        plt.scatter(ref_v[0], -ref_v[1], color=color, facecolors='none')
        plt.arrow(ref_v[0], -ref_v[1], tun_v[0], -tun_v[1], color=color, linewidth=1, linestyle=':')  # reference

        plt.scatter(lmk[0], -lmk[1], color=color, marker='x')

        if lmk_proj is not None:
            plt.text(lmk[0] + .5, -lmk[1] + .5, "{:.2f}".format(lmk_proj[i]), fontsize="xx-small")

    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.axis('off')

    if proj is not None:
        plt.text(-5, 18, "{:.2f}".format(proj))  # text in data coordinates

    if fig_title is not None:
        plt.title(fig_title)

    # transform figure to numpy array
    fig.canvas.draw()
    fig.tight_layout(pad=0)

    figure = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # clear figure
    plt.cla()
    plt.clf()
    plt.close()

    return figure


def get_image_sequence(image, lmks, ref_lmks, colors, lmk_size=3):
    lmks = lmks.astype(int)
    ref_lmks = ref_lmks.astype(int)

    # compute padding
    lmk_pad = int(lmk_size / 2)

    # construct image of the sequence
    # add lmk on image
    for l, lmk, r_lmk, color in zip(range(len(lmks)), lmks, ref_lmks, colors):

        # add ref lmk on image
        image[r_lmk[1] - lmk_pad:r_lmk[1] + lmk_pad, r_lmk[0] - lmk_pad:r_lmk[0] + lmk_pad] = [255, 0, 0]

        # add arrows
        color = np.array(color)[:3] * 255
        image = cv2.arrowedLine(image, (r_lmk[0], r_lmk[1]), (lmk[0], lmk[1]), color, 1)

        # add lmks
        image[lmk[1] - lmk_pad:lmk[1] + lmk_pad, lmk[0] - lmk_pad:lmk[0] + lmk_pad] = [0, 255, 0]

    return image


def get_graph(data, frame, fig_size=(4, 1), dpi=200):
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    plt.plot([frame, frame], [0, 10], 'r-')
    plt.plot(data)

    fig.canvas.draw()
    # fig.tight_layout(pad=0.5)

    figure = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # clear figure
    plt.cla()
    plt.clf()
    plt.close()

    return figure


def plot_signature_proj_analysis(data, lmks, ref_vectors, tun_vectors, proj, config, lmk_proj=None,
                                 pre_processing=None, video_name=None, path=None, lmk_size=3, img_seq_ratio=4):
    matplotlib.use('agg')

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
    img_seq_height = 1000
    img_seq_width = 1600
    img_seq = np.zeros((img_seq_height, img_seq_width, 3)).astype(np.uint8)

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

    # create colors
    colors = cm.rainbow(np.linspace(0, 1, np.shape(lmks)[1]))

    # set video recorder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(path, video_name), fourcc, 30, (img_seq_width, img_seq_height))

    # set conditions
    conditions = config["condition_names"]
    if len(conditions) !=4:
        raise NotImplementedError("Function is not working for other than 4 conditions yet!")
    x_pos = [0, 0, 1, 1]
    y_pos = [0, 1, 0, 1]

    # construct movie
    for i, image in tqdm.tqdm(enumerate(data)):
        img = get_image_sequence(image, lmks[i]*img_seq_ratio, ref_vectors[0]*img_seq_ratio, colors,
                                 lmk_size=lmk_size)
        img = cv2.resize(img, (800, 800))
        img_text = "Frame {}, Seq. {}".format(i, int(i/150))
        img = cv2.putText(img, img_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        img_seq[100:900, :800] = img

        # construct signatures
        for c, cond in enumerate(conditions):
            lmk_p_ = None
            if lmk_proj is not None:
                lmk_p_ = lmk_proj[i, c + 1]

            fig = get_tuning_plot(ref_vectors[0], tun_vectors[c + 1] * 4, lmks[i], colors,
                                  proj=proj[i, c + 1],
                                  lmk_proj=lmk_p_,
                                  fig_title=cond,
                                  fig_size=(2, 2),
                                  dpi=200)  # construct fig of 400x400
            img_seq[(x_pos[c]*400):(x_pos[c]*400+400), (y_pos[c]*400+800):(y_pos[c]*400+1200)] = fig

        # add graphics
        graph = get_graph(proj, i)
        img_seq[800:, 800:] = graph

        # write image
        video.write(img_seq)

    cv2.destroyAllWindows()
    video.release()

    matplotlib.use('macosx')