import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
viridis = cm.get_cmap('viridis', 12)


def plot_space(positions, labels, n_cat, ref_vector=[0, 0], tun_vectors=None, min_axis=15, max_axis=15,
               shifts=None, show=False, ref_pos=None, ref_labels=None, alpha=1):

    marker_vectors = ['o', '^', 's', 'p', '*', 'D', 'v', '<', '>']

    # retrieve variables
    n_pts = positions.shape[0]
    n_feat_map = positions.shape[1]
    n_ref = 1
    if shifts is not None:
        n_ref = shifts.shape[0]

    # set img size
    n_rows = int(np.sqrt(n_feat_map))
    n_columns = np.ceil(n_feat_map / n_rows).astype(int)
    image = np.zeros((n_rows*400, n_columns*400, 3))

    # plot
    cmap = cm.get_cmap('viridis', n_cat)
    for ft in range(n_feat_map):
        fig = plt.figure(figsize=(4, 4), dpi=100)  # each subplot is 400x400

        for r in range(n_ref):
            # shifts positions
            ref_idx = np.arange(n_pts)  # construct all index for common pipeline
            if shifts is not None:
                ref_idx = ref_idx[labels[:, 1] == r]
                pos_ft = positions[ref_idx, ft] - shifts[r, ft]
            else:
                pos_ft = positions[:, ft]

            # get unique labels
            if ref_labels is not None:
                uniques = np.unique(ref_labels)
            else:
                uniques = np.unique(labels)

            for i, color in zip(range(n_cat), cmap(range(n_cat))):
                label = uniques[i]

                label_name = None
                if r == 0:
                    label_name = "cat {}".format(label)
                    if label == 0:
                        label_name = "reference"

                # get all positions for this label
                pos = pos_ft[labels[ref_idx, 0] == label]

                # print only if there's some data of the labels
                if len(pos) > 0:
                    # scale tuning vector for plotting
                    pos_norm = np.linalg.norm(pos, axis=1)
                    max_length = np.max(pos_norm)
                    x_direction = pos[np.argmax(pos_norm)][0]

                    sign = np.sign(x_direction * tun_vectors[i, ft, 0])

                    # plot the positions
                    plt.scatter(pos[:, 0], pos[:, 1],
                                color=color,
                                label=label_name,
                                marker=marker_vectors[r],
                                alpha=alpha)

                    # plot tuning line
                    if tun_vectors is not None and ref_vector is not None:
                        # x_ = [ref_vector[ft, 0], sign * max_length * tun_vectors[i, 0]]
                        x_ = [0, tun_vectors[i, ft, 0]]
                        # y_ = [ref_vector[ft, 1], sign * max_length * tun_vectors[i, 1]]
                        y_ = [0, tun_vectors[i, ft, 1]]
                        plt.plot(x_, y_, color=color)

                if ref_pos is not None:
                    r_pos = pos_ft[ref_labels[ref_idx, 0] == label]
                    plt.scatter(r_pos[:, 0], r_pos[:, 1], color=color, label=label_name)


        # plot cross
        plt.plot(0, 0, 'xk')
        # plt.plot(ref_vector[ft, 1], ref_vector[ft, 0], 'xk')

        if ft == 0:
            plt.legend()
        plt.axis([-min_axis, max_axis, -min_axis, max_axis])
        # plt.axis([-10, 10, -10, 10])

        if show:
            plt.show()

        # transform figure to numpy array
        fig.canvas.draw()
        fig.tight_layout(pad=0)

        # compute row/col number
        n_col = ft % n_columns
        n_row = ft // n_columns

        # transform figure to numpy and append it in the correct place
        figure = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image[n_row*400:(n_row+1)*400, n_col*400:(n_col+1)*400] = figure

        # clear figure
        plt.cla()
        plt.clf()
        plt.close()

    return image.astype(np.uint8)