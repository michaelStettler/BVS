import numpy as np
import matplotlib.pyplot as plt


def display_image(image, lmks=None, ref_lmks=None, lmk_size=5, pre_processing=None, is_black_n_white=False, figure=None, title=None):
    img = np.copy(image)

    if pre_processing == 'VGG19':
        # (un-)process image from VGG19 pre-processing
        img = np.array(img + 128) / 256
        img = img[..., ::-1]  # rgb
        img[img > 1] = 1.0

    # compute padding
    lmk_padding = int(lmk_size/2)

    if ref_lmks is not None:
        # add lmk on image
        ref_lmks = np.array(ref_lmks).astype(int)
        for r_lmk in ref_lmks:
            img[r_lmk[1]-lmk_padding:r_lmk[1]+lmk_padding, r_lmk[0]-lmk_padding:r_lmk[0]+lmk_padding] = [0, 1, 0]

    if lmks is not None:
        # add lmk on image
        lmks = np.array(lmks).astype(int)
        for lmk in lmks:
            img[lmk[1]-lmk_padding:lmk[1]+lmk_padding, lmk[0]-lmk_padding:lmk[0]+lmk_padding] = [0, 1, 1]

    if figure is None:
        plt.figure()

        if is_black_n_white:
            plt.imshow(img, cmap='Greys', vmin=0., vmax=255.)
        else:
            plt.imshow(img)

        if title is not None:
            plt.title(title)
    else:
        if is_black_n_white:
            figure.imshow(img, cmap='Greys', vmin=0., vmax=255.)
        else:
            figure.imshow(img)

        if title is not None:
            figure.set_title(title)


def display_images(images, lmks=None, ref_lmks=None, n_max_col=7, size_img=4, lmk_size=5, pre_processing=None,
                   is_black_n_white=False, titles=None):
    n_image = len(images)

    # compute n_row and n_column
    n_col = np.min([n_max_col, n_image])
    n_row = int(n_image / n_col)

    # declare figure
    fig, axs = plt.subplots(n_row, n_col)
    fig.set_figheight(n_row*size_img)
    fig.set_figwidth(n_col*size_img)

    lmk_pos = None

    for i in range(n_row):
        for j in range(n_col):
            img_idx = i * n_col + j

            if lmks is not None:
                lmk_pos = lmks[img_idx]

            title=None
            if titles is not None:
                title = titles[img_idx]

            if n_row == 1 and n_col == 1:
                display_image(images[img_idx], lmks=lmk_pos, ref_lmks=ref_lmks, lmk_size=lmk_size,
                              pre_processing=pre_processing,
                              is_black_n_white=is_black_n_white,
                              figure=axs,
                              title=title)
            elif n_row == 1:
                display_image(images[img_idx], lmks=lmk_pos, ref_lmks=ref_lmks, lmk_size=lmk_size,
                              pre_processing=pre_processing,
                              is_black_n_white=is_black_n_white,
                              figure=axs[img_idx],
                              title=title)
            else:
                display_image(images[img_idx], lmks=lmk_pos, ref_lmks=ref_lmks, lmk_size=lmk_size,
                              pre_processing=pre_processing,
                              is_black_n_white=is_black_n_white,
                              figure=axs[i, j],
                              title=title)