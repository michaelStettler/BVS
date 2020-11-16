import numpy as np
import cv2
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt


def warp_image(img, src_lmk, dst_lmk, do_plot=False, tri_on_source=False, remove_borders=False):
    # ------------------------ WARP Image ----------------------------
    # https://www.learnopencv.com/face-morph-using-opencv-cpp-python/
    # https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/

    # plot raw image
    if do_plot:
        plt.figure()
        plt.imshow(img)

    # compute 8 points around the images
    width_img = 640
    height_img = 480
    lmk_img = np.array([[0, 0], [int(width_img / 2), 0], [width_img - 1, 0], [0, int(height_img / 2)],
               [width_img - 1, int(height_img / 2)],
               [0, height_img - 1], [int(width_img / 2), height_img - 1], [width_img - 1, height_img - 1]])

    # extend landmarks with the border landmarks
    lmks_extended = np.concatenate((src_lmk, lmk_img))
    lmk_dst_extended = np.concatenate((dst_lmk, lmk_img))

    # triangulate mean_lmk
    if tri_on_source:
        tri = Delaunay(lmks_extended)
    else:
        tri = Delaunay(lmk_dst_extended)

    # create warp image
    warp_img = 255 * np.ones(img.shape, dtype=img.dtype)

    if do_plot:
        # plot triangulation
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.triplot(lmks_extended[:, 0], lmks_extended[:, 1], tri.simplices)
        # ax.triplot(lmk_dst_extended[:,0], lmk_dst_extended[:,1], tri_mean.simplices)

    # compute affine transform for every triangle pair
    for t in range(len(tri.simplices)):
        # built triangles
        src_tri = np.array(lmks_extended[tri.simplices[t]]).astype(np.float32)
        dst_tri = np.array(lmk_dst_extended[tri.simplices[t]]).astype(np.float32)

        # create bounding box around the triangle
        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)

        # Offset points by left top corner of the respective rectangles
        tri1Cropped = []
        tri2Cropped = []
        for i in range(3):
            tri1Cropped.append(((src_tri[i, 0] - r1[0]), (src_tri[i, 1] - r1[1])))
            tri2Cropped.append(((dst_tri[i, 0] - r2[0]), (dst_tri[i, 1] - r2[1])))

        # Apply warpImage to small rectangular patches
        img1Cropped = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        # Given a pair of triangles, find the affine transform.
        warp_mat = cv2.getAffineTransform(np.array(tri1Cropped).astype(np.float32),
                                          np.array(tri2Cropped).astype(np.float32))

        # transform the cropped rectangle
        img2Cropped = cv2.warpAffine(img1Cropped, warp_mat, (r2[2], r2[3]), borderMode=cv2.BORDER_REFLECT_101)

        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

        # Apply mask to cropped region
        img2Cropped = img2Cropped * mask

        # Copy triangular region of the rectangular patch to the output image
        warp_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = warp_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
        warp_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = warp_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Cropped

    # remove borders
    if remove_borders:
        for t in range(len(tri.simplices)):
            # built triangles
            dst_tri = np.array(lmk_dst_extended[tri.simplices[t]]).astype(np.float32)

            tri_border = np.isin(dst_tri, lmk_img)
            if np.any(tri_border):
                # create bounding box around the triangle
                r2 = cv2.boundingRect(dst_tri)

                # Offset points by left top corner of the respective rectangles
                tri2Cropped = []
                for i in range(3):
                    tri2Cropped.append(((dst_tri[i, 0] - r2[0]), (dst_tri[i, 1] - r2[1])))

                # fill triangle
                cv2.fillConvexPoly(warp_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]],
                                   np.int32(tri2Cropped), (1.0, 1.0, 1.0))
    if do_plot:
        plt.figure()
        plt.imshow(warp_img)

    return np.array(warp_img)
