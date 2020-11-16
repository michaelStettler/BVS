import os
import json
import numpy as np
import pandas as pd
import scipy.io
import re
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.warp_image import warp_image

"""
run: python3 -m datasets_utils.create_shape_appearance_images
"""

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=220)

config_file_path = 'configs/face_units/find_face_units_test_mac.json'

# load find_face config
with open(config_file_path) as json_file:
    config = json.load(json_file)

# get front view .mat file
mat = scipy.io.loadmat(os.path.join(config['lmk_path'], config['front_view']))
# print(mat.keys())
lmk_pos = mat['FEILandmarks']
lmk_pos = np.rollaxis(lmk_pos, 2)
n_particpants, n_lmk, n_channel = lmk_pos.shape
print("shape lmk_pos", np.shape(lmk_pos))
view = re.split('[._]', config['front_view'])[1]
print("min max lmk_pos[:, 0]", np.amin(lmk_pos[:, 0]), np.amax(lmk_pos[:, 0]))
print("min max lmk_pos[:, 1]", np.amin(lmk_pos[:, 1]), np.amax(lmk_pos[:, 1]))

# # for each participant
# # for p in range(n_particpants):
# for p in range(5):
#     # test landmarks pos
#     im = cv2.imread(os.path.join(config['orig_img_path'], '{}-{}.jpg'.format(p+1, view)))
#
#     # for each lmk
#     for l in range(n_lmk):
#         im[int(lmk_pos[l, 1, p]), int(lmk_pos[l, 0, p]), :] = [0, 255, 255]
#
#     cv2.imshow("image", im)
#     cv2.waitKey(0)  # waits until a key is pressed
#     cv2.destroyAllWindows()  # destroys the window showing image

# -------------------------------------------------------------------------------------
#                       construct shape features
print("[Shape] Start computing shape features")
# center each faces
mid_image = (640 / 2, 480 / 2)
print("[Shape] Center faces")
centred_lmk = []
for l, lmk in tqdm(enumerate(lmk_pos)):
    mean_face = np.mean(lmk, axis=0)
    offset = mid_image - mean_face
    offset[1] += 15  # add a bit of offset in the vertical axis to avoid shape to get out of the frame
    centred_lmk.append(lmk + offset)
print("shape centred lmk", np.shape(centred_lmk))
# reshape
x_shape = np.reshape(centred_lmk, (n_particpants, -1))
# standardize features
x_shape_mean = np.mean(x_shape, axis=0)
x_shape_std = np.std(x_shape, axis=0)
x_shape_standardize = (x_shape - x_shape_mean)/x_shape_std
print("[Shape] min max x_shape_standardize", np.amin(x_shape_standardize), np.amax(x_shape_standardize))
print("[Shape] Feature standardized", np.shape(x_shape_standardize))
# perform PCA
pca_shape = PCA(n_components=25)
# get 25 first face feature vectors
pca_shape.fit(x_shape_standardize)
print("[Shape] PCA explained variance", pca_shape.explained_variance_ratio_[:5])
print("[Shape] PCA singular values", pca_shape.singular_values_[:5])
print("[Shape] Finished computing shape features")
print()

# -------------------------------------------------------------------------------------
#                       normalize image
print("[Norm Images] Start normalizing images")
# get mean_landmarks
mean_lmk = np.mean(centred_lmk, axis=0)
print("[Norm Images] shape mean_lmk", np.shape(mean_lmk))

# # warp test image
# img = cv2.imread(os.path.join(config['orig_img_path'], '1-11.jpg'))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print("[Norm Images] type img:", img.dtype)
# warp_image(img, lmk_pos[0], mean_lmk, do_plot=True)

# warp all images
norm_images = []
for p in tqdm(range(n_particpants)):
    img = cv2.imread(os.path.join(config['orig_img_path'], '{}-{}.jpg'.format(p+1, view)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    lmks = lmk_pos[p]
    img = warp_image(img, lmks, mean_lmk, do_plot=False)
    norm_images.append(img)

# print("shape images", np.shape(norm_images))
im_width = np.shape(norm_images)[2]
im_height = np.shape(norm_images)[1]
im_channel = np.shape(norm_images)[3]
print("[Norm Images] Finished warping all images")
print()

# -------------------------------------------------------------------------------------
#                       construct appearance features
print("[Appearance] Start computing appearance features")
# flatten feature space
x_appear = np.reshape(norm_images, (n_particpants, -1))
# no standardization since all pixel are from the same distributivity
print("[Appearance] min max x_appear", np.amin(x_appear), np.amax(x_appear))
# perform PCA on the normalized images
pca_appear = PCA(n_components=25)
pca_appear.fit(x_appear)
print("[Appearance] PCA explained variance", pca_appear.explained_variance_ratio_[:5])
print("[Appearance] PCA singular values", pca_appear.singular_values_[:5])
print("[Appearance] Finished computing appearance features")
print()

# -------------------------------------------------------------------------------------
#                       generate


def generate_image(shape_vect, appear_vect, do_plot=False, tri_on_source=False, remove_borders=False):
    # transform shape vector back to original dimension
    gen_shape = pca_shape.inverse_transform(shape_vect)
    gen_shape = gen_shape * x_shape_std + x_shape_mean
    gen_shape = np.reshape(gen_shape, (n_lmk, n_channel))
    # clamp values to 0 - img_width, 0 - img_height
    gen_shape[gen_shape < 0] = 0
    gen_shape[gen_shape[:, 0] >= 479] = 479
    gen_shape[gen_shape[:, 1] >= 639] = 639

    # compute appearance vector
    gen_appear = pca_appear.inverse_transform(appear_vect)
    gen_appear[gen_appear < 0] = 0.0
    gen_appear[gen_appear > 1] = 1.0
    gen_appear = np.reshape(gen_appear, (im_height, im_width, im_channel))

    # built image
    img = warp_image(gen_appear, src_lmk=mean_lmk, dst_lmk=gen_shape, tri_on_source=tri_on_source, remove_borders=remove_borders)

    if do_plot:
        plt.figure()
        plt.imshow(img)
        # cv2.imshow("figure", img)
        # cv2.waitKey(0)

    return img


# test random feature
rand_vect = np.zeros(50)

# create Fig. 4 of the paper by testing shape appearance
for i in [-3, 0, 3]:
    rand_vect[0] = i  # test 1 component of shape features
    # rand_vect[25] = i  # test 1 component of appearance features

    # split shape/appearance vector
    # the sqrt comes from an answer from Rajani Raman which was not described in the paper
    shape_vect = rand_vect[:25] * np.sqrt(pca_shape.singular_values_)
    appear_vect = rand_vect[25:] * np.sqrt(pca_appear.singular_values_)

    # generate img
    img = generate_image(shape_vect, appear_vect, do_plot=True, tri_on_source=True, remove_borders=True)

    img *= 255.0
    img_bgr = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(config['SA_img_path'], 'test' + str(i) + '.jpg'), img_bgr)


# create set of 200 is
df = pd.DataFrame(columns=["img_name", "feature_vector"])
for i in tqdm(range(2000)):
    # generate random vector
    rand_vect = np.random.normal(0, .4, 50)
    # print("rand_vect")
    # print(rand_vect)

    # split shape/appearance vector
    # the sqrt comes from an answer from Rajani Raman which was not described in the paper
    shape_vect = rand_vect[:25] * np.sqrt(pca_shape.singular_values_)
    appear_vect = rand_vect[25:] * np.sqrt(pca_appear.singular_values_)

    img = generate_image(shape_vect, appear_vect, do_plot=False, tri_on_source=True, remove_borders=True)
    img *= 255.0
    img_bgr = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(config['SA_img_path'], str(i)+'.jpg'), img_bgr)

    df = df.append({'img_name': str(i)+'.jpg', 'feature_vector': rand_vect}, ignore_index=True)

df.to_csv(config['SA_csv'])
# plt.show()