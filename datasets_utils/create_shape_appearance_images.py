import os
import json
import numpy as np
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

np.set_printoptions(precision=3, suppress=True, linewidth=200)

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
x_shape = np.reshape(lmk_pos, (n_particpants, -1))
# normalize each feature
# todo check what pre-processing to do before PCA (center data, norm, standardize?)
x_shape_mean = np.linalg.norm(x_shape, axis=0)
# x_shape_normalized = x_shape / np.repeat(np.expand_dims(x_shape_norm, axis=0), n_particpants, axis=0)
x_shape_normalized = x_shape - np.repeat(np.expand_dims(x_shape_mean, axis=0), n_particpants, axis=0)
print("[Shape] Feature normalized")
# perform PCA
pca_shape = PCA(n_components=25)
# get 25 first face feature vectors
pca_shape.fit(x_shape_normalized)
print("[Shape] PCA explained variance", pca_shape.explained_variance_ratio_[:5])
print("[Shape] PCA singular values", pca_shape.singular_values_[:5])
print("[Shape] Finished computing shape features")
print()

# -------------------------------------------------------------------------------------
#                       normalize image
print("[Norm Images] Start normalizing images")
# get mean_landmarks
mean_lmk = np.mean(lmk_pos, axis=0)
print("[Norm Images] shape mean_lmk", np.shape(mean_lmk))

# # # warp test image
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

print("shape images", np.shape(norm_images))
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
# normalize features
# x_appear_norm = np.linalg.norm(x_appear, axis=0)
# x_appear_normalized = x_appear / np.repeat(np.expand_dims(x_appear_norm, axis=0), n_particpants, axis=0)
# print("shape images", np.shape(norm_images))
# perform PCA on the normalized images
pca_appear = PCA(n_components=25)
pca_appear.fit(x_appear)
print("[Appearance] PCA explained variance", pca_appear.explained_variance_ratio_[:5])
print("[Appearance] PCA singular values", pca_appear.singular_values_[:5])
print("[Appearance] Finished computing appearance features")
print()

# -------------------------------------------------------------------------------------
#                       generate
# test random feature
rand_vect = np.zeros(50)

# for i in shape_test:
for i in [-0.2, 0, 0.2]:
    print("i", i)
    rand_vect[0] = i
    # rand_vect[25] = i

    # compute shape vector
    shape_vect = rand_vect[:25] * pca_shape.singular_values_
    # transform vector back to original dimension
    gen_shape = pca_shape.inverse_transform(shape_vect)
    gen_shape = gen_shape * x_shape_norm
    gen_shape = np.reshape(gen_shape, (n_lmk, n_channel))

    # compute appearance vector
    appear_vect = rand_vect[25:] * pca_appear.singular_values_
    gen_appear = pca_appear.inverse_transform(appear_vect)
    # gen_appear = np.reshape(gen_appear, (im_height, im_width, im_channel)).astype(np.uint8)
    gen_appear = np.reshape(gen_appear, (im_height, im_width, im_channel))

    # built image
    img = warp_image(gen_appear, mean_lmk, gen_shape)

    plt.figure()
    plt.imshow(img)
plt.show()