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

config_file_path = 'configs/face_units/find_face_units_test.json'

# load find_face config
with open(config_file_path) as json_file:
    config = json.load(json_file)

# get front view .mat file
mat = scipy.io.loadmat(os.path.join(config['lmk_path'], config['front_view']))
# print(mat.keys())
lmk_pos = mat['FEILandmarks']
n_lmk, n_channel, n_particpants = lmk_pos.shape
print("shape", n_lmk, n_channel, n_particpants)
view = re.split('[._]', config['front_view'])[1]

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
# perform PCA
x = np.reshape(lmk_pos, (-1, n_particpants)).T
print("[Shape] shape x", np.shape(x))
pca_shape = PCA(n_components=25)
# get 25 first face feature vectors
pca_shape.fit(x)
# print(pca.explained_variance_ratio_[:25])
# print(pca.singular_values_[:25])
print("[Shape] Finished computing shape features")
print()

# -------------------------------------------------------------------------------------
#                       normalize image
print("[Norm Images] Start normalizing images")
# get mean_landmarks
mean_lmk = np.mean(lmk_pos, axis=2)

# # warp test image
img = cv2.imread(os.path.join(config['orig_img_path'], '1-11.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("[Norm Images] type img:", img.dtype)
# lmks = lmk_pos[:, :, 0]
# warp_image(img, lmks, mean_lmk, do_plot=True)

# warp all images
norm_images = []
for p in tqdm(range(n_particpants)):
    img = cv2.imread(os.path.join(config['orig_img_path'], '{}-{}.jpg'.format(p+1, view)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lmks = lmk_pos[:, :, p]
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
norm_images = np.reshape(norm_images, (n_particpants, -1))
print("shape images", np.shape(norm_images))
# perform PCA on the normalized images
pca_appear = PCA(n_components=25)
pca_appear.fit(norm_images)
print("[Appearance] Finished computing appearance features")
print()

# -------------------------------------------------------------------------------------
#                       generate
# test random feature
rand_vect = np.zeros(50)
shape_test = [-3, 0, 3]
for i in shape_test:
    rand_vect[0] = i
    # transform vector back to original dimension
    gen_shape = pca_shape.inverse_transform(rand_vect[:25])
    # gen_shape = np.reshape(gen_shape, (n_lmk, n_channel))
    gen_shape = np.reshape(gen_shape, (n_channel, n_lmk)).T
    print("gen_shape", np.shape(gen_shape))
    gen_appear = pca_appear.inverse_transform(rand_vect[25:])
    gen_appear = np.reshape(gen_appear, (im_height, im_width, im_channel)).astype(np.uint8)

    # built image
    img = warp_image(gen_appear, mean_lmk, gen_shape, do_plot=True)

plt.show()