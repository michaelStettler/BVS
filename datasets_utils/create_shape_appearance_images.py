import os
import json
import numpy as np
import scipy.io
import cv2
import re
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

"""
run: python3 -m datasets_utils.create_shape_appearance_images
"""

np.set_printoptions(precision=3, suppress=True, linewidth=200)

config_file_path = 'configs/face_units/find_face_units_test_mac.json'

# load find_face config
with open(config_file_path) as json_file:
    config = json.load(json_file)

# get all front view .mat file
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

# perform PCA
x = np.reshape(lmk_pos, (n_lmk * n_channel, n_particpants)).T
pca = PCA(n_components=25)
x_new = pca.fit_transform(x)
# print(pca.explained_variance_ratio_[:25])
# print(pca.singular_values_[:25])
print("shape x_new", np.shape(x_new))

# get mean_landmarks
mean_lmk = np.mean(lmk_pos, axis=2)
print("shape mean_lmk", np.shape(mean_lmk))

# ------------------------ WARP Image ----------------------------
# https://www.learnopencv.com/face-morph-using-opencv-cpp-python/

# triangulate mean_lmk
tri_mean = Delaunay(mean_lmk)

# triangulate image
p = 0
lmks = lmk_pos[:, :, p]
print("shape lmks", np.shape(lmks))

tri = Delaunay(lmks)

# plot
# img = plt.imread(os.path.join(config['orig_img_path'], '1-11.jpg'))
img = cv2.imread(os.path.join(config['orig_img_path'],  '1-11.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for l in range(n_lmk):
        img[int(lmks[l, 1]), int(lmks[l, 0]), :] = [0, 255, 255]
fig, ax = plt.subplots()
ax.imshow(img)
ax.triplot(lmks[:,0], lmks[:,1], tri.simplices)
plt.show()