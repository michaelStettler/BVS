import numpy as np
import cv2

img_size = (256, 256, 3)
sequence_length = 16

# declare variables
theta = 0  # np.pi/4
k = 1
omega = 1 / sequence_length * 2 * np.pi

# compute parameters
(y, x) = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
x_theta = x * np.cos(theta) + y * np.sin(theta)

# create sequence
sequence = np.zeros((sequence_length, img_size[0], img_size[1], img_size[2]))
for t in range(sequence_length):
    # create ratings
    img = np.cos(k * x_theta + t * omega)
    # modify to be a 3D image
    img = np.expand_dims(img, axis=2)   # img = img[:, :, np.newaxis]
    img = np.repeat(img, 3, axis=2)

    # normalize image
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255

    # cv2.imshow("image", img)
    # cv2.waitKey(0)

    cv2.imwrite("bvs/video/grating_{}.jpeg".format(t), img.astype(np.uint8))

