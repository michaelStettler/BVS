import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from bvs.layers import GaborFilters
from bvs.utils.create_preds_seq import create_multi_frame
from bvs.utils.create_preds_seq import create_multi_frame_heatmap
from bvs.utils.create_preds_seq import create_multi_frame_from_multi_channel

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model

print("Visualize saliency")

config = 'configs/config_test2.json'
image_type = 'simple_vertical_bar'
# load data
if image_type == 'monkey':
    img_path = '../data/02_FearGrin_1.0_120fps/02_FearGrin_1.0_120fps.0000.jpeg'  # monkey face
    img = cv2.imread(img_path)
    # crop and resize image
    img = img[:, 280:1000, :]
    img = cv2.resize(img, (256, 256))
elif image_type == 'simple_vertical_bar':
    img = np.zeros((50, 50, 1))
    img[10:40, 25, :] = 255
    config = 'configs/simple_config.json'
else:
    raise NotImplementedError("Please select a valid image_type to test!")

# load config
# config = 'config.json'
with open(config) as f:
  config = json.load(f)

# # test case img
# img = np.zeros((256, 256, 3))
# img[:128, :128, 0] = 255
# img[128:, 128:, 1] = 255
# plt.imshow(img)

print("Num orientations {}, lambda {}, gamma {}".format(config['n_rot'], config['lamdas'], config['gamma']))

# build model
input = Input(shape=np.shape(img))

n_rot = config['n_rot']
thetas = np.array(range(n_rot)) * np.pi / n_rot
lamdas = np.array(config['lamdas']) * np.pi
gamma = config['gamma']
phi = np.array(config['phi']) * np.pi
use_octave = config['use_octave']
octave = config['octave']
per_channel = config['per_channel']
per_color_channel = config['per_color_channel']
gabor_layer = GaborFilters((15, 15),
                           theta=thetas,
                           lamda=lamdas,
                           gamma=gamma,
                           phi=phi,
                           use_octave=use_octave,
                           octave=octave,
                           per_channel=per_channel,
                           per_color_channel=per_color_channel)
x = gabor_layer(input)

# g_kernels = np.moveaxis(gabor_layer.kernel, -1, 0)
# for gb in g_kernels:
#     if np.shape(gb)[-1] == 1:
#         gb = np.squeeze(gb)
#     gb = (gb+1)/2
#     plt.figure()
#     plt.imshow(gb.astype(np.float32))
# plt.show()
# print("shape g_kernels", np.shape(g_kernels))
# print(g_kernels[0])

kernels = gabor_layer.kernel
num_kernels = np.shape(kernels)[-1]
print("shape kernels", np.shape(kernels))
max_column = 6
num_column = min(num_kernels, max_column)
num_row = math.ceil(num_kernels / num_column)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame(kernels, num_row, num_column, (256, 256))
cv2.imwrite("bvs/video/gabor_filters.jpeg", multi_frame.astype(np.uint8))
# gb0 = kernels[:, :, :, 0]
# gb0 = np.expand_dims(gb0, axis=3)
# print("shape gb0", np.shape(gb0))
# gb0 = create_multi_frame(gb0, 1, 1, (256, 256))
# cv2.imwrite("bvs/video/gabor_filter_hori.jpeg", gb0.astype(np.uint8))


model = Model(inputs=input, outputs=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

test_img = np.expand_dims(img, axis=0)
pred = model.predict(x=test_img)
print("shape pred", np.shape(pred))

activations = pred[0]
print("shape activations", np.shape(activations))

num_activations = np.shape(activations)[-1]
num_column = min(num_activations, max_column)
num_row = math.ceil(num_activations / num_column)
print("num column", num_column, "num_row", num_row)
multi_frame = create_multi_frame_heatmap(img, activations, num_row, num_column, (np.shape(img)[1], np.shape(img)[0]))

# plt.figure()
# plt.imshow(multi_frame)
# plt.show()
cv2.imwrite("bvs/video/heatmap_gabor_filters.jpeg", multi_frame.astype(np.uint8))

print("shape activations", np.shape(activations))
saliency_type = "max"
if saliency_type == 'max':
    saliency = activations.max(axis=2)
elif saliency_type == 'sum':
    saliency = np.sum(activations, axis=2)
elif saliency_type == "ReLu_sum":
    activations[activations < 0] = 0
    saliency = np.sum(activations, axis=2)
else:
    raise NotImplementedError("please chose a valid saliency type!")

# normalize saliency
saliency = saliency - np.min(saliency)
saliency = saliency / np.max(saliency)
saliency *= 255
print("shape saliency", np.shape(saliency))
cv2.imwrite("bvs/video/saliency.jpeg", saliency.astype(np.uint8))

# --------------------- Zhaoping li's V1 Saliency Model --------------------- #
# start from activation of each filter
# normalize outputs
activations = activations - np.min(activations)
activations = activations / np.max(activations)


# declare variables
def gx(x, Tx=1):
    x = np.array(x)
    x[x < Tx] = 0
    x = x - Tx
    x[x > Tx + 1] = 1
    return x
    # if x < Tx:
    #     return 0
    # elif x < Tx + 1:
    #     return x - Tx
    # else:
    #     return 1


def gy(y, Ly=1.2, g1=0.21, g2=2.5):
    y = np.array(y)
    y[y < 0] = 0
    y[y <= Ly] = g1 * y[y <= Ly]
    y[y > Ly] = g1 * Ly * g2 * (y[y > Ly] - Ly)
    return y
    # if y < 0:
    #     return 0
    # elif y <= Ly:
    #     return g1 * y
    # else:
    #     return g1 * Ly * g2 * (y - Ly)


def psi(theta, K, epsilon=0.001):
    theta = np.abs(theta)
    if -epsilon < theta < epsilon:
        return 1
    elif np.pi / K - epsilon < theta < np.pi / K + epsilon:
        return 0.8
    elif 2 * np.pi / K - epsilon < theta < 2 * np.pi / K + epsilon:
        return 0.7
    else:
        return 0

# --------------------------------- W/J Matrix --------------------------------
kernel_size = (15, 15)
# kernel_size = (15, 15)
translate = int(kernel_size[0]/2)
max_theta = np.pi / (n_rot - 0.001)
max_beta = np.pi / 1.1
max_d_theta = np.pi / 3
max_theta2 = np.pi / (n_rot / 2 - 0.1)
Ic_control = 0
Ic = 1 + Ic_control
alphaX = 1
alphaY = 1
J0 = 0.8
mirror_idx = 2

W = np.zeros((kernel_size[0], kernel_size[1], n_rot, n_rot))
J = np.zeros((kernel_size[0], kernel_size[1], n_rot, n_rot))
print("shape J", np.shape(J))
print("shape W", np.shape(W))
for k in range(n_rot):
    theta = k * np.pi / n_rot
    # print()
    # print("theta", theta, theta / np.pi * 180)
    if theta <= np.pi/2:  # save some computations by mirroring matrix
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                # built kernel with center at the middle
                di = i - translate
                dj = j - translate
                alpha = np.arctan2(-di, dj)  # -di because think of alpha in a normal x,y coordinate and not a matrix
                # therefore axes i should goes up
                if np.abs(alpha) >= np.pi / 2:
                    if alpha < 0:
                        alpha += np.pi
                    else:
                        alpha -= np.pi
                d = np.sqrt(di**2 + dj**2)

                for dp in range(n_rot):
                    # compute delta theta
                    theta_p = dp * np.pi / n_rot  # convert dp index to theta_prime in radians
                    a = np.abs(theta - theta_p)
                    d_theta = min(a, np.pi - a)
                    # compute theta1 and theta2 according to the axis from i, j
                    theta1 = theta - alpha
                    theta2 = np.pi - (theta_p - alpha)
                    # condition: |theta1| <= |theta2| <= pi/2
                    if np.abs(theta1) > np.pi / 2:  # condition1
                        if theta1 < 0:
                            theta1 += np.pi
                        else:
                            theta1 -= np.pi
                    theta2 = np.pi - (theta_p - alpha)
                    if np.abs(theta2) > np.pi / 2:  # condition 2
                        if theta2 < 0:
                            theta2 += np.pi
                        else:
                            theta2 -= np.pi
                    # compute beta
                    beta = 2 * np.abs(theta1) + 2 * np.sin(np.abs(theta1 + theta2))
                    d_max = 10 * np.cos(beta/4)
                    if d != 0 and d <= d_max and beta >= max_beta and np.abs(theta1) >= max_theta and d_theta <= max_d_theta:
                        W[i, j, k, dp] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))

                    if np.abs(theta2) < max_theta2:
                        max_beta_J = max_beta
                    else:
                        max_beta_J = np.pi / 2.69

                    if 0 < d <= 10 and beta <= max_beta_J:
                        # J[i, j, k, dp] = 1
                        b_div_d = beta/d
                        J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d, 2)/90)
                        # J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7))
                        # J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2))

    else:
        # since the angle are symmetric, we can flip and roll the matrix
        # mirror index is used to compensate for the "backward" operation from the middle angle (pi/2)
        W[:, :, k, :] = np.roll(np.flip(W[:, :, k-mirror_idx, :], axis=0), mirror_idx, axis=2)
        J[:, :, k, :] = np.roll(np.flip(J[:, :, k-mirror_idx, :], axis=0), mirror_idx, axis=2)
        mirror_idx += 2

# np.set_printoptions(precision=3, linewidth=200)
# for dp in range(n_rot):
#     print()
#     print(J[:, :, 0, dp])
#     print("theta prime:", dp * np.pi / n_rot / np.pi * 180)

# save W inhibition filters
idx = 11
W_print = W[:, :, idx, :]
W_print = np.expand_dims(W_print, axis=2)
num_filters = np.shape(W)[-1]
num_column = min(num_filters, max_column)
num_row = math.ceil(num_filters / num_column)
print("shape W_print", np.shape(W_print))
multi_frame = create_multi_frame(W_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/W"+str(idx)+"_inibition_filter.jpeg", heatmap.astype(np.uint8))

# save J exitatory filters
J_print = J[:, :, idx, :]
J_print = np.expand_dims(J_print, axis=2)
multi_frame = create_multi_frame(J_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/J"+str(idx)+"_exitatory_filter.jpeg", heatmap.astype(np.uint8))

# save all W inhibition filters
num_input_channel = np.shape(W)[-2]
num_filters = num_input_channel * np.shape(W)[-1]
print("num_filters", num_filters)
num_column = min(num_filters, max_column)
num_row = math.ceil(num_filters / num_column)
print("num_row", num_row, "num_colum", num_column)
multi_frame = create_multi_frame_from_multi_channel(W, num_row, num_column, (256, 256), num_input_channel)
print("shape mulit_frame", np.shape(multi_frame))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/W_inibition_filter.jpeg", heatmap.astype(np.uint8))

# save J exitatory filters
multi_frame = create_multi_frame_from_multi_channel(J, num_row, num_column, (256, 256), num_input_channel)
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/J_exitatory_filter.jpeg", heatmap.astype(np.uint8))

# convolutions
activations = np.expand_dims(activations, axis=0)
inhib = tf.nn.conv2d(activations, W, strides=1, padding='SAME')
inhib_print = np.expand_dims(inhib[0], axis=2)  # todo make it times the gx!

excit = tf.nn.conv2d(activations, J, strides=1, padding='SAME')
excit_print = np.expand_dims(excit[0], axis=2)  # todo make it times the gx!


print("shape inhib", np.shape(inhib))
print("shape inhib_print", np.shape(inhib_print))
print("shape excit_print", np.shape(excit_print))
num_filters = np.shape(inhib)[-1]
num_column = min(num_filters, max_column)
num_row = math.ceil(num_filters / num_column)
# save inhibition
multi_frame = create_multi_frame(inhib_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/inibition_response.jpeg", heatmap.astype(np.uint8))
# save excitation
multi_frame = create_multi_frame(excit_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/exitatory_response.jpeg", heatmap.astype(np.uint8))

# save activations
activations_print = np.expand_dims(activations[0], axis=2)
multi_frame = create_multi_frame(activations_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/activations_x_response.jpeg", heatmap.astype(np.uint8))

# neural response
x = activations.copy()  # todo x0 as activations or zeors ?
# x = np.zeros(np.shape(activations))
y = np.zeros(np.shape(activations))
print("min max activations", np.min(activations), np.max(activations))
y = -alphaY * y + gx(x) + inhib + Ic
print("shape y", np.shape(y))
print("min max y", np.min(y), np.max(y))

# print y neuronal response (inhibitory)
y = y - np.min(y)
y = y / np.max(y)
y_print = np.expand_dims(y[0], axis=2)
multi_frame = create_multi_frame(y_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/V1_y_response.jpeg", heatmap.astype(np.uint8))
# print gy(y)
gy_print = np.expand_dims(gy(y)[0], axis=2)
multi_frame = create_multi_frame(gy_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/gy(y)_response.jpeg", heatmap.astype(np.uint8))

# build inhibitory psi matrix
inhib_psi = np.zeros(np.shape(y))
print("shape inhib_psi", np.shape(inhib_psi))
for t in range(n_rot):
    psi_tmp = []
    for t_p in range(n_rot):
        if t != t_p:
            theta = t * np.pi / n_rot
            theta_p = t_p * np.pi / n_rot
            a = np.abs(theta - theta_p)
            dt = min(a, np.pi - a)

            psi_tmp.append(psi(dt, n_rot) * gy(y[:, :, :, t_p]))
    inhib_psi[:, :, :, t] = np.sum(psi_tmp, axis=0)

# save inhib psi
inhib_psi_print = np.expand_dims(inhib_psi[0], axis=2)
multi_frame = create_multi_frame(inhib_psi_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/inibition_psi.jpeg", heatmap.astype(np.uint8))

# x = -alphaX * x - gy(y) - inhib_psi + J0 * gx(x) + excit  # term, I_{I, theta} and I0
x = -alphaX * x - gy(y) - inhib_psi + J0 * gx(x) + excit
print("shape x", np.shape(x))
print("min max x", np.min(x), np.max(x))


# plot V1 response
x = x - np.min(x)
x = x / np.max(x)
x_print = np.expand_dims(x[0], axis=2)
multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/V1_x_response.jpeg", heatmap.astype(np.uint8))

saliency_map = np.expand_dims(np.squeeze(np.max(x, axis=3)), axis=2)
saliency_map = np.array(saliency_map * 255).astype(np.uint8)
print("shape saliency map", np.shape(saliency_map))
heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/V1_saliency_map.jpeg", heatmap.astype(np.uint8))



