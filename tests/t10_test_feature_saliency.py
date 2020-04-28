import numpy as np
import cv2
import json
import math
from bvs.utils.create_examples import *
from bvs.utils.create_preds_seq import create_multi_frame
from bvs.utils.create_preds_seq import create_multi_frame_heatmap
from bvs.utils.create_preds_seq import create_multi_frame_from_multi_channel

import tensorflow as tf

print("Visualize saliency from Zhaoping model")

print("MODEL IS WRONG! KEPT IT FOR HISTORIC REASONS!!!!!!!!")
print("BASICALLY KERNELS ARE 90 DEGREE ROTATED")

img = get_fig_5_14F()
n_rot = 12
max_column = 6
# --------------------- Zhaoping li's V1 Saliency Model --------------------- #
# start from activation of each filter
# normalize outputs
activations = img - np.min(img)
activations = activations / np.max(activations)
activations = np.expand_dims(activations, axis=0)


# declare variables
def gx(x, Tx=1):
    x = np.array(x)
    x[x <= Tx] = 0
    x[x > Tx] = x[x > Tx] - Tx
    x[x > 1] = 1
    return x


def gy(y, Ly=1.2, g1=0.21, g2=2.5):
    y = np.array(y)
    y[y < 0] = 0
    y[y <= Ly] = g1 * y[y <= Ly]
    y[y > Ly] = g1 * Ly + g2 * (y[y > Ly] - Ly)
    return y


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
max_d_theta = np.pi / 3 - 0.00001  # -0.0001 it's just to compensate some minor imprecision when using np.pi
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
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            # built kernel with center at the middle
            di = i - translate
            dj = j - translate
            alpha = np.arctan2(-di, dj)  # -di because think of alpha in a normal x,y coordinate and not a matrix
            # therefore axes i should goes up
            if np.abs(alpha) > np.pi / 2:
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
                if d != 0 and d < d_max and beta >= max_beta and np.abs(theta1) > max_theta and d_theta < max_d_theta:
                    W[i, j, k, dp] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))

                if np.abs(theta2) < max_theta2:
                    max_beta_J = max_beta
                else:
                    max_beta_J = np.pi / 2.69
                if 0 < d <= 10 and beta < max_beta_J:
                    # J[i, j, k, dp] = 1
                    b_div_d = beta/d
                    J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d, 2)/90)

# np.set_printoptions(precision=3, linewidth=200)
# for dp in range(n_rot):
#     print()
#     print(J[:, :, 0, dp])
#     print("theta prime:", dp * np.pi / n_rot / np.pi * 180)

# # save W inhibition filters
# idx = 11
# W_print = W[:, :, idx, :]
# W_print = np.expand_dims(W_print, axis=2)
# num_filters = np.shape(W)[-1]
# num_column = min(num_filters, max_column)
# num_row = math.ceil(num_filters / num_column)
# print("shape W_print", np.shape(W_print))
# multi_frame = create_multi_frame(W_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/W"+str(idx)+"_inibition_filter.jpeg", heatmap.astype(np.uint8))
#
# # save J exitatory filters
# J_print = J[:, :, idx, :]
# J_print = np.expand_dims(J_print, axis=2)
# multi_frame = create_multi_frame(J_print, num_row, num_column, (256, 256))
# heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
# cv2.imwrite("bvs/video/J"+str(idx)+"_exitatory_filter.jpeg", heatmap.astype(np.uint8))

# save all W inhibition filters
num_input_channel = np.shape(W)[-2]
num_filters = num_input_channel * np.shape(W)[-1]
num_column = min(num_filters, 12)
num_row = math.ceil(num_filters / num_column)
multi_frame = create_multi_frame_from_multi_channel(W, num_row, num_column, (256, 256), num_input_channel)
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/W_inibition_filter.jpeg", heatmap.astype(np.uint8))

# save J exitatory filters
multi_frame = create_multi_frame_from_multi_channel(J, num_row, num_column, (256, 256), num_input_channel)
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/J_exitatory_filter.jpeg", heatmap.astype(np.uint8))


# start dynamic
# ----------------------------------------------------------------------------------------------------------------------
# save activations
save_intermediate_img = True
I_i_theta = activations
print("[declaration] shape I_i_theta", np.shape(I_i_theta))
print("[declaration] min max I_i_theta", np.min(I_i_theta), np.max(I_i_theta))
# x = activations.copy() + 0.5
x = np.zeros(np.shape(activations))
y = np.zeros(np.shape(activations))
i_norm_k = np.ones((5, 5, n_rot, n_rot))
print("[declaration] shape x", np.shape(x), "min max x", np.min(x), np.max(x))
print("[declaration] shape y", np.shape(y), "min max y", np.min(y), np.max(y))
print()

num_filters = np.shape(I_i_theta)[-1]
num_column = min(num_filters, max_column)
num_row = math.ceil(num_filters / num_column)
x_print = np.expand_dims(I_i_theta[0], axis=2)
multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/0_00_I_i_theta.jpeg", heatmap.astype(np.uint8))

# reshape W and J
print("shape W", np.shape(W))
W = np.expand_dims(np.moveaxis(W, 2, 0), axis=4)
J = np.expand_dims(np.moveaxis(J, 2, 0), axis=4)
print("shape W", np.shape(W))

# epsilon = 0.01
epsilon = 0.1
for t in range(2):
    print()
    print("----------------------------------------------------------")
    print("t", t)
    # i_noise_y = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1
    # i_noise_x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3]) / 10 + 0.1
    i_noise_y = 0
    i_noise_x = 0

    if save_intermediate_img:
        num_filters = np.shape(x)[-1]
        num_column = min(num_filters, max_column)
        num_row = math.ceil(num_filters / num_column)
        x_print = np.expand_dims(x[0], axis=2)
        multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_01_x.jpeg", heatmap.astype(np.uint8))
        # save gx activations
        gx_print = np.expand_dims(gx(x)[0], axis=2)
        multi_frame = create_multi_frame(gx_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_02_gx(x)_response.jpeg", heatmap.astype(np.uint8))

    # convolutions
    print("[convolution] shape W, np.shape(W)", np.shape(W))
    print("[convolution] shape gx(activations)", np.shape(gx(x)))
    print("[convolution] min max gx(activations)", np.min(gx(x)), np.max(gx(x)))
    i_norm = 0.85 - 2 * np.power(tf.nn.conv2d(gx(x), i_norm_k, strides=1, padding='SAME') / (np.shape(i_norm_k)[0]**2), 2)
    print("[convolution] 03")
    print("[convolution] shape i_norm", np.shape(i_norm))
    print("[convolution] min max i_norm", np.min(i_norm), np.max(i_norm))
    inhibs = []
    excits = []
    for i in range(n_rot):
        # W_print = W[i]
        # J_print = J[i]
        # print("shape W print", np.shape(W_print))

        inhib = tf.nn.conv2d(gx(x), W[i], strides=1, padding='SAME')
        excit = tf.nn.conv2d(gx(x), J[i], strides=1, padding='SAME')
        inhibs.append(inhib)
        excits.append(excit)

        # # print filter
        # W_print = np.swapaxes(W_print, 2, 3)
        # print("shape W print", np.shape(W_print))
        # num_filters = np.shape(W_print)[-1]
        # num_column = min(num_filters, max_column)
        # num_row = math.ceil(num_filters / num_column)
        # print("num_row", num_row, "num_column", num_column)
        # multi_frame = create_multi_frame(W_print, num_row, num_column, (256, 256))
        # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        # cv2.imwrite("bvs/video/"+str(t)+"_W"+str(i)+"_inibition_filter.jpeg", heatmap.astype(np.uint8))
        #
        # J_print = np.swapaxes(J_print, 2, 3)
        # multi_frame = create_multi_frame(J_print, num_row, num_column, (256, 256))
        # heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        # cv2.imwrite("bvs/video/"+str(t)+"_J"+str(i)+"_excitatory_filter.jpeg", heatmap.astype(np.uint8))
        #
        # # print inhib convolution
        # print("shape inhib", np.shape(inhib))
        # inhib = np.squeeze(inhib)
        # print("min max inhib", np.min(inhib), np.max(inhib))
        # inhib = np.array(inhib * 255).astype(np.uint8)
        # heatmap = cv2.applyColorMap(inhib, cv2.COLORMAP_VIRIDIS)
        # cv2.imwrite("bvs/video/"+str(t)+"_inhib" + str(i) + "_response.jpeg", heatmap.astype(np.uint8))
        #
        # # print excit convolution
        # print("shape excit", np.shape(excit))
        # excit = np.squeeze(excit)
        # print("min max excit", np.min(excit), np.max(excit))
        # excit = np.array(excit * 255).astype(np.uint8)
        # heatmap = cv2.applyColorMap(excit, cv2.COLORMAP_VIRIDIS)
        # cv2.imwrite("bvs/video/"+str(t)+"_excit" + str(i) + "_response.jpeg", heatmap.astype(np.uint8))

    print("shape inhibs", np.shape(inhibs))
    inhibs = np.swapaxes(np.expand_dims(np.squeeze(inhibs), axis=3), 3, 0)
    excits = np.swapaxes(np.expand_dims(np.squeeze(excits), axis=3), 3, 0)

    if save_intermediate_img:
        # save i_norm
        i_norm_print = np.expand_dims(i_norm[0], axis=2)
        i_norm_print = np.array(i_norm_print).astype(np.uint8)
        multi_frame = create_multi_frame(i_norm_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_03_i_norm.jpeg", heatmap.astype(np.uint8))
        # save inhibition
        inhibs_print = np.expand_dims(inhibs[0], axis=2)
        inhibs_print = np.array(inhibs_print).astype(np.uint8)
        num_filters = np.shape(inhibs)[-1]
        num_column = min(num_filters, max_column)
        num_row = math.ceil(num_filters / num_column)
        multi_frame = create_multi_frame(inhibs_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_04_inibition_response.jpeg", heatmap.astype(np.uint8))
        # save excitation
        excits_print = np.expand_dims(excits[0], axis=2)
        excits_print = np.array(excits_print).astype(np.uint8)
        multi_frame = create_multi_frame(excits_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_05_exitatory_response.jpeg", heatmap.astype(np.uint8))

    print("[convolution] 04")
    print("[convolution] shape inhibs", np.shape(inhibs))
    print("[convolution] min max inhibs", np.min(inhibs), np.max(inhibs))
    print("[convolution] 05")
    print("[convolution] shape excit", np.shape(excits))
    print("[convolution] min max excit", np.min(excits), np.max(excits))
    print()

    # neural response
    ####################################################################################################################
    # y = -alphaY * y + gx(x) + inhibs + Ic + i_noise_y
    y += epsilon * (-alphaY * y + gx(x) + inhibs + Ic + i_noise_y)
    ####################################################################################################################
    print("[Y response] 06")
    print("[Y response] shape y", np.shape(y))
    print("[Y response] min max y", np.min(y), np.max(y))
    print("[Y response] 07")
    print("[Y response] shape gy(y)", np.shape(gy(y)))
    print("[Y response] min max gy(y)", np.min(gy(y)), np.max(gy(y)))
    print()

    if save_intermediate_img:
        # print y neuronal response (inhibitory)
        y_print = np.expand_dims(y[0], axis=2)
        multi_frame = create_multi_frame(y_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_06_y_responses.jpeg", heatmap.astype(np.uint8))
        # print gy(y)
        gy_print = np.expand_dims(gy(y)[0], axis=2)
        multi_frame = create_multi_frame(gy_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_07_gy(y)_response.jpeg", heatmap.astype(np.uint8))

    # build inhibitory psi matrix
    inhibs_psi = np.zeros(np.shape(y))
    print("[psi matrix] 08")
    print("[psi matrix] shape inhib_psi", np.shape(inhibs_psi))
    for th in range(n_rot):
        psi_tmp = []
        for t_p in range(n_rot):
            if th != t_p:
                theta = th * np.pi / n_rot
                theta_p = t_p * np.pi / n_rot
                a = np.abs(theta - theta_p)
                dth = min(a, np.pi - a)

                psi_tmp.append(psi(dth, n_rot) * gy(y[:, :, :, t_p]))
        inhibs_psi[:, :, :, th] = np.sum(psi_tmp, axis=0)
    print("[psi matrix] shape inhibs_psi", np.shape(inhibs_psi))
    print("[psi matrix] min man inhibs_psi", np.min(inhibs_psi), np.max(inhibs_psi))

    # save inhib psi
    inhibs_psi_print = np.expand_dims(inhibs_psi[0], axis=2)
    multi_frame = create_multi_frame(inhibs_psi_print, num_row, num_column, (256, 256))
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("bvs/video/"+str(t)+"_08_inibition_psi.jpeg", heatmap.astype(np.uint8))

    ####################################################################################################################
    x_inhib = -alphaX * x - gy(y) - inhibs_psi
    x_excit = J0 * gx(x) + excits + I_i_theta + i_norm + i_noise_x
    # x = -alphaX * x - gy(y) - inhib_psi + J0 * gx(x) + excit  # term, I_{I, theta} and I0
    # x = x_inhib + x_excit
    x += epsilon * (x_inhib + x_excit)  # that what I understood from Zhaoping's li code
    ####################################################################################################################
    print("[X response] 09 min max x_inhib", np.min(x_inhib), np.max(x_inhib))
    print("[X response] 10 min max x_excit", np.min(x_excit), np.max(x_excit))
    print("[X response] 11 min max x", np.min(x), np.max(x))


    if save_intermediate_img:
        # plot V1 response
        x_inhib_print = np.expand_dims(x_inhib[0], axis=2)
        multi_frame = create_multi_frame(x_inhib_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_09_x_inhib_response.jpeg", heatmap.astype(np.uint8))

        x_excit_print = np.expand_dims(x_excit[0], axis=2)
        multi_frame = create_multi_frame(x_excit_print, num_row, num_column, (256, 256))
        heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite("bvs/video/"+str(t)+"_10_x_excit_response.jpeg", heatmap.astype(np.uint8))

    x_print = np.expand_dims(x[0], axis=2)
    # x_print = x_print - np.min(x_print)
    # x_print = x_print / np.max(x_print)
    # x_print[x_print < 0] = 0
    print("min max x_print", np.min(x_print), np.max(x_print))
    multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("bvs/video/"+str(t)+"_11_V1_x_response.jpeg", heatmap.astype(np.uint8))

    x_print = np.expand_dims(gx(x)[0], axis=2)
    multi_frame = create_multi_frame(x_print, num_row, num_column, (256, 256))
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite("bvs/video/"+str(t)+"_12_V1_gx(x)_response.jpeg", heatmap.astype(np.uint8))


saliency_map = np.expand_dims(np.squeeze(np.sum(gx(x)[0], axis=2)), axis=2)
saliency_map = np.array(saliency_map * 255).astype(np.uint8)
print("shape saliency map", np.shape(saliency_map))
heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_VIRIDIS)
cv2.imwrite("bvs/video/V1_saliency_map.jpeg", heatmap.astype(np.uint8))



