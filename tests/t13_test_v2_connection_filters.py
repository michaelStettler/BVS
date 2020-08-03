import numpy as np
import cv2
import math
from bvs.utils.create_preds_seq import create_multi_frame_from_multi_channel

"""
script to test different parameters to build the connectivity network for v2 facilitation and iso-suppression

run: python -m tests.t13_test_v2_connection_filters
"""

np.set_printoptions(precision=3, linewidth=250)


# def _build_v2_interconnection(ksize, K, maxBeta, maxTheta, maxTheta2, maxDTheta, verbose):
#     # declare filters
#     W = np.zeros((ksize[0], ksize[1], K, K))
#     J = np.zeros((ksize[0], ksize[1], K, K))
#     kernel = np.zeros((ksize[0], ksize[1], K, K))
#     alpha_kernel = np.zeros(ksize)
#
#     # compute filters for each orientation (K)
#     translate = int(ksize[0] / 2)
#     for k in range(K):
#     # for k in range(2):
#         theta = np.pi / 2 - k * np.pi / K
#         # theta = k * np.pi / K
#         # if theta > np.pi/2:
#         #     theta -= np.pi
#         # theta = -k * np.pi / K
#         for i in range(ksize[0]):
#             for j in range(ksize[1]):
#                 # built kernel with center at the middle
#                 di = i - translate
#                 dj = j - translate
#                 if np.abs(di) > 0:
#                     alpha = np.arctan(dj / di)
#                 elif dj != 0:
#                     alpha = np.pi / 2
#                 else:
#                     alpha = 0
#                 alpha = alpha + np.pi/2
#                 if alpha > np.pi/2:
#                     alpha -= np.pi
#
#                 alpha_kernel[i,j] = alpha
#
#                 d = np.sqrt(di ** 2 + dj ** 2)
#
#                 for dp in range(K):
#                     # compute delta theta
#                     theta_p = np.pi / 2 - dp * np.pi / K  # convert dp index to theta_prime in radians
#                     # theta_p = dp * np.pi / K  # convert dp index to theta_prime in radians
#                     # if theta_p > np.pi/2:
#                     #     theta_p -= np.pi
#                     # theta_p = -dp * np.pi / K  # convert dp index to theta_prime in radians
#
#                     # # Zhao ping's li model
#                     # a = np.abs(theta - theta_p)
#                     # d_theta = min(a, np.pi - a)
#
#                     # # straight line facilitation
#                     # a = np.abs(theta - theta_p)
#                     # d_theta = min(a, np.pi - a)
#                     # b = np.abs(theta - alpha)
#                     # d_beta = min(b, np.pi - b)
#
#                     # spline line facilitation ?
#                     # a = np.abs(theta - (theta_p - alpha))
#                     # a = np.abs(theta - (theta_p - alpha))
#                     # a = theta - theta_p
#                     a = np.abs(theta - theta_p)
#                     a = min(a, np.pi - a)
#                     # b = theta - alpha
#                     b = np.abs(theta - alpha)
#                     b = min(b, np.pi - b)
#                     # c = theta_p - alpha
#                     c = np.abs(theta_p - alpha)
#                     c = min(c, np.pi - c)
#                     # d_theta = a - b
#                     # d_theta = a - c
#                     # d_theta = a - b - c
#                     d_theta = a - b - c
#                     # d_theta = min(a, np.pi - a)
#                     d_theta = d_theta%(2*np.pi)
#
#                     # kernel[i, j, dp, k] = alpha
#                     # kernel[i, j, dp, k] = theta
#                     # kernel[i, j, dp, k] = theta_p
#                     kernel[i, j, dp, k] = d_theta
#                     # kernel[i, j, dp, k] = a
#
#                     # # compute J connections (excitatory)
#                     if 0 < d <= 10 and np.abs(a) < np.pi/2 and d_theta < maxBeta:
#                     # if 0 < d <= 10 and d_theta < maxBeta and d_beta < maxBeta:
#                         # kernel[k, dp, i, j] = d_theta
#                         # kernel[i, j, dp, k] = 1
#                         b_div_d = d_theta / d
#                         # kernel[i, j, dp, k] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d,2) / 90)
#                         J[i, j, dp, k] = 1
#                         # J[i, j, dp, k] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d,2) / 90)
#
#     print("alpha kernel")
#     print(alpha_kernel)
#     print()
#     print("lim:", np.pi/(2*K))
#     # print(J[:,:,0,0])
#     print()
#     f = 0
#     print("f =", f, ": np.pi / 2 - f * np.pi / K =", np.pi / 2 - f * np.pi/K)
#     print()
#     print("dp = 0", np.pi / 2 - 0 * np.pi / K)
#     print(kernel[:,:,0,f])
#     print(J[:,:,0, f])
#     # print()
#     # print("dp = 15", 1 * np.pi / K)
#     # print(kernel[:,:,1,f])
#     # print(J[:, :, 1, f])
#     # # print(J[:,:,1,0])
#     print()
#     print("dp = 30", np.pi / 2 - 2 * np.pi / K)
#     print(kernel[:,:,2,f])
#     print(J[:, :, 2, f])
#     # print(J[:,:,2,0])
#     print()
#     print("dp = 45", np.pi / 2 - 3 * np.pi / K)
#     print(kernel[:,:,3,f])
#     print(J[:, :, 3, f])
#     print()
#     print("dp = 90°", np.pi / 2 - 6 * np.pi / K)
#     print(kernel[:,:,6,f])
#     print(J[:, :, 6, f])
#     print()
#     print("dp = 135", np.pi / 2 - 9 * np.pi / K)
#     print(kernel[:,:,9,f])
#     print(J[:, :, 9, f])
#     print()
#     print("dp = 150", np.pi / 2 - 10 * np.pi / K)
#     print(kernel[:,:,10,f])
#     print(J[:, :, 10, f])
#     # print(J[:,:,3,0])
#
#                     # # compute theta1 and theta2 according to the axis from i, j
#                     # theta1 = theta - alpha
#                     # theta2 = theta_p - alpha
#                     #
#                     # # condition: |theta1| <= |theta2| <= pi/2
#                     # if np.abs(theta1) > np.pi / 2:  # condition 1
#                     #     if theta1 < 0:
#                     #         theta1 += np.pi
#                     #     else:
#                     #         theta1 -= np.pi
#                     #
#                     # if np.abs(theta2) > np.pi / 2:  # condition 2
#                     #     if theta2 < 0:
#                     #         theta2 += np.pi
#                     #     else:
#                     #         theta2 -= np.pi
#                     #
#                     # if np.abs(theta1) > np.abs(theta2):
#                     #     tmp = theta1
#                     #     theta1 = theta2
#                     #     theta2 = tmp
#                     #
#                     # # compute beta
#                     # beta = 2 * np.abs(theta1) + 2 * np.sin(np.abs(theta1 + theta2))
#                     # d_max = 10 * np.cos(beta / 4)
#                     #
#                     # # compute W connections (inhibition)
#                     # if d != 0 and d < d_max and beta >= maxBeta and np.abs(
#                     #         theta1) > maxTheta and d_theta < maxDTheta:
#                     #     # W[i, j, k, dp] = 0.141 * (1 - np.exp(-0.4 * np.power(beta/d, 1.5)))*np.exp(-np.power(d_theta/(np.pi/4), 1.5))
#                     #     W[i, j, dp, k] = 0.141 * (1 - np.exp(-0.4 * np.power(beta / d, 1.5))) * np.exp(
#                     #         -np.power(d_theta / (np.pi / 4),
#                     #                   1.5))  # note the order of k and dp, changed it to fit conv2d
#                     #
#                     # if np.abs(theta2) < maxTheta2:
#                     #     max_beta_J = maxBeta
#                     # else:
#                     #     # max_beta_J = np.pi / 2.69
#                     #     max_beta_J = 0
#                     #
#                     # # compute J connections (excitatory)
#                     # if 0 < d <= 10 and beta < max_beta_J:
#                     #     b_div_d = beta / d
#                     #     # J[i, j, k, dp] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d, 2)/90)
#                     #     J[i, j, dp, k] = 0.126 * np.exp(-np.power(b_div_d, 2) - 2 * np.power(b_div_d, 7) - np.power(d,
#                     #                                                                                                 2) / 90)  # note the order of k and dp, changed it to fit conv2d
#
#     if verbose >= 2:
#         _save_multi_frame_from_multi_channel(W, "bvs/video/v2_inibition_filter.jpeg")
#         _save_multi_frame_from_multi_channel(J, "bvs/video/v2_exitatory_filter.jpeg")
#
#     return W, J

# def _build_v2_interconnection(ksize, K, maxBeta, maxTheta, maxTheta2, maxDTheta, verbose):
#     # declare filters
#     W = np.zeros((ksize[0], ksize[1], K, K))
#     J = np.zeros((ksize[0], ksize[1], K, K))
#
#     alpha_kernel = np.zeros((ksize[0], ksize[1], K, K))
#     d_kernel = np.zeros((ksize[0], ksize[1], K, K))
#     theta_kernel = np.zeros((ksize[0], ksize[1], K, K))
#     theta_p_kernel = np.zeros((ksize[0], ksize[1], K, K))
#
#     translate = int(ksize[0] / 2)
#     for i in range(ksize[0]):
#         for j in range(ksize[1]):
#             # built kernel with center at the middle
#             di = i - translate
#             dj = j - translate
#             if np.abs(di) > 0:
#                 alpha = np.arctan(dj / di)
#             elif dj != 0:
#                 alpha = np.pi / 2
#             else:
#                 alpha = 0
#             alpha = alpha + np.pi / 2
#             if alpha > np.pi / 2:
#                 alpha -= np.pi
#
#             # if di < 0 and alpha < 0:
#             #     alpha += np.pi
#             # if di > 0 and alpha > 0:
#             #     alpha -= np.pi
#
#             alpha_kernel[i, j, :, :] = alpha
#
#             d = np.sqrt(di ** 2 + dj ** 2)
#             d_kernel[i, j, :, :] = d
#
#     # compute filters for each orientation (K)
#     for k in range(K):
#         theta = np.pi / 2 - k * np.pi / K
#         theta_kernel[:,:,:,k] = theta
#
#     for dp in range(K):
#         theta_p = np.pi / 2 - dp * np.pi / K  # convert dp index to theta_prime in radian
#         theta_p_kernel[:, :, dp, :] = theta_p
#
#     # print("d kernel")
#     # print(d_kernel)
#     # print()
#     # print("alpha kernel")
#     # print(alpha_kernel)
#     # print()
#
#     # print()
#     # print("a kernel")
#     # print(a_kernel[0,0,:,0])
#     # print(a_kernel[:,:,0,0])
#     # print(a_kernel[:,:,0,1])
#     # print(a_kernel[:,:,1,1])
#
#     # print()
#     # print("theta kernel")
#     # print(theta_kernel[:,:,0,0])
#     # print(theta_kernel[:,:,0,1])
#     # print(theta_kernel[:,:,0,2])
#     # print(theta_kernel[:,:,1,0])
#     #
#     # print()
#     # print("theta p kernel")
#     # print(theta_p_kernel[:,:,0,0])
#     # print(theta_p_kernel[:,:,1,0])
#     # print(theta_p_kernel[:,:,2,0])
#     # print(theta_p_kernel[:,:,0,1])
#
#     a = np.abs(theta_kernel - theta_p_kernel)
#     a = np.minimum(a, np.pi - a)
#     # print()
#     # print("a")
#     # for i in range(K):
#     #     print("i", i)
#     #     print(a[:,:,0,i])
#
#     # b = theta_kernel - alpha_kernel
#     b = np.abs(theta_kernel - alpha_kernel)
#     b = np.minimum(b, np.pi - b)
#     # print()
#     # print("b")
#     # for i in range(K):
#     #     print("i", i)
#     #     print(b[:,:,0,i])
#
#     # c = theta_p_kernel - alpha_kernel
#     c = np.abs(theta_p_kernel - alpha_kernel)
#     c = np.minimum(c, np.pi - c)
#
#     d_theta = a - b - c
#     # print(a[:,:,0,0])
#     # print(b[:,:,0,0])
#     # print(c[:,:,0,0])
#     # print(d_theta[:,:,0,0])
#
#     print(theta_kernel[:,:,3,0])
#     print(theta_p_kernel[:,:,3,0])
#     print(alpha_kernel[:,:,3,0])
#     #
#     print()
#     print(a[:,:,3,0])
#     print(b[:,:,3,0])
#     print(c[:,:,3,0])
#     print(d_theta[:,:,3,0])
#
#
#     # remove too far and too small
#     d_theta[d_kernel[:] <= 0] = 1
#     d_theta[d_kernel[:] > 10] = 1
#
#     # remove pi/2  remove a small epsilon just to ensure floating differences
#     d_theta[a[:] >= np.pi/2 - 0.001] = 1
#
#     d_theta = np.abs(d_theta)
#     J[d_theta < maxBeta] = 1
#     if verbose >= 2:
#         # _save_multi_frame_from_multi_channel(W, "bvs/video/v2_inibition_filter.jpeg")
#         _save_multi_frame_from_multi_channel(J, "bvs/video/v2_exitatory_filter.jpeg")
#
#     return W, J

def _build_v2_interconnection(ksize, K, maxBeta, maxTheta, maxTheta2, maxDTheta, verbose):
    # declare filters
    W = np.zeros((ksize[0], ksize[1], K, K))
    J = np.zeros((ksize[0], ksize[1], K, K))
    print("maxBeta", maxBeta)

    # alpha_kernel = np.zeros((ksize[0], ksize[1], K, K))
    # beta_kernel = np.zeros((ksize[0], ksize[1], K, K))
    # d_kernel = np.zeros((ksize[0], ksize[1], K, K))
    # theta_kernel = np.zeros((ksize[0], ksize[1], K, K))
    # theta_p_kernel = np.zeros((ksize[0], ksize[1], K, K))

    accept_matrix = np.zeros((ksize[0], ksize[1], K, K))

    translate = int(ksize[0] / 2)
    dMax = 10

    # compute filters for each orientation (K)
    for k in range(K):
    # for k in range(1):
    # for k in [0,1]:
        print()
        print("k", k)
        theta = np.pi / 2 - k * np.pi / K
        # print()
        # print(k, "theta", theta)
        # print()

        for dp in range(K):
        # for dp in range(1):
        # for dp in [0, 1]:
            print()
            theta_p = np.pi / 2 - dp * np.pi / K  # convert dp index to theta_prime in radian
            print("dp", dp, theta_p)
            # print(dp, "theta_p", theta_p)
            # print()

            d_mat = np.zeros(ksize)
            init_d_mat = np.zeros(ksize)
            alpha_mat = np.zeros(ksize)
            d_theta_mat = np.zeros(ksize)
            d_p_theta_mat = np.zeros(ksize)
            for i in range(ksize[0]):
            # for i in [0]:
                for j in range(ksize[1]):
                # for j in [1]:
                    # built kernel with center at the middle
                    di = i - translate
                    dj = j - translate

                    # compute alpha
                    if np.abs(di) > 0:
                        alpha = np.arctan2(dj, di)
                    elif dj != 0:
                        alpha = np.pi / 2
                    else:
                        alpha = np.pi/2
                    # alpha = alpha - np.pi / 2
                    alpha = alpha - theta
                    if alpha < -np.pi:
                        alpha += 2*np.pi
                    # if alpha > 2*np.pi:
                    #     alpha -= 2*np.pi

                    alpha_mat[i, j] = alpha

                    # d_theta = np.abs(alpha - theta)
                    # d_theta = min(d_theta, np.pi - d_theta)
                    # d_theta = np.abs(d_theta)
                    # d_theta_mat[i,j] = d_theta

                    d_p_theta = np.abs(alpha - theta_p)
                    d_p_theta = min(d_p_theta, np.pi - d_p_theta)
                    d_p_theta = np.abs(d_p_theta)
                    d_p_theta_mat[i, j] = d_p_theta

                    # compute d-intercept
                    # if theta == np.pi/2:
                    #     x = 0
                    #     y = di - alpha * dj
                    # else:
                    #     x = di / alpha
                    if theta != theta_p:
                        if theta != np.pi/2 and theta_p != np.pi/2:
                            a1 = np.tan(theta)
                            a2 = np.tan(theta_p)
                            b2 = -di - a2*dj
                            x = b2 / (a1 - a2)
                            y = a1 * x
                        elif theta == np.pi/2:
                            a2 = np.tan(theta_p)
                            b2 = -di - a2*dj
                            x = 0
                            y = b2
                    else:
                        x = 0
                        y = 0

                    d = np.sqrt(x**2 + y**2)
                    d_mat[i, j] = d
                    #
                    d_init = np.sqrt(di**2 + dj**2)
                    init_d_mat[i, j] = d_init
                    # if alpha <= maxBeta and d <= init_d:
                    # # if 0 < d <= d_max and diff_theta <= maxBeta:
                    # if d_p_theta <= maxBeta + 1e-8:
                    if d_p_theta <= maxBeta + 1e-8 and d < d_init:
                        accept_matrix[i, j, dp, k] = 1

            # print("alpha")
            # print(alpha_mat)
            # print("d_theta")
            # print(d_theta_mat)
            print("d_p_theta")
            print(d_p_theta_mat)
            print("d")
            print(d_mat)
            print("init_d")
            print(init_d_mat)
            print("accept_matrix")
            print(accept_matrix[:, :, dp, k])
    # print("accept_matrix")
    # print(accept_matrix)
    # print()
    J[accept_matrix[:] == 1] = 1
    if verbose >= 2:
        # _save_multi_frame_from_multi_channel(W, "bvs/video/v2_inibition_filter.jpeg")
        _save_multi_frame_from_multi_channel(J, "bvs/video/v2_exitatory_filter.jpeg")

    return W, J



def _save_multi_frame_from_multi_channel(x, name):
    max_column = 12
    num_input_channel = np.shape(x)[-2]
    num_filters = num_input_channel * np.shape(x)[-1]
    num_column = min(num_filters, max_column)
    num_row = math.ceil(num_filters / num_column)
    multi_frame = create_multi_frame_from_multi_channel(x, num_row, num_column, (256, 256), num_input_channel)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(name, heatmap.astype(np.uint8))


if __name__ == "__main__":
    ksize = (5, 5)
    K=12
    _build_v2_interconnection(ksize, K,
                              maxBeta=np.pi/(K),   # initial: np.pi / 1.1
                              maxTheta=np.pi / (K - 0.001),
                              maxTheta2=(np.pi/2) / (K / 2 - 0.1),  # width of J filter
                              maxDTheta=np.pi / 3 - 0.00001,
                              verbose=2)