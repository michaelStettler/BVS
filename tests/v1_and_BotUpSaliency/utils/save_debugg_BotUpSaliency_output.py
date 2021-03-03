import numpy as np
import cv2
import math

from models.layers import create_multi_frame


def save_debugg_BotUp_output(pred, t=None, custom_img_size=False, idx=0):
    # remember to change the output in BotUp_saliency_layer
    if t is None:
        path = 'layers/video/a_'
    else:
        path = 'layers/video/a'+str(t)+'_'

    if custom_img_size is True:
        img_size = np.shape(pred[0][0])[0:2]
        img_size = (img_size[0] * 4, img_size[1] * 4)
    else:
        img_size = (256, 256)

    I_i_theta = pred[0][0]
    x = pred[1][0]
    gx = pred[2][0]
    i_norm = pred[3][0]
    inhibs = pred[4][0]
    excits = pred[5][0]
    y = pred[6][0]
    gy = pred[7][0]
    inhibs_psi = pred[8][0]
    x_inhib = pred[9][0]
    x_excit = pred[10][0]
    saliency = pred[11][0]

    print()
    print("shape I_i_theta", np.shape(I_i_theta))
    print("min max I_i_theta", np.min(I_i_theta), np.max(I_i_theta))
    # print("shape x", np.shape(x))
    print("min max x", np.min(x), np.max(x))
    # print("shape gx", np.shape(gx))
    print("min max gx", np.min(gx), np.max(gx))
    print("shape i_norm", np.shape(i_norm))
    print("min max i_norm", np.min(i_norm), np.max(i_norm))
    # print("shape inhibs", np.shape(inhibs))
    print("min max inhibs", np.min(inhibs), np.max(inhibs))
    # print("shape excits", np.shape(excits))
    print("min max excits", np.min(excits), np.max(excits))
    # print("shape y", np.shape(y))
    print("min max y", np.min(y), np.max(y))
    # print("shape gy", np.shape(gy))
    print("min max gy", np.min(gy), np.max(gy))
    print("shape inhibs_psi", np.shape(inhibs_psi))
    print("min max inhibs_psi", np.min(inhibs_psi), np.max(inhibs_psi))
    print("min max x_inhib", np.min(x_inhib), np.max(x_inhib))
    print("min max x_excit", np.min(x_excit), np.max(x_excit))
    # print("shape saliency", np.shape(saliency))
    print("min max saliency", np.min(saliency), np.max(saliency))

    print()
    print("x")
    print(x[:, :, idx])
    print("y")
    print(y[:, :, idx])
    print("gx")
    print(gx[:, :, idx])
    # print("gy")
    # print(gy[:, :, 6])
    print("excit")
    print(excits[:, :, idx])
    # print("inhibs_psi[0,0,0]", inhibs_psi[0,0,0])
    # print("force")
    # print(x_inhib[:, :, 6])  #modified x_inhib to be like the force
    # print("force excit")
    # print(x_excit[:, :, 6])  #modified x_excit to be like the force with excitation
    print("saliency")
    # print(saliency[:, :, 6])
    print(saliency[:, :, idx])


    max_column = 6

    num_filters = np.shape(I_i_theta)[-1]
    num_column = min(num_filters, max_column)
    num_row = math.ceil(num_filters / num_column)
    x_print = np.expand_dims(I_i_theta, axis=2)
    multi_frame = create_multi_frame(x_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "00_I_i_theta.jpeg", heatmap.astype(np.uint8))

    # save x activations
    num_filters = np.shape(x)[-1]
    num_column = min(num_filters, max_column)
    num_row = math.ceil(num_filters / num_column)
    x_print = np.expand_dims(x, axis=2)
    multi_frame = create_multi_frame(x_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "01_x.jpeg", heatmap.astype(np.uint8))

    # save gx activations
    gx_print = np.expand_dims(gx, axis=2)
    multi_frame = create_multi_frame(gx_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "02_gx(x)_response.jpeg", heatmap.astype(np.uint8))

    # save i_norm
    i_norm_print = np.expand_dims(i_norm, axis=2)
    i_norm_print = np.array(i_norm_print).astype(np.uint8)
    multi_frame = create_multi_frame(i_norm_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "03_i_norm.jpeg", heatmap.astype(np.uint8))

    # save inhibition
    inhibs_print = np.expand_dims(inhibs, axis=2)
    inhibs_print = np.array(inhibs_print).astype(np.uint8)
    num_filters = np.shape(inhibs)[-1]
    num_column = min(num_filters, max_column)
    num_row = math.ceil(num_filters / num_column)
    multi_frame = create_multi_frame(inhibs_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "04_inibition_response.jpeg", heatmap.astype(np.uint8))

    # save excitation
    excits_print = np.expand_dims(excits, axis=2)
    excits_print = np.array(excits_print).astype(np.uint8)
    multi_frame = create_multi_frame(excits_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "05_exitatory_response.jpeg", heatmap.astype(np.uint8))

    # print y neuronal response (inhibitory)
    y_print = np.expand_dims(y, axis=2)
    multi_frame = create_multi_frame(y_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "06_y_responses.jpeg", heatmap.astype(np.uint8))

    # print gy(y)
    gy_print = np.expand_dims(gy, axis=2)
    multi_frame = create_multi_frame(gy_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "07_gy(y)_response.jpeg", heatmap.astype(np.uint8))

    # save inhib psi
    inhibs_psi_print = np.expand_dims(inhibs_psi, axis=2)
    multi_frame = create_multi_frame(inhibs_psi_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "08_inibition_psi.jpeg", heatmap.astype(np.uint8))

    # plot inhib response
    x_inhib_print = np.expand_dims(x_inhib, axis=2)
    multi_frame = create_multi_frame(x_inhib_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "09_x_inhib_response.jpeg", heatmap.astype(np.uint8))

    # plot excit response
    x_excit_print = np.expand_dims(x_excit, axis=2)
    multi_frame = create_multi_frame(x_excit_print, num_row, num_column, img_size)
    heatmap = cv2.applyColorMap(multi_frame, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(path + "10_x_excit_response.jpeg", heatmap.astype(np.uint8))

    # plot saliency
    saliency = saliency - np.min(saliency)
    if np.max(saliency) != 0:
        saliency = saliency / np.max(saliency)
    saliency_map = np.expand_dims(saliency, axis=2)
    saliency_map = np.array(saliency_map * 255).astype(np.uint8)
    cv2.imwrite(path + "11_saliency_response.jpeg", saliency_map.astype(np.uint8))