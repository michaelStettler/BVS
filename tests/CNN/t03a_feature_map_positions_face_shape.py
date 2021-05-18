import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.feat_map_filter_processing import get_feat_map_filt_preds
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the computations of feature positions within a feature map

run: python -m tests.CNN.t03a_feature_map_positions_face_shape
"""

# define configuration
config_path = 'CNN_t03a_feature_map_positions_face_shape_m0001.json'

# declare parameters
eyebrow_ft_idx = [148, 209, 208, 67, 211, 141, 90, 196, 174, 179, 59, 101, 225, 124, 125, 156]  # from t02_find_semantic_units
lips_ft_idx = [79, 120, 125, 0, 174, 201, 193, 247, 77, 249, 210, 149, 89, 197, 9, 251, 237, 165, 101, 90, 27, 158, 154, 10, 168, 156, 44, 23, 34, 85, 207]
ft_idx = [eyebrow_ft_idx, lips_ft_idx]
slice_pos_eyebrow = 9
slice_pos_lips = 13

# load config
config = load_config(config_path, path='configs/CNN')

# create directory if non existant
save_path = os.path.join("models/saved", config["config_name"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

# load and define model
model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(model.output)[1:3])
print("[LOAD] size_ft", size_ft)
print("[LOAD] Model loaded")
print()


def filter_feature_maps(preds):
    # filter feat maps
    preds_cleaned = np.copy(preds)
    for i in range(len(preds)):
        for x in range(1, np.shape(preds)[1] - 1):
            for y in range(1, np.shape(preds)[2] - 1):
                median = np.median(preds[i, x-1:x+2, y-1:y+2], axis=[0, 1])
                preds_cleaned[i, x, y] = median

    return preds_cleaned


def predict_expression(expression, ft_idx, test_num=None):
    # set config
    config["train_expression"] = [expression]

    # -----------------------------------------------------------------------------------------------------------------
    # human avatar
    config["train_avatar"] = "human_orig"

    # load data
    data = load_data(config)[0]

    # predict and filter responses
    print("[PE] Human {} loaded".format(expression))
    print("[PE] shape data", np.shape(data))
    print("[PE] Start predictions")
    preds_human = model.predict(data)
    preds_human = get_feat_map_filt_preds(preds_human,
                                          ft_idx,
                                          ref_type="self0",
                                          norm=1000,
                                          activation='ReLu',
                                          filter='spatial_mean',
                                          verbose=True)
    print("[PE] shape preds Human", np.shape(preds_human))
    print("[PE] min max preds Human", np.amin(preds_human), np.amax(preds_human))

    # get xy positions
    preds_human_pos = calculate_position(preds_human, mode="weighted average", return_mode="xy float")
    print("[PE] shape preds Human positions", np.shape(preds_human_pos))

    # # filter feat maps
    # preds_human_cleaned = filter_feature_maps(preds_human)
    # preds_human_pos_cleaned = calculate_position(preds_human_cleaned, mode="weighted average", return_mode="xy float")

    # -----------------------------------------------------------------------------------------------------------------
    # Monkey Avatar
    config["train_avatar"] = "monkey_orig"

    # load data
    data = load_data(config)[0]
    print("[PE] Monkey {} loaded".format(expression))
    print("[PE] shape data", np.shape(data))
    print("[PE] Start predictions")
    preds_monkey = model.predict(data)
    preds_monkey = get_feat_map_filt_preds(preds_monkey,
                                           ft_idx,
                                           ref_type="self0",
                                           norm=1000,
                                           activation='ReLu',
                                           filter='spatial_mean',
                                           verbose=True)
    print("[PE] shape preds Human", np.shape(preds_monkey))
    print("[PE] min max preds Human", np.amin(preds_monkey), np.amax(preds_monkey))

    # get xy positions
    preds_monkey_pos = calculate_position(preds_monkey, mode="weighted average", return_mode="xy float")
    print("[PE] shape preds Monkey positions", np.shape(preds_monkey_pos))

    # # filter feat maps
    # preds_monkey_cleaned = filter_feature_maps(preds_monkey)
    # preds_monkey_pos_cleaned = calculate_position(preds_monkey_cleaned, mode="weighted average", return_mode="xy float")

    # -----------------------------------------------------------------------------------------------------------------
    # plot raw feature maps
    print("[PE] Create plot")

    # plot human
    cnn_output_name = 'test_human_cnn_output.gif'
    if test_num is not None:
        cnn_output_name = 'test{}_human_cnn_output.gif'.format(test_num)

    plot_cnn_output(preds_human, save_path, cnn_output_name, video=True)

    # # plot human filtered feature maps
    # cnn_output_name = 'test_human_filtered_cnn_output.gif'
    # if test_num is not None:
    #     cnn_output_name = 'test{}_human_filtered_cnn_output.gif'.format(test_num)
    #
    # plot_cnn_output(preds_human_cleaned, save_path, cnn_output_name, video=True)

    # plot monkey
    cnn_output_name = 'test_monkey_cnn_output.gif'
    if test_num is not None:
        cnn_output_name = 'test{}_monkey_cnn_output.gif'.format(test_num)

    plot_cnn_output(preds_monkey, save_path, cnn_output_name, video=True)

    # # plot monkey cleaned
    # cnn_output_name = 'test_monkey_filtered_cnn_output.gif'
    # if test_num is not None:
    #     cnn_output_name = 'test{}_monkey_filtered_cnn_output.gif'.format(test_num)
    #
    # plot_cnn_output(preds_monkey_cleaned, save_path, cnn_output_name, video=True)

    # plot slice predictions
    # plot raw responses of the dynamic
    max_pred = np.amax([preds_human, preds_monkey])
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(preds_human[:, :, slice_pos_eyebrow, 0])  # slice over the 10 column to try to get the eyebrow
    plt.ylim(0, max_pred)
    plt.title("Human Avatar Eyebrow")
    plt.subplot(2, 2, 2)
    plt.plot(preds_monkey[:, :, slice_pos_eyebrow, 0])  # slice over the 10 column to try to get the eyebrow
    plt.ylim(0, max_pred)
    plt.title("Monkey Avatar Eyebrow")
    plt.subplot(2, 2, 3)
    plt.plot(preds_human[:, :, slice_pos_lips, 1])  # slice over the 10 column to try to get the lips
    plt.ylim(0, max_pred)
    plt.title("Human Avatar Lips")
    plt.subplot(2, 2, 4)
    plt.plot(preds_monkey[:, :, slice_pos_lips, 1])  # slice over the 10 column to try to get the lips
    plt.ylim(0, max_pred)
    plt.title("Monkey Avatar Lips")
    plt.suptitle(expression)

    if test_num is not None:
        plt.savefig(os.path.join(save_path, "test{}_Hum_vs_Monk_{}_expression_slice_eb{}_lips_eb{}".format(test_num,
                                                                                                           expression,
                                                                                                           slice_pos_eyebrow,
                                                                                                           slice_pos_lips)))
    else:
        plt.savefig(os.path.join(save_path, "test_Hum_vs_Monk_{}_expression_slice_eb{}_lips_eb{}".format(expression,
                                                                                                      slice_pos_eyebrow,
                                                                                                      slice_pos_lips)))

    # plot positions
    # set color to represent time
    color_seq = np.arange(len(preds_human_pos))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(preds_human_pos[:, 1, 0], preds_human_pos[:, 0, 0], c=color_seq)
    plt.xlim(12, 15)
    plt.colorbar()
    plt.title("Human Avatar Eyebrow")
    plt.subplot(2, 2, 2)
    plt.scatter(preds_monkey_pos[:, 1, 0], preds_monkey_pos[:, 0, 0], c=color_seq)
    plt.xlim(12, 15)
    plt.colorbar()
    plt.title("Monkey Avatar Eyebrow")
    plt.subplot(2, 2, 3)
    plt.scatter(preds_human_pos[:, 1, 1], preds_human_pos[:, 0, 1], c=color_seq)
    plt.xlim(12, 15)
    plt.colorbar()
    plt.title("Human Avatar Lips")
    plt.subplot(2, 2, 4)
    plt.scatter(preds_monkey_pos[:, 1, 1], preds_monkey_pos[:, 0, 1], c=color_seq)
    plt.xlim(12, 15)
    plt.colorbar()
    plt.title("Monkey Avatar Lips")

    plt.suptitle(expression + " xy-pos")

    if test_num is not None:
        plt.savefig(os.path.join(save_path, "test{}_Hum_vs_Monk_{}_expression_pos_eb{}_lips_eb{}".format(test_num,
                                                                                                         expression,
                                                                                                         slice_pos_eyebrow,
                                                                                                         slice_pos_lips)))
    else:
        plt.savefig(os.path.join(save_path, "test_Hum_vs_Monk_{}_expression_pos_eb{}_lips_eb{}".format(expression,
                                                                                                       slice_pos_eyebrow,
                                                                                                       slice_pos_lips)))

    # # plot positions cleaned
    # # set color to represent time
    # color_seq = np.arange(len(preds_human_pos))
    #
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.scatter(preds_human_pos_cleaned[:, 1, 0], preds_human_pos_cleaned[:, 0, 0], c=color_seq)
    # plt.xlim(12, 15)
    # plt.colorbar()
    # plt.title("Human Avatar Eyebrow")
    # plt.subplot(2, 2, 2)
    # plt.scatter(preds_monkey_pos_cleaned[:, 1, 0], preds_monkey_pos_cleaned[:, 0, 0], c=color_seq)
    # plt.xlim(12, 15)
    # plt.colorbar()
    # plt.title("Monkey Avatar Eyebrow")
    # plt.subplot(2, 2, 3)
    # plt.scatter(preds_human_pos_cleaned[:, 1, 1], preds_human_pos_cleaned[:, 0, 1], c=color_seq)
    # plt.xlim(12, 15)
    # plt.colorbar()
    # plt.title("Human Avatar Lips")
    # plt.subplot(2, 2, 4)
    # plt.scatter(preds_monkey_pos_cleaned[:, 1, 1], preds_monkey_pos_cleaned[:, 0, 1], c=color_seq)
    # plt.xlim(12, 15)
    # plt.colorbar()
    # plt.title("Monkey Avatar Lips")
    #
    # plt.suptitle(expression + " xy-pos")
    #
    # if test_num is not None:
    #     plt.savefig(os.path.join(save_path, "test{}_Hum_vs_Monk_{}_expression_pos_filtered_eb{}_lips_eb{}".format(test_num,
    #                                                                                                      expression,
    #                                                                                                      slice_pos_eyebrow,
    #                                                                                                      slice_pos_lips)))
    # else:
    #     plt.savefig(os.path.join(save_path, "test_Hum_vs_Monk_{}_expression_pos_eb{}_filtered_lips_eb{}".format(expression,
    #                                                                                                    slice_pos_eyebrow,
    #                                                                                                    slice_pos_lips)))

# # --------------------------------------------------------------------------------------------------------------------
# # test 1 - compare cnn eye_brow units responses for expression C1 over Human and Monkey avatar
# print("[TEST 1] Compare eyebrow feat. map for C1 across Human and Monkey avatar")
# predict_expression("c1", ft_idx, test_num="1")
# print()

# --------------------------------------------------------------------------------------------------------------------
# test 2 - compare cnn eye_brow units responses for expression C2 over Human and Monkey avatar
print("[TEST 2] Compare eyebrow feat. map for C2 across Human and Monkey avatar")
predict_expression("c2", ft_idx, test_num="2")
print()
#
# # --------------------------------------------------------------------------------------------------------------------
# # test 3 - compare cnn eye_brow units responses for expression C3 over Human and Monkey avatar
# print("[TEST 3] Compare eyebrow feat. map for C3 across Human and Monkey avatar")
# predict_expression("c3", ft_idx, test_num="3")
# print()
#
# # --------------------------------------------------------------------------------------------------------------------
# # test 4 - compare cnn eye_brow units responses for expression C4 over Human and Monkey avatar
# print("[TEST 4] Compare eyebrow feat. map for C4 across Human and Monkey avatar")
# predict_expression("c4", ft_idx, test_num="4")
# print()