import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_from_csv import load_from_csv
from utils.CNN.extraction_model import load_extraction_model

plt.style.use('ggplot')

"""
run: python -m tests.CNN.t01_cnn_noise_variability
"""


def test_features_map_variability(preds, sequence, condition="", linewidth=.5):
    # test variability across the mouth region
    plt.figure()
    plt.plot(preds[:, :, 14, 0], linewidth=linewidth)  # middle columns of first feature map
    plt.title("1) Mouth region 1st feature map")
    plt.savefig(os.path.join("models/saved", config["config_name"], condition + "_feat_map_mouth_region_fm1.png"))

    # test variability at the 4 corners
    plt.figure()
    plt.plot(preds[:, 0, 0, 0], label="top-left", linewidth=linewidth)  # top left corner of first feature map
    plt.plot(preds[:, 0, -1, 0], label="top-right", linewidth=linewidth)  # top right corner of first feature map
    plt.plot(preds[:, -1, 0, 0], label="bottom-left", linewidth=linewidth)  # bottom left corner of first feature map
    plt.plot(preds[:, -1, -1, 0], label="bottom-right", linewidth=linewidth)  # bottom right corner of first feature map
    plt.title("2) 4 corners region of 1st feature map")
    plt.ylim([0, 4000])
    plt.legend()
    plt.savefig(os.path.join("models/saved", config["config_name"],  condition + "_feat_map_4_corners_fm1.png"))

    # plot image bottom right corner compared to activity at bottom right
    print()
    plt.figure()
    v4_bottom_right_norm = preds[:, -1, -1, 0]
    img_bottom_right_norm = sequence[:, -1, -1, 0]
    plt.plot(v4_bottom_right_norm, label="bottom-right")  # bottom right corner of first feature map
    plt.plot(img_bottom_right_norm, label="image r-pixel value")
    plt.title("3) right corner feature vs. pixel of 1st feature map")
    plt.legend()
    plt.savefig(os.path.join("models/saved", config["config_name"],  condition + "_feat_map_bottom_left_vs_red_fm1.png"))

    # compute variance
    preds_var = np.std(preds[:, -1, -1, 0]) / np.mean(preds[:, -1, -1, 0])
    img_pixel_var = np.std(sequence[:, -1, -1, 0]) / np.mean(sequence[:, -1, -1, 0])
    print("[{}] std/mean of v4 bottom-right response: {:.2f}%".format(condition, preds_var * 100))
    print("[{}] std/mean of r-pixel bottom-right response: {:.2}%".format(condition, img_pixel_var * 100))

    # plot image middle compared to activity at bottom right
    print()
    plt.figure()
    v4_bottom_right_norm = preds[:, 14, 14, 0]
    img_bottom_right_norm = sequence[:, 112, 112, 0]
    plt.plot(v4_bottom_right_norm, label="bottom-right")  # bottom right corner of first feature map
    plt.plot(img_bottom_right_norm, label="image r-pixel value")
    plt.title("4) middle feature vs. pixel of 1st feature map")
    plt.legend()
    plt.savefig(os.path.join("models/saved", config["config_name"],  condition + "_feat_map_middle_vs_red_fm1.png"))

    # compute variance
    preds_var = np.std(preds[:, 14, 14, 0]) / np.mean(preds[:, 14, 14, 0])
    img_pixel_var = np.std(sequence[:, 112, 112, 0]) / np.mean(sequence[:, 112, 112, 0])
    print("[{}] std/mean of v4 bottom-right response: {:.2f}%".format(condition, preds_var * 100))
    print("[{}] std/mean of r-pixel bottom-right response: {:.2f}%".format(condition, img_pixel_var * 100))

    # plot image middle pixel 50 first feature map
    print()
    plt.figure()
    x_pos = 9
    y_pos = 14
    ft_middle_pixel = preds[:, x_pos, y_pos, :50]
    plt.plot(ft_middle_pixel, linewidth=linewidth)
    plt.title("5) middle pixel across 50 first feature map")
    plt.legend()
    plt.savefig(os.path.join("models/saved", config["config_name"],  condition + "_pixel_({}, {})_across_50fm.png".format(x_pos, y_pos)))


if __name__ == '__main__':
    # config_name = 'CNN_t01_human_c2_m0001.json'
    # # config_name = 'CNN_t01_human_c3_m0001.json'
    # # config_name = 'CNN_t01_monkey_c2_m0001.json'
    # # config_name = 'CNN_t01_monkey_c3_m0001.json'
    # config = load_config(config_name, path='configs/CNN')
    #
    # if not os.path.exists(os.path.join("models/saved", config["config_name"])):
    #     os.mkdir(os.path.join("models/saved", config["config_name"]))
    #
    # # load data
    # data = load_data(config, train=True)
    # raw_data = load_data(config, train=True, get_raw=True)[0]
    #
    # sequence = np.array(data[0])
    # print("shape sequence", sequence.shape)
    # print("min max sequence", np.amin(sequence), np.amax(sequence))

    # # ------------------------------------------------------------------------------------------------------------------
    # # test 1: plot variability of the features map at the layer block3_pool
    # # declare model
    # model = load_extraction_model(config, tuple(config['input_shape']))
    # model = tf.keras.Model(inputs=model.input,
    #                        outputs=model.get_layer(config['v4_layer']).output)
    #
    # # predict model
    # preds = model.predict(data)
    # print("shape preds", np.shape(preds))
    #
    # # plot feature maps
    # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    #                 config['v4_layer'] + ".gif",
    #                 image=raw_data,
    #                 video=True)
    #
    # # test model
    # test_features_map_variability(preds, sequence, "01_" + config["v4_layer"])
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # test 2: plot variability of the features map at the layer block1_conv1
    # config["v4_layer"] = "block1_conv1"
    # # declare model
    # model = load_extraction_model(config, tuple(config['input_shape']))
    # model = tf.keras.Model(inputs=model.input,
    #                        outputs=model.get_layer(config['v4_layer']).output)
    #
    # # predict model
    # preds = model.predict(data)
    # print("shape preds", np.shape(preds))
    #
    # # # plot feature maps
    # # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    # #                 config['v4_layer'] + ".gif",
    # #                 image=raw_data,
    # #                 video=True)
    #
    # # test model
    # test_features_map_variability(preds, sequence, "02_" + config["v4_layer"])
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # # test 3: plot variability of the features map at the layer block3_pool with constant sequence
    # # # create constant sequence
    # # cste_sequence = np.ones(np.shape(data[0]))  # set to one so I don't divide by zero
    # # cste_sequence[50:100, 82:142, 82:142, :] = 120
    # # data[0] = cste_sequence
    # # raw_data = cste_sequence + 125  # todo add the true rgb mean from imagenet
    # #
    # # print("shape cste_sequence", np.shape(cste_sequence))
    # # config["v4_layer"] = "block3_pool"
    # #
    # # # declare model
    # # model = load_extraction_model(config, tuple(config['input_shape']))
    # # model = tf.keras.Model(inputs=model.input,
    # #                        outputs=model.get_layer(config['v4_layer']).output)
    # #
    # # # predict model
    # # preds = model.predict(data)
    # # print("shape preds", np.shape(preds))
    # # print("max preds[:, -1, -1, 0]", np.max(preds[:, -1, -1, 0]))
    # #
    # # # # plot feature maps
    # # # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    # # #                 config['v4_layer'] + ".gif",
    # # #                 image=raw_data,
    # # #                 video=True)
    # #
    # # # test model
    # # test_features_map_variability(preds, raw_data, config["v4_layer"] + '_cste')
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # test 4: plot variability of the features map at the layer block3_pool with small variations sequence
    # # create variable (noisy) sequence
    # shape_data = np.shape(data[0])
    # variation_sequence = np.random.rand(shape_data[0], shape_data[1], shape_data[2], shape_data[3]) * 3
    # variation_sequence[50:100, 82:142, 82:142, :] += 120
    # data[0] = variation_sequence
    # raw_data = variation_sequence + 125  # todo add the true rgb mean from imagenet
    #
    # print("shape cste_sequence", np.shape(variation_sequence))
    # config["v4_layer"] = "block3_pool"
    #
    # # declare model
    # model = load_extraction_model(config, tuple(config['input_shape']))
    # model = tf.keras.Model(inputs=model.input,
    #                        outputs=model.get_layer(config['v4_layer']).output)
    #
    # # predict model
    # preds = model.predict(data)
    # print("shape preds", np.shape(preds))
    # print("max preds[:, -1, -1, 0]", np.max(preds[:, -1, -1, 0]))
    #
    # # # plot feature maps
    # # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    # #                 config['v4_layer'] + ".gif",
    # #                 image=raw_data,
    # #                 video=True)
    #
    # # test model
    # test_features_map_variability(preds, raw_data, config["v4_layer"] + '_variations')
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # test 5: plot amplification of the variability of the features map at between layer conv1 and block3_pool
    # # plot between "block1_conv1" and "block3_pool"
    #
    # print("shape cste_sequence", np.shape(variation_sequence))
    # config["v4_layer"] = "block1_conv1"
    #
    # # declare model
    # model1 = load_extraction_model(config, tuple(config['input_shape']))
    # model1 = tf.keras.Model(inputs=model.input,
    #                        outputs=model.get_layer(config['v4_layer']).output)
    #
    #
    # config["v4_layer"] = "block3_pool"
    # model2 = load_extraction_model(config, tuple(config['input_shape']))
    # model2 = tf.keras.Model(inputs=model.input,
    #                        outputs=model.get_layer(config['v4_layer']).output)
    #
    # # predict model
    # preds1 = model1.predict(data)
    # preds2 = model2.predict(data)
    #
    # # plot
    # plt.figure()
    # plt.plot(preds1[:, -1, -1, 0], label="conv1")
    # plt.plot(preds2[:, -1, -1, 0], label="block3_pool")
    # plt.plot(raw_data[:, -1, -1, 0], label="pixel")
    # plt.legend()
    # plt.title("amplification conv1 vs. block3_pool")
    # plt.savefig(os.path.join("models/saved", config["config_name"], "05_amplification_fm1.png"))

    # ------------------------------------------------------------------------------------------------------------------
    # test 6: test martin image
    import pandas as pd

    config_name = 'CNN_t01_martin_test_m0001.json'
    config = load_config(config_name, path='configs/CNN')

    if not os.path.exists(os.path.join("models/saved", config["config_name"])):
        os.mkdir(os.path.join("models/saved", config["config_name"]))

    # load csv
    df = pd.read_csv(config['csv'], index_col=0)
    # df["category"] = df["category"].astype(int)

    # get only human c2 = category 1
    category = "1"
    df = df[df['category'].isin([category])]

    # load data
    data = load_from_csv(df, config)
    print("shape data", np.shape(data[0]))

    sequence = np.copy(data[0])
    data[0] = tf.keras.applications.vgg19.preprocess_input(data[0])

    # declare model
    config["v4_layer"] = "block3_pool"
    model = load_extraction_model(config, tuple(config['input_shape']))
    model = tf.keras.Model(inputs=model.input,
                           outputs=model.get_layer(config['v4_layer']).output)

    # predict model
    preds = model.predict(data)
    print("shape preds", np.shape(preds))

    # # plot feature maps
    # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    #                 config['v4_layer'] + ".gif",
    #                 image=raw_data,
    #                 video=True)

    # test model
    test_features_map_variability(preds, sequence, "06_" + config["v4_layer"])