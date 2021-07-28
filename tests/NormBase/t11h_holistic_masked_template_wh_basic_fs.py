import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.PatternFeatureReduction import PatternFeatureSelection
from utils.ref_feature_map_neurons import ref_feature_map_neuron
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output
from plots_utils.plot_ft_map_pos import plot_ft_map_pos
from plots_utils.plot_ft_map_pos import plot_pos_on_images
from models.NormBase import NormBase

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try the implementation of the holistic representation model on the BasicFaceShape dataset

run: python -m tests.NormBase.t11h_holistic_masked_template_wh_basic_fs
"""

# define configuration
config_path = 'NB_t11h_holistic_masked_template_wh_basic_fs_m0001.json'
compute_NB = True
plot_intermediate = False

train = True
test = True
plot = True


# declare parameters
best_eyebrow_IoU_ft = [68, 125]
best_lips_IoU_ft = [235, 203, 68, 125, 3, 181, 197, 2, 87, 240, 6, 95, 60, 157, 227, 111]

# load config
config = load_config(config_path, path='configs/norm_base_config')

# create directory if non existant
save_path = os.path.join("models/saved", config["config_name"])
if not os.path.exists(save_path):
    os.mkdir(save_path)

# load and define model
v4_model = load_extraction_model(config, input_shape=tuple(config["input_shape"]))
v4_model = tf.keras.Model(inputs=v4_model.input, outputs=v4_model.get_layer(config['v4_layer']).output)
size_ft = tuple(np.shape(v4_model.output)[1:3])
print("[LOAD] size_ft", size_ft)
print("[LOAD] Model loaded")
print()

nb_model = NormBase(config, tuple(config['input_shape']))

if train:
    # -------------------------------------------------------------------------------------------------------------------
    # train

    # load data
    data = load_data(config)

    # predict
    preds = v4_model.predict(data[0], verbose=1)
    print("[TRAIN] shape prediction", np.shape(preds))

    # get feature maps that mimic a semantic selection pipeline
    # keep only highest IoU semantic score
    eyebrow_preds = preds[..., best_eyebrow_IoU_ft]
    print("shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
    lips_preds = preds[..., best_lips_IoU_ft]
    print("shape lips semantic feature selection", np.shape(lips_preds))
    preds = np.concatenate((eyebrow_preds, lips_preds), axis=3)
    print("[TRAIN] shape preds", np.shape(preds))

    # add holistic templates
    rbf_template = [[[17, 20], [19, 22]], [[17, 20], [24, 27]], [[17, 20], [32, 35]], [[17, 20], [37, 40]],
                [[36, 39], [22, 25]], [[35, 38], [28, 32]], [[36, 39], [33, 36]], [[38, 40], [28, 32]]]

    rbf_mask = [[[13, 24], [15, 26]], [[13, 24], [20, 31]], [[13, 24], [28, 39]], [[13, 24], [33, 44]],
                [[35, 42], [20, 27]], [[32, 41], [25, 35]], [[34, 41], [31, 38]], [[36, 50], [24, 36]]]

    # config['rbf_sigma'] = [1800, 1800, 1800, 1800, 1800, 3300, 2200, 2200]  # loose left lips on angry (2) and right lip on fear (5)
    config['rbf_sigma'] = [1800, 1800, 1800, 1800, 2000, 3300, 2300, 2200]
    patterns = PatternFeatureSelection(config, template=rbf_template, mask=rbf_mask)

    # fit templates
    # template = patterns.fit(mask_template)
    template_preds = np.repeat(np.expand_dims(preds, axis=0), len(rbf_template), axis=0)
    template = patterns.fit(template_preds)
    template[template < 0.1] = 0

    # compute positions
    pos = calculate_position(template, mode="weighted average", return_mode="xy float flat")
    print("[TRAIN] shape pos", np.shape(pos))

    if compute_NB:
        nb_model.n_features = np.shape(pos)[-1]  # todo add this to init
        # train manually ref vector
        nb_model.r = np.zeros(nb_model.n_features)
        nb_model._fit_reference([pos, data[1]], config['batch_size'])
        print("[TRAIN] model.r", np.shape(nb_model.r))
        ref_train = np.copy(nb_model.r)
        # train manually tuning vector
        nb_model.t = np.zeros((nb_model.n_category, nb_model.n_features))
        nb_model.t_mean = np.zeros((nb_model.n_category, nb_model.n_features))
        nb_model._fit_tuning([pos, data[1]], config['batch_size'])
        ref_tuning = np.copy(nb_model.t)
        # get it resp
        it_train = nb_model._get_it_resp(pos)
        print("[TRAIN] shape it_train", np.shape(it_train))

        print()
        # print true labels versus the predicted label
        for i, label in enumerate(data[1]):
            t_label = int(label)
            p_label = np.argmax(it_train[i])

            if t_label == p_label:
                print(i, "true label:", t_label, "vs. pred:", p_label, " - OK")
            else:
                print(i, "true label:", t_label, "vs. pred:", p_label, " - wrong!")
        print()

if test:
    # -------------------------------------------------------------------------------------------------------------------
    # test monkey

    # load data
    test_data = load_data(config, train=False)

    # predict
    test_preds = v4_model.predict(test_data[0], verbose=1)
    print("[TEST] shape test_preds", np.shape(test_preds))

    # get feature maps that mimic a semantic selection pipeline
    # keep only highest IoU semantic score
    test_eyebrow_preds = test_preds[..., best_eyebrow_IoU_ft]
    test_lips_preds = test_preds[..., best_lips_IoU_ft]
    print("[TEST] shape eyebrow semantic feature selection", np.shape(test_eyebrow_preds))
    print("[TEST] shape lips semantic feature selection", np.shape(test_lips_preds))
    test_preds = np.concatenate((test_eyebrow_preds, test_lips_preds), axis=3)
    print("[TEST] shape test_preds", np.shape(test_preds))

    # add holistic templates
    test_rbf_template = [[[17, 20], [14, 17]], [[17, 20], [20, 23]], [[17, 20], [30, 33]], [[17, 20], [37, 40]],
                         [[40, 43], [19, 22]], [[39, 42], [26, 29]], [[40, 43], [32, 36]], [[41, 44], [25, 30]]]

    test_rbf_mask = [[[13, 24], [10, 21]], [[13, 24], [16, 27]], [[13, 24], [26, 37]], [[13, 24], [34, 44]],
                         [[36, 47], [15, 26]], [[35, 46], [22, 33]], [[37, 46], [28, 39]], [[40, 56], [21, 34]]]

    test_rbf_zeros = {'0': {'idx': 6, 'pos': [[40, 47], [36, 41]]}}

    config['rbf_sigma'] = [2500, 2800, 3000, 2900, 2200, 3600, 2600, 3700]
    test_patterns = PatternFeatureSelection(config, template=test_rbf_template, mask=test_rbf_mask, zeros=test_rbf_zeros)

    # fit templates
    test_template_preds = np.repeat(np.expand_dims(test_preds, axis=0), len(test_rbf_template), axis=0)
    test_template = test_patterns.fit(test_template_preds)
    test_template[test_template < 0.1] = 0

    # compute positions
    test_pos = calculate_position(test_template, mode="weighted average", return_mode="xy float flat")
    print("[TEST] shape test_pos", np.shape(test_pos))

    if plot_intermediate:
        plot_cnn_output(test_template, os.path.join("models/saved", config["config_name"]),
                        "00_test_template.gif", verbose=True, video=True)

    if compute_NB:
        # get IT responses of the model
        test_it = nb_model._get_it_resp(test_pos)

        # test by training new ref
        nb_model._fit_reference([test_pos, test_data[1]], config['batch_size'])
        ref_test = np.copy(nb_model.r)
        it_ref_test = nb_model._get_it_resp(test_pos)

        print()
        # print true labels versus the predicted label
        for i, label in enumerate(test_data[1]):
            t_label = int(label)
            p_label = np.argmax(it_ref_test[i])

            if t_label == p_label:
                print(i, "true label:", t_label, "vs. pred:", p_label, " - OK")
            else:
                print(i, "true label:", t_label, "vs. pred:", p_label, " - wrong!")
        print()

if plot:
    # --------------------------------------------------------------------------------------------------------------------
    # plots
    print("[PLOT] shape preds", np.shape(preds))
    # print("[PLOT] shape test_preds", np.shape(test_preds))

    # build arrows
    arrow_tail = np.repeat(np.expand_dims(np.reshape(ref_train, (-1, 2)), axis=0), config['n_category'], axis=0)
    arrow_head = np.reshape(ref_tuning, (len(ref_tuning), -1, 2))
    arrows = [arrow_tail, arrow_head]
    # arrows_color = ['#0e3957', '#3b528b', '#21918c', '#5ec962', '#fde725']

    # put one color per label
    labels = data[1]
    color_seq = np.zeros(len(labels))
    color_seq[labels == 1] = 1
    color_seq[labels == 2] = 2
    color_seq[labels == 3] = 3
    color_seq[labels == 4] = 4
    color_seq[labels == 5] = 5
    color_seq[labels == 6] = 6

    pos_2d = np.reshape(pos, (len(pos), -1, 2))
    print("[PLOT] shape pos_flat", np.shape(pos_2d))
    plot_ft_map_pos(pos_2d,
                    fig_name="00b_human_train_pos.png",
                    path=os.path.join("models/saved", config["config_name"]),
                    color_seq=color_seq,
                    arrows=arrows)
                    # arrows_color=arrows_color)

    ref_pos = np.repeat(np.expand_dims(pos_2d[0], axis=0), len(pos_2d), axis=0)
    plot_pos_on_images(pos_2d, data[0],
                       fig_name="00c_human_train.png",
                       save_folder=os.path.join("models/saved", config["config_name"]),
                       ref_pos=ref_pos,
                       ft_size=(56, 56))

    # build arrows
    arrow_tail = np.repeat(np.expand_dims(np.reshape(ref_test, (-1, 2)), axis=0), config['n_category'], axis=0)
    arrow_head = np.reshape(ref_tuning, (len(ref_tuning), -1, 2))
    arrows = [arrow_tail, arrow_head]
    # arrows_color = ['#0e3957', '#3b528b', '#21918c', '#5ec962', '#fde725']

    # put one color per label
    test_labels = test_data[1]
    color_seq = np.zeros(len(test_labels))
    color_seq[test_labels == 1] = 1
    color_seq[test_labels == 2] = 2
    color_seq[test_labels == 3] = 3
    color_seq[test_labels == 4] = 4

    test_pos_2d = np.reshape(test_pos, (len(test_pos), -1, 2))
    plot_ft_map_pos(test_pos_2d,
                    fig_name="00b_monkey_test_pos.png",
                    path=os.path.join("models/saved", config["config_name"]),
                    color_seq=color_seq,
                    arrows=arrows)
                    # arrows_color=arrows_color)

    ref_test_pos = np.repeat(np.expand_dims(test_pos_2d[0], axis=0), len(test_pos_2d), axis=0)
    plot_pos_on_images(test_pos_2d, test_data[0],
                       fig_name="00c_test.png",
                       save_folder=os.path.join("models/saved", config["config_name"]),
                       ref_pos=ref_test_pos,
                       ft_size=(56, 56))

    # ***********************       test 01 model     ******************
    # plot it responses for eyebrow model
    nb_model.plot_it_neurons(it_train,
                             title="01_it_train",
                             save_folder=os.path.join("models/saved", config["config_name"]))
    nb_model.plot_it_neurons(it_ref_test,
                             title="01_it_ref_test",
                             save_folder=os.path.join("models/saved", config["config_name"]))


    print("finished plotting test2")
    print()
