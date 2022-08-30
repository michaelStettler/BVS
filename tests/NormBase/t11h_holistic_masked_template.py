import os
import numpy as np
import tensorflow as tf

from utils.load_config import load_config
from utils.load_data import load_data
from utils.extraction_model import load_extraction_model
from utils.remove_transition_morph_space import remove_transition_frames
from utils.PatternFeatureReduction import PatternFeatureSelection
from utils.ref_feature_map_neurons import ref_feature_map_neuron
from utils.calculate_position import calculate_position
from plots_utils.plot_cnn_output import plot_cnn_output
from plots_utils.plot_ft_map_pos import plot_ft_map_pos
from plots_utils.plot_ft_map_pos import plot_ft_pos_on_sequence
from models.NormBase import NormBase

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

"""
test script to try an implementation of a holistic representation model by a RBF_patch_pattern function of the face
it uses mask and "zeros" to create receptive field around the templates


run: python -m tests.NormBase.t11h_holistic_masked_template
"""

# define configuration
config_path = 'NB_t11h_holistic_masked_template_m0005.json'
plot_intermediate = False
compute_NB = True

train = True
human_full = True
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

    # remove transition frames
    data = remove_transition_frames(data)

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

    config['rbf_sigma'] = [1800, 1800, 1800, 1800, 1800, 3300, 2200, 2200]
    patterns = PatternFeatureSelection(config, template=rbf_template, mask=rbf_mask)

    # fit templates
    # template = patterns.fit(mask_template)
    template_preds = np.repeat(np.expand_dims(preds, axis=0), len(rbf_template), axis=0)
    template = patterns.fit(template_preds)
    template[template < 0.1] = 0

    # compute positions
    pos = calculate_position(template, mode="weighted average", return_mode="xy float flat")
    print("[TRAIN] shape pos", np.shape(pos))

    if plot_intermediate:
        plot_cnn_output(template, os.path.join("models/saved", config["config_name"]),
                        "00_template.gif", verbose=True, video=True)

        test_pos_2d = np.reshape(pos, (len(pos), -1, 2))
        plot_ft_map_pos(test_pos_2d,
                        fig_name="00b_human_pos.png",
                        path=os.path.join("models/saved", config["config_name"]))

        test_max_preds = np.expand_dims(np.amax(preds, axis=3), axis=3)
        preds_plot = test_max_preds / np.amax(test_max_preds) * 255
        print("[TRAIN] shape preds_plot", np.shape(preds_plot))
        plot_ft_pos_on_sequence(pos, preds_plot, vid_name='00_ft_pos.mp4',
                                save_folder=os.path.join("models/saved", config["config_name"]),
                                pre_proc='raw', ft_size=(56, 56))

        plot_ft_pos_on_sequence(pos, data[0],
                                vid_name='00_ft_pos_human.mp4',
                                save_folder=os.path.join("models/saved", config["config_name"]),
                                lmk_size=1, ft_size=(56, 56))

    if compute_NB:
        nb_model.n_features = np.shape(pos)[-1]  # todo add this to init
        # train manually ref vector
        nb_model.r = np.zeros(nb_model.n_features)
        nb_model._fit_reference([pos, data[1]], config['batch_size'])
        print("[TRAIN] model.r", np.shape(nb_model.r))
        print(nb_model.r)
        ref_train = np.copy(nb_model.r)
        # train manually tuning vector
        nb_model.t = np.zeros((nb_model.n_category, nb_model.n_features))
        nb_model.t_mean = np.zeros((nb_model.n_category, nb_model.n_features))
        nb_model._fit_tuning([pos, data[1]], config['batch_size'])
        ref_tuning = np.copy(nb_model.t)
        print("[TRAIN] ref_tuning[1]")
        print(ref_tuning[1])
        # get it resp
        it_train = nb_model._get_it_resp(pos)
        print("[TRAIN] shape it_train", np.shape(it_train))

if human_full:
    # -------------------------------------------------------------------------------------------------------------------
    # test Full human
    # load data
    data = load_data(config)
    # predict
    preds = v4_model.predict(data[0], verbose=1)
    print("[PRED] shape prediction", np.shape(preds))

    # get feature maps that mimic a semantic selection pipeline
    # keep only highest IoU semantic score
    eyebrow_preds = preds[..., best_eyebrow_IoU_ft]
    print("[PRED] shape eyebrow semantic feature selection", np.shape(eyebrow_preds))
    lips_preds = preds[..., best_lips_IoU_ft]
    print("[PRED] shape lips semantic feature selection", np.shape(lips_preds))
    preds = np.concatenate((eyebrow_preds, lips_preds), axis=3)
    print("[PRED] shape preds", np.shape(preds))

    # compute templates
    # template = patterns.transform(mask_template)  # (150, 56, 56, 8)
    template_preds = np.repeat(np.expand_dims(preds, axis=0), len(rbf_template), axis=0)
    template = patterns.transform(template_preds)
    print("!!!!!!!!!!!!!!!!!!!!!!!")
    print("shape template", np.shape(template))
    template[template < 0.1] = 0

    # compute positions
    pos = calculate_position(template, mode="weighted average", return_mode="xy float flat")
    print("[PRED] shape pos", np.shape(pos))

    if compute_NB:
        # get it resp for eyebrows
        it_train = nb_model._get_it_resp(pos)
        print("[PRED] shape it_train", np.shape(it_train))


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
    test_rbf_template = [[[10, 13], [19, 22]], [[11, 14], [24, 27]], [[11, 14], [29, 32]], [[10, 13], [35, 38]],
                         [[34, 37], [20, 23]], [[32, 35], [27, 30]], [[34, 37], [33, 36]], [[36, 39], [27, 30]]]

    test_rbf_mask = [[[6, 17], [15, 26]], [[7, 18], [20, 31]], [[7, 18], [25, 36]], [[6, 17], [31, 42]],
                     [[32, 49], [16, 27]], [[29, 38], [24, 33]], [[32, 49], [29, 40]], [[34, 52], [25, 32]]]

    test_rbf_zeros = {'0': {'idx': 4, 'pos': [[40, 50], [16, 21]]},
                      '1': {'idx': 4, 'pos': [[29, 35], [23, 27]]},
                      '2': {'idx': 6, 'pos': [[40, 50], [34, 40]]},
                      '3': {'idx': 6, 'pos': [[29, 35], [29, 35]]}}

    config['rbf_sigma'] = [1800, 1400, 1400, 1800, 1900, 1900, 1900, 2000]
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

        test_pos_2d = np.reshape(test_pos, (len(test_pos), -1, 2))
        plot_ft_map_pos(test_pos_2d,
                        fig_name="00b_monkey_pos.png",
                        path=os.path.join("models/saved", config["config_name"]))

        test_max_preds = np.expand_dims(np.amax(test_preds, axis=3), axis=3)
        test_preds_plot = test_max_preds / np.amax(test_max_preds) * 255
        print("[TEST] shape test_preds_plot", np.shape(test_preds_plot))
        plot_ft_pos_on_sequence(test_pos, test_preds_plot, vid_name='00_ft_test_pos.mp4',
                                save_folder=os.path.join("models/saved", config["config_name"]),
                                pre_proc='raw', ft_size=(56, 56))

        plot_ft_pos_on_sequence(test_pos, test_data[0],
                                vid_name='00_ft_pos_monkey.mp4',
                                save_folder=os.path.join("models/saved", config["config_name"]),
                                lmk_size=1, ft_size=(56, 56))

    if compute_NB:
        # get IT responses of the model
        test_it = nb_model._get_it_resp(test_pos)

        # test by training new ref
        nb_model._fit_reference([test_pos, test_data[1]], config['batch_size'])
        ref_test = np.copy(nb_model.r)
        it_ref_test = nb_model._get_it_resp(test_pos)

if plot:
    # --------------------------------------------------------------------------------------------------------------------
    # plots
    print("[PLOT] shape preds", np.shape(preds))
    print("[PLOT] shape test_preds", np.shape(test_preds))

    # build arrows
    arrow_tail = np.repeat(np.expand_dims(np.reshape(ref_train, (-1, 2)), axis=0), config['n_category'], axis=0)
    arrow_head = np.reshape(ref_tuning, (len(ref_tuning), -1, 2))
    arrows = [arrow_tail, arrow_head]
    arrows_color = ['#0e3957', '#3b528b', '#21918c', '#5ec962', '#fde725']

    # put one color per label
    labels = data[1]
    color_seq = np.zeros(len(labels))
    color_seq[labels == 1] = 1
    color_seq[labels == 2] = 2
    color_seq[labels == 3] = 3
    color_seq[labels == 4] = 4

    pos_2d = np.reshape(pos, (len(pos), -1, 2))
    print("[PLOT] shape pos_2d", np.shape(pos_2d))
    plot_ft_map_pos(pos_2d,
                    fig_name="00b_human_train_pos.png",
                    path=os.path.join("models/saved", config["config_name"]),
                    color_seq=color_seq,
                    arrows=arrows,
                    arrows_color=arrows_color)

    # build arrows
    arrow_tail = np.repeat(np.expand_dims(np.reshape(ref_test, (-1, 2)), axis=0), config['n_category'], axis=0)
    arrow_head = np.reshape(ref_tuning, (len(ref_tuning), -1, 2))
    arrows = [arrow_tail, arrow_head]
    arrows_color = ['#0e3957', '#3b528b', '#21918c', '#5ec962', '#fde725']

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
                    arrows=arrows,
                    arrows_color=arrows_color)

    # ***********************       test 01 model     ******************
    # plot it responses for eyebrow model
    nb_model.plot_it_neurons(it_train,
                             title="01_it_train",
                             save_folder=os.path.join("models/saved", config["config_name"]))
    nb_model.plot_it_neurons(it_ref_test,
                             title="01_it_ref_test",
                             save_folder=os.path.join("models/saved", config["config_name"]))

    # ***********************       test 02 model     ******************
    # plot it responses for eyebrow model
    nb_model.plot_it_neurons_per_sequence(it_train,
                             title="02_it_train",
                             save_folder=os.path.join("models/saved", config["config_name"]))
    nb_model.plot_it_neurons_per_sequence(it_ref_test,
                             title="02_it_ref_test",
                             save_folder=os.path.join("models/saved", config["config_name"]))


    print("finished plotting test2")
    print()
    # ***********************       test 03 feature map visualisation     ******************
    # plot tracked vector on sequence
    plot_ft_pos_on_sequence(pos_2d, data[0],
                            vid_name='03_pos_human.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            lmk_size=1, ft_size=(56, 56))

    # plot tracked vector on sequence
    plot_ft_pos_on_sequence(test_pos_2d, test_data[0],
                            vid_name='03_pos_monkey.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            lmk_size=1, ft_size=(56, 56))
    print("finished plotting on sequence")

    # plot tracked vector on feature maps
    max_preds = np.expand_dims(np.amax(preds, axis=-1), axis=3)
    max_test_preds = np.expand_dims(np.amax(test_preds, axis=-1), axis=3)
    print("[PRED] max_preds", np.shape(max_preds))
    print("[PRED] max_test_preds", np.shape(max_test_preds))

    # plot tracked vector on feature maps
    max_preds_plot = max_preds / np.amax(max_preds) * 255
    plot_ft_pos_on_sequence(pos_2d, max_preds_plot, vid_name='03_ft_pos_human.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            pre_proc='raw', ft_size=(56, 56))
    # monkey
    test_max_preds_plot = max_test_preds / np.amax(max_test_preds) * 255
    plot_ft_pos_on_sequence(test_pos_2d, test_max_preds_plot, vid_name='03_ft_pos_monkey.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            pre_proc='raw', ft_size=(56, 56))
    print()
