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
test script to try an implementation of a holistic representation model by a RBF_pattern function of the face


run: python -m tests.NormBase.t11g_holisitc_template
"""

# define configuration
config_path = 'NB_t11g_holistic_template_m0002.json'
plot_intermediate = True
compute_NB = False

train = True
human_full = False
test = False
plot = False


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
    # left ext eyebrow, left int eyebrow, right int eyebrow, right ext eyebrow, left lips, up lip, right lip, down lip
    rbf_template = [[[16, 21], [16, 21]], [[15, 20], [21, 26]], [[16, 21], [30, 35]], [[16, 21], [37, 42]],
            [[35, 40], [21, 26]], [[34, 39], [27, 32]], [[36, 41], [33, 38]], [[38, 43], [27, 32]]]
    config['rbf_sigma'] = [2100, 2100, 2100, 2100, 2100, 3000, 2100, 2500]

    # left lip
    rbf_template = [[[36, 41], [33, 38]]]
    config['rbf_sigma'] = [2100]
    # up lip
    # rbf_template = [[[34, 39], [27, 32]]]  # good on c2 with 3000
    # config['rbf_sigma'] = [3000]
    template = [[[37, 38], [29, 30]]]
    # rbf_template = [[[37, 38], [29, 31]]]  # a bit jittery with 600
    config['rbf_sigma'] = [500]
    # config['rbf_sigma'] = [600]
    # down lip
    rbf_template = [[[38, 40], [29, 31]]]  # [1350] start to be jittery
    rbf_template = [[[38, 40], [28, 32]]]
    config['rbf_sigma'] = [2200]  # upper bound
    config['rbf_sigma'] = [2100]
    # config['rbf_sigma'] = [2000]  # lower bound
    patterns = PatternFeatureSelection(config, template=rbf_template)
    rbf_template = np.repeat(np.expand_dims(preds, axis=0), len(rbf_template), axis=0)
    print("[TRAIN] shape rbf_template", np.shape(rbf_template))
    template = patterns.fit(rbf_template)
    template[template < 0.1] = 0

    # # using eyebrow ft map only
    # eyebrow_mask = [[[15, 22], [14, 21]]]  # left eye ext
    # config['rbf_sigma'] = [980]
    # eyebrow_mask = [[[15, 22], [21, 28]]]  # left eye int
    # config['rbf_sigma'] = [950]
    # eyebrow_mask = [[[17, 24], [27, 34]]]  # right eye int
    # config['rbf_sigma'] = [950]
    # eyebrow_mask = [[[16, 23], [38, 45]]]  # right eye ext
    # eyebrow_mask = [[[19, 20], [38, 39]]]  # right eye ext  point where I want it to be
    # config['rbf_sigma'] = [850]  # works on c1
    # # 1
    # eyebrow_mask = [[[17, 22], [36, 43]]]  # right eye ext
    # # 2
    # config['rbf_sigma'] = [750]  # works on c2
    # # 3
    # config['rbf_sigma'] = [850]  # avoid jump on c1
    # # 4
    # eyebrow_mask = [[[16, 23], [35, 44]]]  # 7x7
    # config['rbf_sigma'] = [1500]  # catch front
    # config['rbf_sigma'] = [1400]
    # # config['rbf_sigma'] = [1300]  # jump
    # # good on c2
    # # eyebrow_mask = [[[15, 22], [14, 21]], [[15, 22], [21, 28]], [[17, 24], [27, 34]], [[15, 22], [36, 43]]]
    # # config['rbf_sigma'] = [980, 900, 950, 900]
    # # test on c1
    # # eyebrow_mask = [[[15, 22], [14, 21]], [[15, 22], [21, 28]], [[17, 24], [27, 34]], [[15, 22], [36, 43]]]
    # # config['rbf_sigma'] = [980, 950, 950, 920]
    # eyebrow_patterns = PatternFeatureSelection(config, mask=eyebrow_mask)  # 3x3  eyebrow
    # mask_eyebrow_template = np.repeat(np.expand_dims(eyebrow_preds, axis=0), len(eyebrow_mask), axis=0)
    # print("[TRAIN] shape mask_eyebrow_template", np.shape(mask_eyebrow_template))
    # eyebrow_template = eyebrow_patterns.fit(mask_eyebrow_template)
    # print("[TRAIN] shape eyebrow_template", np.shape(eyebrow_template))
    #
    # # using lips ft map only
    # lips_mask = [[[35, 40], [21, 26]]]  # left lips
    # config['rbf_sigma'] = [2100]
    # lips_mask = [[[34, 39], [27, 32]]]  # up lips
    # config['rbf_sigma'] = [3000]
    # lips_mask = [[[36, 41], [33, 38]]]  # right lips
    # config['rbf_sigma'] = [1800]
    # lips_mask = [[[38, 43], [27, 32]]]  # down lips
    # config['rbf_sigma'] = [2500]
    # lips_mask = [[[35, 40], [21, 26]], [[34, 39], [27, 32]], [[36, 41], [33, 38]], [[38, 43], [27, 32]]]
    # config['rbf_sigma'] = [2100, 3000, 1800, 2500]
    # lips_patterns = PatternFeatureSelection(config, mask=lips_mask)  # 3x3  eyebrow
    # mask_lips_template = np.repeat(np.expand_dims(lips_preds, axis=0), len(lips_mask), axis=0)
    # print("[TRAIN] shape mask_lips_template", np.shape(mask_lips_template))
    # lips_template = lips_patterns.fit(mask_lips_template)
    # print("[TRAIN] shape lips_template", np.shape(lips_template))
    #
    # template = np.concatenate((eyebrow_template, lips_template), axis=3)
    # template = eyebrow_template
    # print("[TRAIN] shape template", np.shape(template))
    # template[template < 0.1] = 0

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

        # test_max_preds = np.expand_dims(np.amax(test_preds, axis=3), axis=3)
        test_max_preds = np.expand_dims(np.amax(eyebrow_preds, axis=3), axis=3)
        # test_max_preds = np.expand_dims(np.amax(lips_preds, axis=3), axis=3)
        preds_plot = test_max_preds / np.amax(test_max_preds) * 255
        print("shape preds_plot", np.shape(preds_plot))
        plot_ft_pos_on_sequence(pos, preds_plot, vid_name='00_ft_eyebrow_pos.mp4',
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

        # ds_train = nb_model._get_decisions_neurons(it_train, config['seq_length'])
        # print("[TRAIN] shape ds_train", np.shape(ds_train))
        # print()

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

    # max activation
    max_eyebrow_preds = np.expand_dims(np.amax(eyebrow_preds, axis=-1), axis=3)
    max_lips_preds = np.expand_dims(np.amax(lips_preds, axis=-1), axis=3)
    print("[PRED] max_eyebrow_preds", np.shape(max_eyebrow_preds))
    print("[PRED] max_lips_preds", np.shape(max_lips_preds))

    # compute templates
    mask_template = np.repeat(np.expand_dims(preds, axis=0), len(mask), axis=0)
    template = patterns.transform(mask_template)
    print("[PRED] shape template", np.shape(template))
    template[template < 0.1] = 0

    # mask_eyebrow_template = np.repeat(np.expand_dims(eyebrow_preds, axis=0), len(eyebrow_mask), axis=0)
    # print("[TRAIN] shape mask_eyebrow_template", np.shape(mask_eyebrow_template))
    # eyebrow_template = eyebrow_patterns.transform(mask_eyebrow_template)
    # mask_lips_template = np.repeat(np.expand_dims(lips_preds, axis=0), len(lips_mask), axis=0)
    # print("[TRAIN] shape mask_lips_template", np.shape(mask_lips_template))
    # lips_template = lips_patterns.transform(mask_lips_template)
    #
    # template = np.concatenate((eyebrow_template, lips_template), axis=3)
    # print("[PRED] shape template", np.shape(template))
    # template[template < 0.1] = 0

    # compute positions
    pos = calculate_position(template, mode="weighted average", return_mode="xy float flat")
    print("[PRED] shape pos", np.shape(pos))

    if compute_NB:
        # get it resp for eyebrows
        it_train = nb_model._get_it_resp(pos)
        print("[PRED] shape it_train", np.shape(it_train))

        # ds_train = nb_model._get_decisions_neurons(it_train, config['seq_length'])
        # print("[PRED] shape ds_train", np.shape(ds_train))
        # print()

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

    # max activation
    test_max_eyebrow_preds = np.expand_dims(np.amax(test_eyebrow_preds, axis=-1), axis=3)
    print("[TEST] test_max_eyebrow_preds", np.shape(test_max_eyebrow_preds))
    test_max_lips_preds = np.expand_dims(np.amax(test_lips_preds, axis=-1), axis=3)
    print("[TEST] test_max_lips_preds", np.shape(test_max_lips_preds))

    # add holistic templates
    test_mask = [[[8, 15], [17, 24]], [[9, 16], [22, 29]], [[9, 16], [28, 35]], [[8, 15], [33, 40]],
                 [[33, 38], [19, 24]], [[32, 37], [26, 31]], [[33, 38], [32, 37]], [[34, 39], [26, 31]]]
    config['rbf_sigma'] = [3500, 3500, 3500, 3500, 2100, 3200, 2100, 3500]

    test_mask = [[[32, 37], [26, 31]]]
    config['rbf_sigma'] = [3200]
    test_patterns = PatternFeatureSelection(config, mask=test_mask)  # 3x3  eyebrow
    test_mask_template = np.repeat(np.expand_dims(test_preds, axis=0), len(test_mask), axis=0)
    test_template = test_patterns.fit(test_mask_template)
    test_template[test_template < 0.1] = 0

    #
    # # using eyebrow ft map only
    # test_eyebrow_maks = [[[9, 16], [17, 24]]]  # left eye ext
    # config['rbf_sigma'] = [800]
    # test_eyebrow_maks = [[[10, 17], [22, 29]]]  # left eye int
    # config['rbf_sigma'] = [800]
    # test_eyebrow_maks = [[[9, 16], [28, 35]]]  # right eye int
    # config['rbf_sigma'] = [800]
    # test_eyebrow_maks = [[[8, 15], [33, 40]]]  # right eye ext
    # config['rbf_sigma'] = [540]
    # test_eyebrow_maks = [[[9, 16], [17, 24]], [[10, 17], [22, 29]], [[9, 16], [28, 35]], [[8, 15], [33, 40]]]
    # config['rbf_sigma'] = [800, 800, 800, 540]
    # test_patterns = PatternFeatureSelection(config, mask=test_eyebrow_maks)  # 3x3  eyebrow
    # test_mask_eyebrow_template = np.repeat(np.expand_dims(test_eyebrow_preds, axis=0), len(test_eyebrow_maks), axis=0)
    # print("[TEST] shape test_mask_eyebrow_template", np.shape(test_mask_eyebrow_template))
    # test_eyebrow_template = test_patterns.fit(test_mask_eyebrow_template)
    # print("[TEST] shape test_eyebrow_template", np.shape(test_eyebrow_template))
    #
    # # using lips ft map only
    # test_lips_mask = [[[33, 38], [19, 24]]]  # left eye ext
    # config['rbf_sigma'] = [1800]
    # test_lips_mask = [[[32, 37], [26, 31]]]  # left eye int
    # config['rbf_sigma'] = [3000]
    # test_lips_mask = [[[33, 38], [32, 37]]]  # right eye int
    # config['rbf_sigma'] = [2100]
    # test_lips_mask = [[[35, 40], [26, 31]]]  # right eye ext
    # config['rbf_sigma'] = [3200]
    # test_lips_mask = [[[33, 38], [19, 24]], [[32, 37], [26, 31]], [[33, 38], [32, 37]], [[35, 40], [26, 31]]]
    # config['rbf_sigma'] = [1800, 3000, 2100, 3200]
    # test_patterns = PatternFeatureSelection(config, mask=test_lips_mask)  # 3x3  eyebrow
    # test_mask_lips_template = np.repeat(np.expand_dims(test_lips_preds, axis=0), len(test_lips_mask), axis=0)
    # print("[TEST] shape test_mask_lips_template", np.shape(test_mask_lips_template))
    # test_lips_template = test_patterns.fit(test_mask_lips_template)
    # print("[TEST] shape test_lips_template", np.shape(test_lips_template))
    #
    # test_template = np.concatenate((test_eyebrow_template, test_lips_template), axis=3)
    # print("[TEST] shape test_template", np.shape(test_template))
    # test_template[test_template < 0.1] = 0
    #
    # compute positions
    test_pos = calculate_position(test_template, mode="weighted average", return_mode="xy float flat")
    print("[TEST] shape test_pos", np.shape(test_pos))

    if plot_intermediate:
        plot_cnn_output(test_template, os.path.join("models/saved", config["config_name"]),
                        "00_test_template.gif", verbose=True, video=True)

        test_pos_2d = np.reshape(test_pos, (len(test_pos), -1, 2))
        plot_ft_map_pos(test_pos_2d,
                        fig_name="00b_monkey_test_pos.png",
                        path=os.path.join("models/saved", config["config_name"]))

        # test_max_preds = np.expand_dims(np.amax(test_preds, axis=3), axis=3)
        # test_max_preds = np.expand_dims(np.amax(test_eyebrow_preds, axis=3), axis=3)
        test_max_preds = np.expand_dims(np.amax(test_lips_preds, axis=3), axis=3)
        preds_plot = test_max_preds / np.amax(test_max_preds) * 255
        print("shape preds_plot", np.shape(preds_plot))
        plot_ft_pos_on_sequence(test_pos, preds_plot, vid_name='00_ft_eyebrow_pos.mp4',
                                save_folder=os.path.join("models/saved", config["config_name"]),
                                pre_proc='raw', ft_size=(56, 56))


        plot_ft_pos_on_sequence(test_pos, test_data[0],
                                vid_name='00_ft_pos_monkey.mp4',
                                save_folder=os.path.join("models/saved", config["config_name"]),
                                lmk_size=1, ft_size=(56, 56))

    if compute_NB:
        # get IT responses of the model
        it_test = nb_model._get_it_resp(test_pos)

        # test by training new ref
        nb_model._fit_reference([test_pos, test_data[1]], config['batch_size'])
        ref_test = np.copy(nb_model.r)
        it_ref_test = nb_model._get_it_resp(test_pos)
        #
        # # ds_test = nb_model._get_decisions_neurons(it_ref_test, config['seq_length'])
        # # print("[TEST] shape ds_test", np.shape(ds_test))


if plot:
    # --------------------------------------------------------------------------------------------------------------------
    # plots
    # ***********************       test 00 raw output      ******************
    #
    # # raw activity
    # plot_cnn_output(max_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
    #                 "00_max_feature_maps_eyebrow_output.gif", verbose=True, video=True)
    # plot_cnn_output(max_lips_preds, os.path.join("models/saved", config["config_name"]),
    #                 "00_max_feature_maps_lips_output.gif", verbose=True, video=True)
    # plot_cnn_output(test_max_eyebrow_preds, os.path.join("models/saved", config["config_name"]),
    #                 "00_test_max_feature_maps_eyebrow_output.gif", verbose=True, video=True)
    # plot_cnn_output(test_max_lips_preds, os.path.join("models/saved", config["config_name"]),
    #                 "00_test_max_feature_maps_lips_output.gif", verbose=True, video=True)
    #
    # plot_cnn_output(preds, os.path.join("models/saved", config["config_name"]),
    #                 "00a_max_feature_maps_output.gif", verbose=True, video=True)
    # plot_cnn_output(test_preds, os.path.join("models/saved", config["config_name"]),
    #                 "00a_test_max_maps_output.gif", verbose=True, video=True)
    #
    # build arrows
    arrow_tail = np.repeat(np.expand_dims(np.reshape(ref_train, (-1, 2)), axis=0), config['n_category'], axis=0)
    arrow_head = np.reshape(ref_tuning, (len(ref_tuning), -1, 2))
    arrows = [arrow_tail, arrow_head]
    arrows_color = ['#0e3957', '#3b528b', '#21918c', '#5ec962', '#fde725']

    # put one color per label
    labels = data[1]
    color_seq = np.zeros(len(preds))
    color_seq[labels == 1] = 1
    color_seq[labels == 2] = 2
    color_seq[labels == 3] = 3
    color_seq[labels == 4] = 4

    print("[PLOT] shape preds", np.shape(preds))
    pos_2d = np.reshape(pos, (len(pos), -1, 2))
    print("[PLOT] shape pos_flat", np.shape(pos_2d))
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
    # nb_model.plot_it_neurons(it_test,
    #                          title="01_it_test",
    #                          save_folder=os.path.join("models/saved", config["config_name"]))
    nb_model.plot_it_neurons(it_ref_test,
                             title="01_it_ref_test",
                             save_folder=os.path.join("models/saved", config["config_name"]))

    # ***********************       test 02 model     ******************
    # plot it responses for eyebrow model
    nb_model.plot_it_neurons_per_sequence(it_train,
                             title="02_it_train",
                             save_folder=os.path.join("models/saved", config["config_name"]))
    # nb_model.plot_it_neurons_per_sequence(it_test,
    #                          title="02_it_test",
    #                          save_folder=os.path.join("models/saved", config["config_name"]))
    nb_model.plot_it_neurons_per_sequence(it_ref_test,
                             title="02_it_ref_test",
                             save_folder=os.path.join("models/saved", config["config_name"]))

    print()
    # plot tracked vector on sequence
    plot_ft_pos_on_sequence(pos, data[0],
                            vid_name='03_ft_pos_human.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            lmk_size=1, ft_size=(56, 56))

    # plot tracked vector on sequence
    plot_ft_pos_on_sequence(test_pos, test_data[0],
                            vid_name='03_ft_pos_monkey.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            lmk_size=1, ft_size=(56, 56))
    print()

    # plot tracked vector on feature maps
    max_eyebrow_preds_plot = max_eyebrow_preds / np.amax(max_eyebrow_preds) * 255
    plot_ft_pos_on_sequence(pos[:, :8], max_eyebrow_preds_plot, vid_name='03_ft_pos_eyebrow_human.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            pre_proc='raw', ft_size=(56, 56))
    max_lips_preds_plot = max_lips_preds / np.amax(max_lips_preds) * 255
    plot_ft_pos_on_sequence(pos[:, 8:], max_lips_preds_plot, vid_name='03_ft_pos_lips_human.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            pre_proc='raw', ft_size=(56, 56))
    # monkey
    test_max_eyebrow_preds_plot = test_max_eyebrow_preds / np.amax(test_max_eyebrow_preds) * 255
    plot_ft_pos_on_sequence(test_pos[:, :8], test_max_eyebrow_preds_plot, vid_name='03_ft_pos_eyebrow_monkey.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            pre_proc='raw', ft_size=(56, 56))
    test_max_lips_preds_plot = test_max_lips_preds / np.amax(test_max_lips_preds) * 255
    plot_ft_pos_on_sequence(test_pos[:, 8:], test_max_lips_preds_plot, vid_name='03_ft_pos_lips_monkey.mp4',
                            save_folder=os.path.join("models/saved", config["config_name"]),
                            pre_proc='raw', ft_size=(56, 56))


    # ***********************       test 04 decision neuron     ******************
    # nb_model.plot_decision_neurons(ds_train,
    #                                title="04_ds_train",
    #                                save_folder=os.path.join("models/saved", config["config_name"]),
    #                                normalize=True)
    # nb_model.plot_decision_neurons(ds_test,
    #                                title="04_ds_test",
    #                                save_folder=os.path.join("models/saved", config["config_name"]),
    #                                normalize=True)
