import os
import numpy as np
from utils.load_data import load_data
from utils.CSV_data_generator import DataGen
from models.NormBase import NormBase


def evaluate_model(config, save_name, legacy = False):
    """
    This function runs and evaluates the model with given config.
    The model is saved to models/saved/config['save_name']/save_name.
    If already calculated the result is loaded instead of calculated.
    :param config:
    :param save_name:
    :param legacy: set to True to use old version
    :return:
    """
    if not os.path.exists(os.path.join("models/saved", config['save_name'])):
        os.mkdir(os.path.join("models/saved", config['save_name']))
    # folder for save and load
    save_folder = os.path.join("models/saved", config['save_name'], save_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if not legacy:
        try:
            # load results if available
            accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
            it_resp = np.load(os.path.join(save_folder, "it_resp.npy"))
            labels = np.load(os.path.join(save_folder, "labels.npy"))
        except IOError:
            try:
                norm_base = NormBase(config, input_shape=(224,224,3), save_name=save_name)
            except IOError:
                norm_base = NormBase(config, input_shape=(224,224,3))
                data_train = load_data(config)
                norm_base.fit(data_train)
                norm_base.save_model(config, save_name)
            data_test = load_data(config, train=False)
            accuracy, it_resp, labels = norm_base.evaluate(data_test)

        return accuracy, it_resp, labels
    else:
        try:
            # load results if available
            accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
            it_resp = np.load(os.path.join(save_folder, "it_resp.npy"))
            labels = np.load(os.path.join(save_folder, "labels.npy"))
            print("[MODEL] it_resp is available and is loaded from {}".format(save_folder))
            # load vectors if available
            ref_vector = np.load(os.path.join(save_folder, "ref_vector.npy"))
            tun_vector = np.load(os.path.join(save_folder, "tuning_vector.npy"))
        except IOError:
            # calculate results if not available
            print("[LOOP] start training")
            # create model
            norm_base = NormBase(config, input_shape=(224, 224, 3))
            try:
                # load vectors if available
                ref_vector = np.load(os.path.join(save_folder, "ref_vector.npy"))
                tun_vector = np.load(os.path.join(save_folder, "tuning_vector.npy"))
                print("[MODEL] ref_vector and tun_vector are available and loaded from {}"
                      .format(save_folder))

                norm_base.set_ref_vector(ref_vector)
                norm_base.set_tuning_vector(tun_vector)
                print("[MODEL] Set ref vector", np.shape(ref_vector))
                print("[MODEL] Set tuning vector", np.shape(tun_vector))
            except IOError:
                # calculate vectors if not available
                # load train data
                data_train = load_data(config)
                print("[Data] -- Data loaded --")

                # train model
                norm_base.fit(data_train, batch_size=config['batch_size'])
                ref_vector = norm_base.r
                tun_vector = norm_base.t

                # save model
                np.save(os.path.join(save_folder, "ref_vector"), ref_vector)
                np.save(os.path.join(save_folder, "tuning_vector"), tun_vector)

            print("[LOOP] start prediction")
            # load test data
            data_test = load_data(config, train=False, sort_by=['image'])
            print("[Data] -- Data loaded --")

            # evaluate
            accuracy, it_resp, labels = norm_base.evaluate(data_test)
            np.save(os.path.join(save_folder, "accuracy"), accuracy)
            np.save(os.path.join(save_folder, "it_resp"), it_resp)
            np.save(os.path.join(save_folder, "labels"), labels)

        return accuracy, it_resp, labels, ref_vector, tun_vector