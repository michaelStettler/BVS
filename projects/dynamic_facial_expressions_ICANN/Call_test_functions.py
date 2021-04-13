import json
import os
import numpy as np

#from EB_Test_seq_code import test_model_seq
from Test_set_VGG_feature_extr_skPCA import load_EB_data
from Test_set_VGG_feature_extr_skPCA import extract_v4_features
from EB_Test_Seq_code_sklearnPCA import compute_snapshot_neurons
from EB_Test_Seq_code_sklearnPCA import compute_expression_neurons


def predict_EB_responses(config, do_plot=False):
    # load data
    data = load_EB_data(config)

    # extract features
    extract_v4_features(data, config, save=True)

    # predict snapshot responses
    snap = compute_snapshot_neurons(config, do_plot=do_plot, do_reverse=0)

    # compute neural field
    compute_expression_neurons(snap, config)


if __name__ == '__main__':
    do_plot = True  # Added for plotting

    config_path = 'configs/example_base_config'
    # config_name = 'example_base_reproduce_ICANN_cat.json'
    config_name = 'example_base_reproduce_ICANN_expressivity.json'
    # config_name = 'example_base_monkey_test_reduced_dataset_.json'

    config_file_path = os.path.join(config_path, config_name)
    print("config_file_path", config_file_path)

    # Load example_base_config file
    with open(config_file_path) as json_file:
        config = json.load(json_file)

    # predict Example Base responses
    predict_EB_responses(config, do_plot)
