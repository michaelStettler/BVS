from utils.load_config import load_config
from utils.evaluate_model import evaluate_model


def evaluate_nu(config):
    if isinstance(config['nu'], list):
        nus = config['nu']
    else:
        nus = [config['nu']]

    for nu in nus:
        config['nu'] = nu
        print("[LOOP] nu=", nu)

        accuracy, it_resp, labels, ref_vector, tun_vector = evaluate_model(config, "nu_%f" %nu, legacy=True)

        print("accuracy", accuracy)


if __name__ == "__main__":

    """
    2020/12/07
    This script evaluates the accuracy for different nu_values. Results are plotted with t04b_plot_accuracy_nu.py
    2021/01/18
    Note: norm_base_affectNet_sub8_4000_t0006.json does not use dimensionality reduction. Therefore, the accuracy is 
    very bad. Additionally, the tuning function changed significantly since then.
    """

    config_names = ['norm_base_affectNet_sub8_4000_t0006.json']
    for config_name in config_names:
        config = load_config(config_name)
        evaluate_nu(config)