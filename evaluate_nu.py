from utils.load_config import load_config
from utils.evaluate_model import evaluate_model

config = load_config("norm_base_affectNet_sub8_4000_t0006.json")

def evaluate_nu(config):
    if isinstance(config['nu'], list):
        nus = config['nu']
    else:
        nus = [config['nu']]

    for nu in nus:
        config['nu'] = nu
        print("[LOOP] nu=", nu)

        accuracy, it_resp, labels, ref_vector, tun_vector = evaluate_model(config, "nu_%f"%nu)

        print("accuracy", accuracy)

if __name__ == "__main__":
    config_names = ['norm_base_affectNet_sub8_4000_t0006.json']
    for config_name in config_names:
        config = load_config(config_name)
        evaluate_nu(config)