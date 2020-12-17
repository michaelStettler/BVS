import os
import numpy as np
from utils.load_config import load_config
from models.NormBase import NormBase
from utils.load_data import load_data

config = load_config("norm_base_affectNet_sub8_4000_t0007.json")

#accuracy, it_resp, labels, ref_vector, tun_vector = evaluate_model(config, config['v4_layer'])

norm_base = NormBase(config, input_shape=(224, 224, 3))
# load vectors
save_folder = os.path.join("models/saved", config['save_name'], config['v4_layer'])
ref_vector = np.load(os.path.join(save_folder, "ref_vector.npy"))
tun_vector = np.load(os.path.join(save_folder, "tuning_vector.npy"))
# set vectors
norm_base.set_ref_vector(ref_vector)
norm_base.set_tuning_vector(tun_vector)

#TODO load data
data = load_data(config, train=False, sort_by=['image'])
print("type(data)", type(data))

projection, labels = norm_base.projection_tuning(data)

# tun_vector is normed in 2-norm
print("tun_vector 1-norm", np.linalg.norm(tun_vector, ord=1, axis=1))
print("tun_vector 2-norm", np.linalg.norm(tun_vector, ord=2, axis=1))