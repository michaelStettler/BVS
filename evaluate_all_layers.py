import json
import os
import numpy as np
from utils.load_data import load_data

from models.NormBase import NormBase

congig_path = 'configs/norm_base_config'
# config_name = 'norm_base_monkey_test.json'
config_name = 'norm_base_affectNet_sub8_4000_t0001.json'
config_file_path = os.path.join(congig_path, config_name)
print("config_file_path", config_file_path)

# load norm_base_config file
with open(config_file_path) as json_file:
    config = json.load(json_file)

if config['v4_layer']!='':
    raise ValueError("v4_layer: {} is chosen, but should be empty! Please choose '' instead!"
                     .format(config['v4_layer']))

for layer in ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']:
    config['v4_layer'] = layer
    print('[LOOP] start with v4_layer: {}'.format(config['v4_layer']))

    print("[LOOP] start training")

    # load train data
    data = load_data(config)
    print("[Data] -- Data loaded --")

    # create model
    norm_base = NormBase(config, input_shape=(224, 224, 3))

    # train model
    ref_vector, tun_vector = norm_base.fit(data, batch_size=config['batch_size'])

    print("[LOOP] start prediction")

    # load test data
    data = load_data(config, train=False, sort_by=['image'])
    print("[Data] -- Data loaded --")

    #TODO check if necessary
    norm_base.set_ref_vector(ref_vector)
    norm_base.set_tuning_vector(tun_vector)
    print("[MODEL] Set ref vector", np.shape(ref_vector))
    print("[MODEL] Set tuning vector", np.shape(tun_vector))

    #evaluate
    it_resp = norm_base.evaluate(data)
    print("shape it_resp", np.shape(it_resp))

    print('[LOOP] finished with v4_layer: {}'.format(config['v4_layer']))