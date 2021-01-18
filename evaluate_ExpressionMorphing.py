"""
2020/12/16
This script tried to create transfer from human to monkey dataset.
2021/01/18
The results are bad.
Look at plot_directions.py and calculate_data_distance.py instead.
"""

import os
import matplotlib.pyplot as plt

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase


config = load_config("norm_base_expressionMorphing_t0002.json")
save_name = "transfer_human_monkey"
save_folder = os.path.join("models/saved", config['save_name'], save_name)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

data_train = load_data(config)
print("data_train[0].shape",data_train[0].shape)
data_test = load_data(config, train=False, sort_by=['monkey_avatar', 'anger'])
data_test_human = [data_test[0][0:300], data_test[1][0:300]]
data_test_monkey = [data_test[0][300:600], data_test[1][300:600]]

if True:
    norm_base = NormBase(config, input_shape=(224, 224, 3))
    norm_base.fit(data_train)
    norm_base.save_model(config,save_name)
else:
    norm_base = NormBase(config, input_shape=(224,224,3), save_name=save_name)

accuracy_both, it_resp_both, labels_both = norm_base.evaluate(data_train)
accuracy_human, it_resp_human, labels_human = norm_base.evaluate(data_test_human)
accuracy_monkey, it_resp_monkey, labels_monkey = norm_base.evaluate(data_test_monkey)

# evaluate on human data
norm_base.fit(data_test_human, fit_dim_red=False, fit_ref=True, fit_tun=False)
accuracy_human, it_resp_human, labels_human = norm_base.evaluate(data_test_human)

# evaluate on monkey data
norm_base.fit(data_test_monkey, fit_dim_red=False, fit_ref=True, fit_tun=False)
accuracy_monkey, it_resp_monkey, labels_monkey = norm_base.evaluate(data_test_monkey)

plt.figure()
plt.plot(range(150), it_resp_human[0:150])
plt.savefig(os.path.join(save_folder, "it_resp_human1.png"))

plt.figure()
plt.plot(range(150), it_resp_human[150:300])
plt.savefig(os.path.join(save_folder, "it_resp_human2.png"))

plt.figure()
plt.plot(range(150), it_resp_monkey[150:300])
plt.savefig(os.path.join(save_folder, "it_resp_monkey1.png"))

plt.figure()
plt.plot(range(150), it_resp_monkey[150:300])
plt.savefig(os.path.join(save_folder, "it_resp_monkey2.png"))
