import os
import numpy as np
import matplotlib.pyplot as plt

from utils.load_config import load_config

config = load_config("norm_base_expressivityLevels_t0003.json")
save_name = "simplified"
save_folder = os.path.join("models/saved", config['save_name'], save_name)
accuracy = np.load(os.path.join(save_folder, "accuracy.npy"))
it_resp = np.load(os.path.join(save_folder, "it_resp.npy"))
labels = np.load(os.path.join(save_folder, "labels.npy"))

print("accuracy", accuracy)
print("it_resp.shape", it_resp.shape)
print("labels.shape", labels.shape)

#--> data loaded is the concatenation of train_data and test_data
# Fear 1.0 in 0:120
# LipSmacking 1.0 in 120:240
# Threat 1.0 in 240:360
# Fear 0.25 in 360:480
# Fear 0.5 in 480:600
# Fear 0.75 in 600:720
# LipSmacking 0.25 in 720:840
# LipSmacking 0.5 in 840:960
# LipSmacking 0.75 in 960:1080
# Threat 0.25 in 1080:1200
# Threat 0.5 in 1200:1320
# Threat 0.75 in 1320:1440
# it_resp 0=Neutral, 1=Threat, 2=Fear, 3=LipSmacking
it_resp_threat = [it_resp[240:360,1], it_resp[1320:1440,1],it_resp[1200:1320,1], it_resp[1080:1200,1]]
it_resp_fear = [it_resp[0:120,2], it_resp[600:720,2],it_resp[480:600,2], it_resp[360:480,2]]
it_resp_lipSmacking = [it_resp[120:240,3], it_resp[960:1080,3],it_resp[840:960,3], it_resp[720:840,3]]

fig = plt.figure(figsize=(15,10))
plt.subplots(3,1)
plt.suptitle("Face Neuron Response")
for i, data in enumerate([it_resp_threat, it_resp_fear, it_resp_lipSmacking]):
    plt.subplot(3,1,i+1)
    plt.ylabel(['Threat', 'Fear', 'LipSmacking'][i])
    for j, percent in enumerate(['100%', '75%', '50%', '25%']):
        plt.plot(range(120),data[j], label=percent)
    plt.legend()

# save plot
plt.savefig(os.path.join(save_folder, "plot_ICANN.png"))