import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from projects.behavourial.project_utils import *
from scipy.stats import wilcoxon

from G_compute_KL_div_CNN import *

np.set_printoptions(precision=3, suppress=True)
"""
run: python -m projects.behavourial.07_compute_KL_div_CNN
"""

#%% define computer path
computer = 'windows'
# computer = 'alex'

computer_path, computer_letter = get_computer_path(computer)

conditions = ["human_orig", "monkey_orig"]

model_name = 'NRE_frobenius_static'


def softmax(array, beta):
    a = np.exp(array * beta)
    denom = np.sum(a, axis=-1, keepdims=True)
    out = a / denom
    return out


#%%
predictions = {}
for cond, condition in enumerate(conditions):
    condition_preds = {}
    load_path = os.path.join(computer_path, 'model_behav_preds')

    # load data
    # load behavioural data
    behavioural_path = os.path.join(computer_path, 'morphing_psychophysics_result')
    if condition == "human_orig":
        behav_data = np.load(os.path.join(behavioural_path, "human_avatar_orig.npy"))
    elif condition == "monkey_orig":
        behav_data = np.load(os.path.join(behavioural_path, "monkey_avatar_orig.npy"))

    behav_data = np.moveaxis(behav_data, 0, -1)
    print("shape behav_data", np.shape(behav_data))
    condition_preds['behavioral'] = behav_data

    # load model preds
    if 'linear' in model_name:
        preds = get_pred(join(load_path, 'linear_fits'), model_name.replace('_linear', ''), condition)
    else:
        preds = get_pred(load_path, model_name, condition)
    condition_preds['NRE'] = preds
    predictions[condition] = condition_preds


# Compute baseline values without softmax
baselines = {'human_orig': {}, 'monkey_orig': {}}
for condition in conditions:   # Do humans and monkey seperately
    # Read predictions from dictionary
    nre_preds = predictions[condition]['NRE']
    humans = predictions[condition]['behavioral']
    baselines[condition]['kl'] = compute_morph_space_KL_div(humans, nre_preds, sum=True)
    baselines[condition]['tot_var'] = compute_morph_space_total_variation(humans, nre_preds, sum=True)
    baselines[condition]['entropy_diff'] = compute_entropy_difference(humans, nre_preds, sum=True)

print(baselines)

# Compute difference metrics for softmaxed nre output with varying temperature
betas = np.arange(0, 10, step=0.01)
kl = {'human_orig': [], 'monkey_orig': []}
tot_var = {'human_orig': [], 'monkey_orig': []}
entropy_diff = {'human_orig': [], 'monkey_orig': []}
model_entropy = {'human_orig': [], 'monkey_orig': []}
for beta in betas:  # Loop over beta values for the softmax
    for condition in conditions:   # Do humans and monkey seperately
        # Read predictions from dictionary
        nre_preds = predictions[condition]['NRE']
        humans = predictions[condition]['behavioral']
        # Compute softmax on nre output
        inhibited_preds = softmax(nre_preds, beta=beta)

        kl[condition].append(compute_morph_space_KL_div(humans, inhibited_preds, sum=True))
        tot_var[condition].append(compute_morph_space_total_variation(humans, inhibited_preds, sum=True))
        entropy_diff[condition].append(compute_entropy_difference(humans, inhibited_preds, sum=True))



def make_line_plot(betas, kl, tot_var, entropy_diff, condition, baselines):
    cols = ['#B1614E', '#FDD692', '#A3B9C9']
    plt.plot(betas, kl[condition], label='KL Divergence', color=cols[0])
    plt.plot(betas, tot_var[condition], label='Total Variation', color=cols[1])
    plt.plot(betas, entropy_diff[condition], label='Entropy', color=cols[2])
    if condition == 'human_orig':
        plt.title('Human Avatar')
    else:
        plt.title('Monkey Avatar')

    plt.hlines(baselines[condition]['kl'], xmin=betas[0], xmax=betas[-1], colors=cols[0], linestyle='dashed')
    plt.hlines(baselines[condition]['tot_var'], xmin=betas[0], xmax=betas[-1], colors=cols[1], linestyle='dashed')
    plt.hlines(baselines[condition]['entropy_diff'], xmin=betas[0], xmax=betas[-1], colors=cols[2], linestyle='dashed')
    plt.ylim(0, 19)
    plt.xlabel('Softmax Temperature')
    plt.ylabel('Difference between NRE and Human Predictions')
    plt.legend()
    # plt.savefig(join('plots', 'tuning_' + condition) + '.' + 'svg', format='svg')
    plt.show()


make_line_plot(betas, kl, tot_var, entropy_diff, 'human_orig', baselines)
make_line_plot(betas, kl, tot_var, entropy_diff, 'monkey_orig', baselines)

for condition in conditions:
    print()
    print('Best values for model {} and condition {}'.format(model_name, condition))
    print('kl:', np.min(kl[condition]))
    print('tot_var:', np.min(tot_var[condition]))
    print('entropy_diff:', np.min(entropy_diff[condition]))