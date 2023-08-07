import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
run: python -m projects.loss_optimization.01_create_alpha_plot
"""

load_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results\single_domain_training'
# save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results\plots'
# load_path = 'D:/Dataset/FERG_DB_256/loss_optimization'

with open(load_path, 'rb') as f:
    domain_dict = pickle.load(f)
    for domain in domain_dict.keys():
        losses = domain_dict[domain]['losses']
        best_epoch = tf.argmin(losses).numpy()
        test_acc = domain_dict[domain]['test_accuracies'][best_epoch]
        print('Domain:', domain, ' - Test Accuracy:', test_acc)




