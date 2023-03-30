import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
run: python -m projects.loss_optimization.01_create_alpha_plot
"""

# save_path = r'C:\Users\Alex\Documents\Uni\NRE\icann_results'
save_path = 'D:/Dataset/FERG_DB_256/loss_optimization'

with open(os.path.join(save_path, 'train_alpha_accuracy'), 'rb') as f:
    train_accuracy = pickle.load(f)

with open(os.path.join(save_path, 'test_alpha_accuracy'), 'rb') as f:
    test_accuracy = pickle.load(f)

with open(os.path.join(save_path, 'alpha_losses'), 'rb') as f:
    losses = pickle.load(f)

for alpha in test_accuracy.keys():
    print(alpha, np.max(test_accuracy[alpha]))
    plt.plot(test_accuracy[alpha], label=str(alpha))
plt.title('Test set classification accuracy for different balancing parameters')
plt.xlabel('Training epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

for alpha in losses.keys():
    plt.plot(losses[alpha], label=str(alpha))
plt.title('Test set classification accuracy for different balancing parameters')
plt.xlabel('Training epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


