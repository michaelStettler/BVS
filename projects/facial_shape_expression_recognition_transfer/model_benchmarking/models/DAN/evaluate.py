"""
Module to evaluate the DAN model
"""
import sys
from os.path import join
from collections import OrderedDict
import torch
sys.path.append(r'C:\Users\Alex\Documents\Uni\NRE\BVS\projects\facial_shape_expression_recognition_transfer\model_benchmarking')
from BFS import *
from DAN import *

# Load model
weight_path = r'C:\Users\Alex\Documents\Uni\NRE\BVS\projects\facial_shape_expression_recognition_transfer\model_benchmarking\pretrained_weights\DAN_Trained_on_AffectNet7.pth'
base_path = r'C:\Users\Alex\Documents\Uni\NRE\BVS\projects\facial_shape_expression_recognition_transfer\model_benchmarking\img'

# Pretrained means pretraining on MS_celeb. These weights we neither have nor need
model = DAN(num_class=7, pretrained=False)
checkpoint = torch.load(weight_path)
# print(checkpoint['model_state_dict'])
state_dict = checkpoint['model_state_dict']

# Convert their labels to alphabetical order
label_conversion = {'0': 4, '1': 3, '2': 5, '3': 6, '4': 2, '5': 1, '6': 0}

model.load_state_dict(state_dict)
model.eval()

# Loop over the types
types = ['human', 'cartoon', 'monkey']
for type in types:
    bfs = BFS(path=join(base_path, type))
    for X, y in bfs.test_loader:
        print(model(X)[0].shape)
        out = torch.argmax(model(X)[0], axis=1)
    for i in range(len(y)):
        print(y[i], out[i])

    # Convert labels
    converted_labels = []
    for i in range(len(y)):
        label = str(out[i].item())
        converted_labels.append(label_conversion[label])

    converted_labels = torch.tensor(converted_labels)
    correct = torch.argwhere(converted_labels == y)
    print('Accuracy for', type, ':', len(correct) / len(y))