"""
Module to evaluate the EfficientFace model.
"""
import sys
from os.path import join
from collections import OrderedDict
import torch
sys.path.append(r'C:\Users\Alex\Documents\Uni\NRE\BVS\projects\facial_shape_expression_recognition_transfer\model_benchmarking')
sys.path.append(r'C:\Users\Alex\Documents\Uni\NRE\BVS\projects\facial_shape_expression_recognition_transfer\model_benchmarking\models\EfficientFace\models')
from EfficientFace import *
from BFS import *

weight_path = r'C:\Users\Alex\Documents\Uni\NRE\BVS\projects\facial_shape_expression_recognition_transfer\model_benchmarking\pretrained_weights\EfficientFace_Trained_on_AffectNet7.pth.tar'
base_path = r'C:\Users\Alex\Documents\Uni\NRE\BVS\projects\facial_shape_expression_recognition_transfer\model_benchmarking\img'

model = efficient_face(num_classes=7)
checkpoint = torch.load(weight_path)
state_dict = checkpoint['state_dict']

# Convert their labels to alphabetical order
label_conversion = {'0': 4, '1': 3, '2': 5, '3': 6, '4': 2, '5': 1, '6': 0}

# Convert state_dict from nn.Parallel to regular format by adjusting the dict keys
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v

# Set model parameters
model.load_state_dict(new_state_dict)
model.eval()

types = ['human', 'cartoon', 'monkey']
for type in types:
    bfs = BFS(path=join(base_path, type))
    for X, y in bfs.test_loader:
        out = torch.argmax(model(X), axis=1)

    # Convert labels
    converted_labels = []
    for i in range(len(y)):
        label = str(out[i].item())
        converted_labels.append(label_conversion[label])

    converted_labels = torch.tensor(converted_labels)
    correct = torch.argwhere(converted_labels == y)
    print('Accuracy for', type, ':', len(correct) / len(y))