from utils.load_config import load_config
from utils.load_data import load_data

"""
run: python -m projects.loss_optimization.00_expression_count
"""

# define configuration
config_file = 'NR_03_FERG_from_LMK_m0001.json'
# config_file = 'NR_03_FERG_from_LMK_w0001.json'
# config_file = 'NR_03_FERG_alex.json'

# load config
config = load_config(config_file, path='/Users/michaelstettler/PycharmProjects/BVS/BVS/configs/norm_reference')

# load data
train_data = load_data(config, get_raw=True, get_only_label=True)
train_label = train_data[1]

expression_names = ['neutral', 'happy', 'anger', 'sadness', 'surprise', 'fear', 'disgust']

count_non_neutral = 0
for e, expr in enumerate(expression_names):
    expr_labels = train_label[train_label == e]
    print(f"{len(expr_labels)} images for expression: {expr}")

    if e > 0:
        count_non_neutral += len(expr_labels)

print("non neutral:", count_non_neutral)