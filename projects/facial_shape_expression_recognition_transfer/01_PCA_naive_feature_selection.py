"""
This script summarizes the main results using the naive PCA feature selection approach as input to the NormBase
mechanism.
These results extend the ICANN paper in the way that we now investigate how to transfer the tuning vector from the
Norm Base model

The dataset use is the morphing space from the eLife psychophysics behavioural paper where the same expressions are
displayed on two different basic face shapes - human and monkey avatar. We study here only the two prototype expressions
c2 (human fear - strong eyebrow movement) and c3 (monkey anger - strong jaw movements)
"""

from utils.load_config import load_config
from utils.load_data import load_data
from models.NormBase import NormBase

# ----------------------------------------------------------------------------------------------------------------------
# 1: train model with human avatar
config_name = 'NB_PCA_human_c1c3_m0001.json'
config = load_config(config_name, path='configs/norm_base_config')
# declare model
model = NormBase(config,
                 input_shape=tuple(config['input_shape']),
                 save_name=config['config_name'])
# train model
model.fit(load_data(config, train=True),
          fit_dim_red=True,
          fit_ref=True,
          fit_tun=True)

# ----------------------------------------------------------------------------------------------------------------------
# 2: test model accuracy on human avatar

# ----------------------------------------------------------------------------------------------------------------------
# 3: train model with monkey avatar

# ----------------------------------------------------------------------------------------------------------------------
# 4: test model with monkey avatar

# ----------------------------------------------------------------------------------------------------------------------
# 5: test transfer learning, predict on monkey avatar using the human avatar training

# ----------------------------------------------------------------------------------------------------------------------
# 6: show that the vector are orthogonal

# ----------------------------------------------------------------------------------------------------------------------
# 7: train PCA using both avatars

# ----------------------------------------------------------------------------------------------------------------------
# 8: train training vector on human avatar with PCA index from both avatars

# ----------------------------------------------------------------------------------------------------------------------
# 9: predict on monkey avatar using the PCA trained on both avatars and tuning vector on human avatar

# ----------------------------------------------------------------------------------------------------------------------
# 10: show that vectors are at exactly 45°
