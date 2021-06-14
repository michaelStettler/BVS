import numpy as np
from utils.get_ref_idx_frames import get_ref_idx_frames


def ref_feature_map_neuron(preds, label, config, activation='relu'):
    neutr_preds = get_ref_idx_frames([preds, label], ref_index=config['ref_category'])
    preds_ref = np.mean(neutr_preds, axis=0)
    new_preds = preds - np.repeat(np.expand_dims(preds_ref, axis=0), len(preds), axis=0)

    if activation == 'relu':
        new_preds[new_preds < 0] = 0
    else:
        raise NotImplementedError("Activation: {} is not yet implemented for the ref_feature_map_neuron".format(activation))

    return new_preds