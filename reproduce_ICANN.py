from utils.load_data import load_data
from utils.load_config import load_config
from models.NormBase import NormBase


config = load_config("norm_base_expressivityLevels_t0001.json")

data_train = load_data(config)

norm_base = NormBase(config, input_shape=(224,224,3))

norm_base.fit(data_train)

data_test = load_data(config, train=False, sort_by=['image'])

norm_base.evaluate(data_test)