from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers, callbacks, regularizers, optimizers
from tensorflow.python.keras import backend as K
import os

"""
Called with models.gru_arch_001.create_arch(blah)
"""


def get_name():
    """
    Get the name of the arch.

    :return: The name of the arch.
    """
    filename = os.path.basename(__file__)
    return os.path.splitext(filename)[0]


def create_arch(n_input, n_output, opt_feature_extract):
    """
    We pass opt because some arch are incompatible with some feature
    extractions… so we need to do an error check for compat sometimes.

    :param n_input:
    :param n_output:
    :param opt_feature_extract:
    :return:
    """
    model = Sequential()
    model.add(layers.GRU(64, return_sequences=True, input_shape=(None, 1)))
    model.add(layers.Dropout(0.1))
    model.add(layers.GRU(32, return_sequences=True))
    model.add(layers.Dropout(0.1))
    model.add(layers.GRU(16, return_sequences=True))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


if __name__ == '__main__':
    """ used for debugging """
