from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers, callbacks, regularizers, optimizers
from tensorflow.python.keras import backend as K
import os

# called with bcg_net_architecture.arch0001.create_arch(blah)

def get_name():
    filename = os.path.basename(__file__)
    return os.path.splitext(filename)[0]


def create_arch(n_input, n_output, opt_feature_extract):
    # we pass opt because some arch are incompatible with some feature
    # extractions… so we need to do an error check for compat sometimes.
    model = Sequential()
    model.add(layers.GRU(n_input, return_sequences=True, input_shape=(None, 1)))
    model.add(layers.Dense(n_output, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    """ used for debugging """

