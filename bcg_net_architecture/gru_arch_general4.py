import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers, callbacks, regularizers, optimizers
from tensorflow.python.keras import backend as K
import os

# called with bcg_net_architecture.gru_arch_general4.create_arch(blah)

def get_name():
    filename = os.path.basename(__file__)
    return os.path.splitext(filename)[0]


def create_arch(n_input, n_output, opt_feature_extract):
    # we pass opt because some arch are incompatible with some feature
    # extractions… so we need to do an error check for compat sometimes.
    # Multi-run, no motion, simple
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=session_config)

    K.set_floatx('float64')
    rs_input = layers.Input(shape=(None, 6), dtype='float64', name='rs_input')
    ecg_input = layers.Input(shape=(None, 1), dtype='float64', name='ecg_input')

    gru1_out = layers.Bidirectional(layers.CuDNNGRU(16, return_sequences=True,
                                                    recurrent_regularizer=regularizers.l2(0.096),
                                                    activity_regularizer=regularizers.l2(0.030)))(ecg_input)

    gru2_out = layers.Bidirectional(layers.CuDNNGRU(16, return_sequences=True,
                                                    recurrent_regularizer=regularizers.l2(0.090),
                                                    activity_regularizer=regularizers.l2(0.013)))(gru1_out)

    d3_out = layers.Dense(8, activation='relu')(gru2_out)
    d3_out_do = layers.Dropout(0.327)(d3_out)

    gru3_out = layers.Bidirectional(layers.CuDNNGRU(16, return_sequences=True,
                                                    recurrent_regularizer=regularizers.l2(0.024),
                                                    activity_regularizer=regularizers.l2(0.067)))(d3_out_do)

    gru4_out = layers.Bidirectional(layers.CuDNNGRU(64, return_sequences=True,
                                                    recurrent_regularizer=regularizers.l2(2.48e-07),
                                                    activity_regularizer=regularizers.l2(0.055)))(gru3_out)

    bcg_out = layers.Dense(63, activation='linear')(gru4_out)
    model = Model(inputs=[ecg_input, rs_input], outputs=bcg_out)

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    return model


if __name__ == '__main__':
    """ used for debugging """

