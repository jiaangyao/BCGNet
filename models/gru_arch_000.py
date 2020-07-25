import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from models.default_models import RNNModel
from os.path import basename, splitext


class gru_arch_000(RNNModel):
    def __init__(self, n_input=1, n_output=63, lr=1e-2, opt_type='adam', **kwargs):
        """
        See documentation of rnn_model in default_models
        If needed can add new instance variables

        """
        super(gru_arch_000, self).__init__(n_input, n_output, lr, opt_type, **kwargs)

        self.name = splitext(basename(__file__))[0]

    """
    Depending on whether the user has tensorflow 2.X or 1.1X installed on their computer, they can modify the following
    two functions in the ways illustrated if they wish to modify the default model
    """

    # for TF 2.X
    @staticmethod
    def _model_tf_v2(n_input, n_output):
        from tensorflow.keras.layers import Input, Dense, Dropout

        session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=session_config)

        K.set_floatx('float64')

        # ---------------------
        # CUSTOM CHANGES STARTS

        # a simple example
        ecg_input = Input(shape=(None, n_input), dtype='float64', name='ecg_input')

        x = Dense(16, activation='relu')(ecg_input)
        x = Dropout(0.1)(x)

        x = Dense(8, activation='relu')(x)
        x = Dropout(0.1)(x)

        bcg_out = Dense(n_output, activation='linear')(x)
        model = Model(inputs=ecg_input, outputs=bcg_out)

        # CUSTOM CHANGES ENDS
        # ---------------------

        return model

    # for tensorflow 1.1X
    @staticmethod
    def _model_tf_v1(n_input, n_output):
        from tensorflow.python.keras.layers import Input, Dense, Dropout

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=session_config)

        K.set_floatx('float64')
        # ---------------------
        # CUSTOM CHANGES STARTS

        # a simple example
        ecg_input = Input(shape=(None, n_input), dtype='float64', name='ecg_input')

        x = Dense(16, activation='relu')(ecg_input)
        x = Dropout(0.1)(x)

        x = Dense(8, activation='relu')(x)
        x = Dropout(0.1)(x)

        bcg_out = Dense(n_output, activation='linear')(x)
        model = Model(inputs=ecg_input, outputs=bcg_out)

        # CUSTOM CHANGES ENDS
        # ---------------------

        return model
