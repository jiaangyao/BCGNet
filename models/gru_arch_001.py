import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from models.default_models import RNNModel
from os.path import basename, splitext


class gru_arch_001(RNNModel):
    def __init__(self, n_input=1, n_output=63, lr=1e-2, opt_type='adam', opt_feature_extract=None, **kwargs):
        """
        See documentation of rnn_model in default_models
        If needed can add new instance variables

        """
        super(gru_arch_001, self).__init__(n_input, n_output, lr, opt_type, opt_feature_extract, **kwargs)

        self.name = splitext(basename(__file__))[0]

    """
    Depending on whether the user has tensorflow 2.X or 1.1X installed on their computer, they can modify the following
    two functions in the ways illustrated if they wish to modify the default model
    """

    # for TF 2.X
    @staticmethod
    def _model_tf_v2(n_input, n_output, opt_feature_extract):
        from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, GRU
        from tensorflow.keras.regularizers import l1, l2

        session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=session_config)

        K.set_floatx('float64')

        # ---------------------
        # CUSTOM CHANGES STARTS

        ecg_input = Input(shape=(None, 1), dtype='float64', name='ecg_input')

        x = Bidirectional(GRU(16, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=2))(ecg_input)

        x = Bidirectional(GRU(16, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=2))(x)

        x = Dense(8, activation='relu')(x)
        x = Dropout(0.327)(x)

        x = Bidirectional(GRU(16, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=2))(x)

        x = Bidirectional(GRU(64, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=2))(x)

        bcg_out = Dense(63, activation='linear')(x)
        model = Model(inputs=ecg_input, outputs=bcg_out)

        # CUSTOM CHANGES ENDS
        # ---------------------

        return model
