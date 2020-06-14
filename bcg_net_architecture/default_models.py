import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import callbacks, regularizers, optimizers
from tensorflow.python.keras import backend as K
import os


class nn_model():
    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def model(self, n_input, n_output, opt_feature_extract):
        raise NotImplementedError

    def get_name(self):
        """
        Get the name of the arch.

        :return: The name of the arch.
        """
        filename = os.path.basename(__file__)
        self.name(os.path.splitext(filename)[0])

    def model(self):
        raise NotImplementedError

    @classmethod
    def disable(self):
        for layer in self.model.layers:
            layer.trainable = False
        self.model.trainable = False

    @classmethod
    def enable(self):
        for layer in self.model.layers:
            layer.trainable = True
        self.model.trainable = True


class rnn_model(nn_model):
    def __init__(self, lr=1e-2, opt_type='adam', **kwargs):
        """
        lr: learning rate
        opt_type: chosen type of optimizer, allowed to be adam, rmsprop or sgd
        kwargs: clipnorm: normalized value for gradient clipping
                clipvalue: numerical value for gradient clipping

                and other Keras optimizier parameters for the chosen optimizer
        """
        super().__init__()
        self._lr = lr

        if opt_type.lower() == 'adam':
            self._opt = optimizers.Adam(lr=lr, **kwargs)

        elif opt_type.lower() == 'rmsprop':
            self._opt = optimizers.RMSprop(lr=lr, **kwargs)

        elif opt_type.lower() == 'sgd':
            self._opt = optimizers.SGD(lr=lr, **kwargs)

        else:
            raise NotImplementedError

    @property
    def opt(self):
        return self._opt

    def model(self, n_input, n_output, opt_feature_extract):
        """
        Default model based on our paper
        """

        if int(tf.__version__[0]) > 1:
            return self.model_tf_v2(n_input, n_output, opt_feature_extract)

        else:
            return self.model_tf_v1(n_input, n_output, opt_feature_extract)


    def model_tf_v2(self, n_input, n_output, opt_feature_extract):
        """
        tensorflow 2.0.0 CuDNNGRU layers are deprecated
        instead the CuDNN implementation is used by default if:
        1. `activation` == `tanh`
        2. `recurrent_activation` == `sigmoid`
        3. `recurrent_dropout` == 0
        4. `unroll` is `False`
        5. `use_bias` is `True`
        6. Inputs are not masked or strictly right padded.
        7. reset_after == True
        """

        from tensorflow.python.keras.layers import Input, Bidirectional, GRU, Dense, Dropout

        session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=session_config)

        K.set_floatx('float64')
        ecg_input = Input(shape=(None, 1), dtype='float64', name='ecg_input')

    def model_tf_v1(self):
        from tensorflow.python.keras.layers import Input, Bidirectional, CuDNNGRU, Dense, Dropout

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=session_config)

        K.set_floatx('float64')
        ecg_input = Input(shape=(None, 1), dtype='float64', name='ecg_input')

        x = Bidirectional(CuDNNGRU(16, return_sequences=True,
                                   recurrent_regularizer=regularizers.l2(0.096),
                                   activity_regularizer=regularizers.l2(0.030)))(ecg_input)

        x = Bidirectional(CuDNNGRU(16, return_sequences=True,
                                   recurrent_regularizer=regularizers.l2(0.090),
                                   activity_regularizer=regularizers.l2(0.013)))(x)

        x = Dense(8, activation='relu')(x)
        x = Dropout(0.327)(x)

        x = Bidirectional(CuDNNGRU(16, return_sequences=True,
                                   recurrent_regularizer=regularizers.l2(0.024),
                                   activity_regularizer=regularizers.l2(0.067)))(x)

        x = Bidirectional(CuDNNGRU(64, return_sequences=True,
                                   recurrent_regularizer=regularizers.l2(2.48e-07),
                                   activity_regularizer=regularizers.l2(0.055)))(x)

        bcg_out = Dense(63, activation='linear')(x)
        model = Model(inputs=ecg_input, outputs=bcg_out)

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()

        return model





if __name__ == '__main__':
    """ used for debugging """

    model = rnn_model(lr=0.01, opt_type='adam')
    model.model()
