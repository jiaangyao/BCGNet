import tensorflow as tf
from tensorflow.python.keras import callbacks, optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
import os


class NNModel:
    def __init__(self):
        self.name = None
        self.model = None
        self.optimizer = None

    @staticmethod
    def get_name():
        """
        Get the name of the arch.

        :return: The name of the arch.
        """
        filename = os.path.basename(__file__)
        return os.path.splitext(filename)[0]

    @staticmethod
    def init_model(self):
        raise NotImplementedError

    def disable(self):
        for layer in self.model.layers:
            layer.trainable = False
        self.model.trainable = False

    def enable(self):
        for layer in self.model.layers:
            layer.trainable = True
        self.model.trainable = True


class RNNModel(NNModel):
    def __init__(self, n_input=1, n_output=63, lr=1e-2, opt_type='adam', opt_feature_extract=None, **kwargs):
        """
        Constructor for RNN model

        :param int n_input: number of input dimensions (number of ECG + aux channels)
        :param int n_output: number of output (number of EEG channels)
        :param float lr: learning rate
        :param str opt_type: chosen type of optimizer, allowed to be adam, rmsprop or sgd
        :param object opt_feature_extract: option object from feature extraction step
        :param kwargs: clipnorm: normalized value for gradient clipping
                       clipvalue: numerical value for gradient clipping

                       and other Keras optimizier parameters for the chosen optimizer

        :return: initialized model object

        """
        super().__init__()
        self.name = self.get_name()
        self.lr = lr
        self.opt_type = opt_type.lower()
        self.n_input = n_input
        self.n_output = n_output
        self.opt_feature_extract = opt_feature_extract

        if opt_type.lower() == 'adam':
            self.optimizer = optimizers.Adam(lr=lr, **kwargs)

        elif opt_type.lower() == 'rmsprop':
            self.optimizer = optimizers.RMSprop(lr=lr, **kwargs)

        elif opt_type.lower() == 'sgd':
            self.optimizer = optimizers.SGD(lr=lr, **kwargs)

        else:
            raise NotImplementedError

    def init_model(self):
        """
        Initialize the default model based on our paper properly depending on the version of tensorflow

        """

        if int(tf.__version__[0]) > 1:
            self.model = self.model_tf_v2(self.n_input, self.n_output, self.opt_feature_extract)

        else:
            self.model = self.model_tf_v1(self.n_input, self.n_output, self.opt_feature_extract)

    # TODO: implement compatibility check with opt_feature_extract
    @staticmethod
    def model_tf_v2(n_input, n_output, opt_feature_extract):
        """
        Initialize the tensorflow 2.X version of the model

        Note:
        from tensorflow 2.0.0 onwards CuDNNGRU layers are deprecated
        instead the CuDNN implementation is used by default if:
        1. `activation` == `tanh`
        2. `recurrent_activation` == `sigmoid`
        3. `recurrent_dropout` == 0
        4. `unroll` is `False`
        5. `use_bias` is `True`
        6. Inputs are not masked or strictly right padded.
        7. reset_after == True

        here implementation also set to 1 to avoid issues with tf loss value blowing up

        :param int n_input: number of input dimensions (number of ECG + aux channels)
        :param int n_output: number of output (number of EEG channels)
        :param object opt_feature_extract: option object from feature extraction step

        :return: initialized model
        """

        from tensorflow.python.keras.layers import Input, Bidirectional, GRU, Dense, Dropout

        session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=session_config)

        K.set_floatx('float64')
        ecg_input = Input(shape=(None, n_input), dtype='float64', name='ecg_input')

        x = Bidirectional(GRU(16, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=1,
                              recurrent_regularizer=l2(0.096),
                              activity_regularizer=l2(0.030)))(ecg_input)

        x = Bidirectional(GRU(16, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=1,
                              recurrent_regularizer=l2(0.090),
                              activity_regularizer=l2(0.013)))(x)

        x = Dense(8, activation='relu')(x)
        x = Dropout(0.327)(x)

        x = Bidirectional(GRU(16, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=1,
                              recurrent_regularizer=l2(0.024),
                              activity_regularizer=l2(0.067)))(x)

        x = Bidirectional(GRU(64, activation='tanh', return_sequences=True,
                              recurrent_activation='sigmoid', recurrent_dropout=0,
                              unroll=False, use_bias=True, reset_after=True,
                              implementation=1,
                              recurrent_regularizer=l2(2.48e-07),
                              activity_regularizer=l2(0.055)))(x)

        bcg_out = Dense(n_output, activation='linear')(x)
        model = Model(inputs=ecg_input, outputs=bcg_out)

        return model

    # TODO: implement compatibility check with opt_feature_extract
    @staticmethod
    def model_tf_v1(n_input, n_output, opt_feature_extract):
        """

        Initialize the tensorflow 1.1X version of the model

        :param int n_input: number of input dimensions (number of ECG + aux channels)
        :param int n_output: number of output (number of EEG channels)
        :param object opt_feature_extract: option object from feature extraction step

        :return: initialized model
        """
        from tensorflow.python.keras.layers import Input, Bidirectional, CuDNNGRU, Dense, Dropout

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess = tf.Session(config=session_config)

        K.set_floatx('float64')
        ecg_input = Input(shape=(None, n_input), dtype='float64', name='ecg_input')

        x = Bidirectional(CuDNNGRU(16, return_sequences=True,
                                   recurrent_regularizer=l2(0.096),
                                   activity_regularizer=l2(0.030)))(ecg_input)

        x = Bidirectional(CuDNNGRU(16, return_sequences=True,
                                   recurrent_regularizer=l2(0.090),
                                   activity_regularizer=l2(0.013)))(x)

        x = Dense(8, activation='relu')(x)
        x = Dropout(0.327)(x)

        x = Bidirectional(CuDNNGRU(16, return_sequences=True,
                                   recurrent_regularizer=l2(0.024),
                                   activity_regularizer=l2(0.067)))(x)

        x = Bidirectional(CuDNNGRU(64, return_sequences=True,
                                   recurrent_regularizer=l2(2.48e-07),
                                   activity_regularizer=l2(0.055)))(x)

        bcg_out = Dense(n_output, activation='linear')(x)
        model = Model(inputs=ecg_input, outputs=bcg_out)

        return model

    def compile_model(self, optimizer=None, loss='mean_squared_error', **kwargs):
        """
        Compile the model based on optional provided optimizer and loss metric

        :param tensorflow.python.keras.optimizers optimizer: optional provided optimizer if wish to use type different
            from the default
        :param str loss: type of loss metric used
        :param kwargs: additional arguments that are accepted by keras compile function

        """

        if optimizer is not None:
            self.optimizer = optimizer

        self.model.compile(optimizer=self.optimizer, loss=loss, **kwargs)
        self.model.summary()


if __name__ == '__main__':
    """ used for debugging """

    model = RNNModel(lr=0.01, n_input=1, n_output=63, opt_type='adam')
    model.init_model()
    model.compile_model()

    print('nothing')
