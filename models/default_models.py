import tensorflow as tf

if int(tf.__version__[0]) > 1:
    from tensorflow.keras import optimizers
    from tensorflow.keras import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import backend as K

else:
    from tensorflow.python.keras import optimizers
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.regularizers import l2
    from tensorflow.python.keras import backend as K


class NNModel:
    def __init__(self):
        self.name = None
        self.model = None
        self.optimizer = None

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
    def __init__(self, n_input=1, n_output=63, lr=1e-3, opt_type='adam', **kwargs):
        """
        Constructor for RNN model

        :param int n_input: number of input dimensions (number of ECG + aux channels)
        :param int n_output: number of output (number of EEG channels)
        :param float lr: learning rate
        :param str opt_type: chosen type of optimizer, allowed to be adam, rmsprop or sgd
        :param kwargs: clipnorm: normalized value for gradient clipping
                       clipvalue: numerical value for gradient clipping

                       and other Keras optimizier parameters for the chosen optimizer

        :return: initialized model object

        """
        super().__init__()
        self.name = 'default_rnn_model'
        self.lr = lr
        self.opt_type = opt_type.lower()
        self.n_input = n_input
        self.n_output = n_output

        K.set_floatx('float64')

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
            self.model = self._model_tf_v2(self.n_input, self.n_output)

        else:
            self.model = self._model_tf_v1(self.n_input, self.n_output)

    @staticmethod
    def _model_tf_v2(n_input, n_output):
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

        Additional Note:
        all the regularization terms have been omitted due to instability encountered when using
        the same regularization parameters as in the TF1 model. The model performance was compared
        using multiple subjects and were largely similar

        :param int n_input: number of input dimensions (number of ECG + aux channels)
        :param int n_output: number of output (number of EEG channels)

        :return: initialized model
        """

        from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense, Dropout

        session_config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
        sess = tf.compat.v1.Session(config=session_config)

        K.set_floatx('float64')
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

        return model

    @staticmethod
    def _model_tf_v1(n_input, n_output):
        """

        Initialize the tensorflow 1.1X version of the model

        :param int n_input: number of input dimensions (number of ECG + aux channels)
        :param int n_output: number of output (number of EEG channels)

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

        :param tensorflow.keras.optimizers optimizer: optional provided optimizer if wish to use type different
            from the default
        :param str loss: type of loss metric used
        :param kwargs: additional arguments that are accepted by keras compile function

        """

        if optimizer is not None:
            self.optimizer = optimizer

        self.model.compile(optimizer='adam', loss=loss, **kwargs)
        self.model.summary()

    def save_model_weights(self, p_weights, f_weights, overwrite=True, format='tf'):
        """
        save the model weights. If TF 1.X, save to the h5 format and if TF 2.X can use the user provided format

        :param pathlib.Path p_weights: absolute path to the directory to save the model weights in
        :param str f_weights: filename of the saved weights
        ;:param bool overwrite: whether or not to overwrite any existing model weights
        :param str format: format to save the weights in
        """

        if int(tf.__version__[0]) > 1:
            self.model.save_weights(filepath=str(p_weights / f_weights), overwrite=overwrite, save_format=format)

        else:
            self.model.save_weights(filepath=str(p_weights / f_weights), overwrite=overwrite, save_format='h5')

    def load_model_weights(self, p_weights, f_weights, **kwargs):
        """
        load model weights from a saved weight file

        :param pathlib.Path p_weights: absolute path to the directory to save the model weights in
        :param str f_weights: filename of the saved weights
        :param kwargs: additional arguments accepted by the tensorflow.keras.Model.load_weights() function
        """

        try:
            if int(tf.__version__[0]) > 1:
                self.model.load_weights(filepath=str(p_weights / f_weights), **kwargs)

            else:
                self.model.save_weights(filepath=str(p_weights / f_weights), **kwargs)

        except:
            raise RuntimeError("Issue encountered while loading saved weights")


if __name__ == '__main__':
    """ used for debugging """

    model = RNNModel(lr=0.01, n_input=1, n_output=63, opt_type='adam')
    model.init_model()
    model.compile_model()

    print('nothing')
