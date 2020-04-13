import numpy as np
from tensorflow.python.keras.layers import Input, Bidirectional, CuDNNGRU, Dense, Dropout
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import Sequence


def get_arch_rnn(str_arch='gru_arch_general4', lr=1e-3):
    """
    Obtain the optimal RNN model that's reported by the original paper

    :param str_arch: name of the architecture to be initialized
    :param lr: learning rate for the optimizers

    :return: model: RNN model that's initialized
    """

    if str_arch == 'gru_arch_general4':
        # Multi-run, no motion, simple

        K.set_floatx('float64')
        ecg_input = Input(shape=(None, 1), dtype='float64', name='ecg_input')

        gru1_out = Bidirectional(CuDNNGRU(16, return_sequences=True, recurrent_regularizer=l2(0.096),
                                          activity_regularizer=l2(0.030)))(ecg_input)

        gru2_out = Bidirectional(CuDNNGRU(16, return_sequences=True, recurrent_regularizer=l2(0.090),
                                          activity_regularizer=l2(0.013)))(gru1_out)

        d3_out = Dense(8, activation='relu')(gru2_out)
        d3_out_do = Dropout(0.327)(d3_out)

        gru3_out = Bidirectional(CuDNNGRU(16, return_sequences=True, recurrent_regularizer=l2(0.024),
                                          activity_regularizer=l2(0.067)))(d3_out_do)

        gru4_out = Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=l2(2.48e-07),
                                          activity_regularizer=l2(0.055)))(gru3_out)

        bcg_out = Dense(63, activation='linear')(gru4_out)
        model = Model(inputs=ecg_input, outputs=bcg_out)

        optimizer_custom = Adam(lr=lr)
        model.compile(loss='mean_squared_error', optimizer=optimizer_custom)
        model.summary()

    else:
        raise Exception("Undefined network arch: {}".format(str_arch))

    return model


def get_callbacks_rnn(opt):
    """
    Setup the early stopping that will be used in training

    :param opt: custom opt class that holds the options for training related hyperparameters
    :return: callbacks_: list containing the early stopping object defined with parameters from opt
    """

    # Make a local copy of the options file
    opt_local = opt

    # Generate the early stopping objects based on specfications from the options file
    callbacks_ = [callbacks.EarlyStopping(monitor='val_loss', min_delta=opt_local.es_min_delta,
                                          patience=opt_local.es_patience, verbose=0, mode='min',
                                          restore_best_weights=True)]

    return callbacks_


# TODO: investigate whether preprocessing can be integrated into this
class Defaultgenerator(Sequence):
    """
    Create the default generator class using Keras sequence as the parent
    """

    def __init__(self, x_data, y_data, batch_size=1, shuffle=True):
        """
        Class initialization

        :param x_data: ECG data from training, validation or test set in the form of (epoch, data) [(sample, features)]
        :param y_data: EEG data from training, validation or test set in the form of (epoch, channel, data) or
            alternatively [(sample, channel, data]
        :param batch_size: batch size to be used in training, default is 1
        :param shuffle: whether to shuffle the set, default is True
        """

        self.x = x_data
        self.y = y_data
        self.indices = np.arange(np.shape(self.x)[0])
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        """
        Obtain the steps needed per epoch

        :return:
        """
        return int(np.ceil(np.shape(self.x)[0]) / float(self.batch_size))

    def __getitem__(self, idx):
        """
        Generate each batch

        :param idx: starting index of current minibatch (needed by Keras)

        :return: batch_x: ECG data from current minibatch in the form of
        :return: batch_y: EEG data from current minibatch
        """

        # Obtain the indices of the samples that correspond to the current minibatch
        idx_batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Obtain the minibatch ECG data in the form of (sample, features) and note that Keras wants data in the form of
        # (sample, features, 1) so need to reshape the data
        batch_x = self.x[idx_batch, :]
        batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], 1)

        # Obtain the minibatch EEG data in the form of (sample, channels, features) and note that Keras wants data in
        # the form of (sample, features, channels) so need to transpose the data
        batch_y = self.y[idx_batch, :, :]
        batch_y = np.transpose(batch_y, axes=(0, 2, 1))

        return batch_x, batch_y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch by shuffling if enabled
        """

        if self.shuffle:
            np.random.shuffle(self.indices)
