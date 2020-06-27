import numpy as np
from tensorflow.python.keras.utils import Sequence


class DefaultGenerator(Sequence):
    """
    Create the default generator class using Keras sequence as the parent
    """

    def __init__(self, x_data, y_data, batch_size=1, shuffle=True):
        """
        Class initialization

        :param x_data: ECG data from session, validation or test set in the form of (epoch, data) [(sample, features)]
        :param y_data: EEG data from session, validation or test set in the form of (epoch, channel, data) or
            alternatively [(sample, channel, data]
        :param batch_size: batch size to be used in session, default is 1
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