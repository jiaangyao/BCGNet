import mne
import numpy as np
import tensorflow as tf

import scipy.io as sio
from scipy.stats import median_absolute_deviation
from utils import temp_seed
from dataset import interpolate_raw_dataset, compute_rms

"""
Each dataset object contains a single run of data from a single subject loaded by mne
"""

# TODO: think about whether or not to implement cross validation mode


class DefaultDataset:
    # TODO: check whether cfg is correctly used here and if it's correctly documented in the docstring
    # TODO: clean up the instance variable definition a little bit in the end
    def __init__(self, d_input, str_sub, idx_run, d_eval=None, str_eval=None,
                 random_seed=1997, cv_mode=False, num_fold=None, cfg=None):
        """
        Load in the dataset and resample if needed

        :param pathlib.Path d_input: pathlib object containing absolute path to the single run of data
        :param cfg: configuration file containing all the hyperparameter information

        """

        self.d_input = d_input
        self.str_sub = str_sub
        self.idx_run = idx_run
        self.d_eval = d_eval
        self.str_eval = str_eval
        self.random_seed = random_seed
        self.cv_mode = cv_mode
        self.cfg = cfg

        self.new_fs = cfg.new_fs
        self.len_epoch = cfg.len_epoch
        self.mad_threshold = cfg.mad_threshold
        self.per_training = cfg.per_training
        self.per_valid = cfg.per_valid
        self.per_test = cfg.per_test

        if self.per_training + self.per_valid + self.per_test > 1:
            raise RuntimeError("Total percentage greater than 1")

        # Load the various dataset
        self.raw_dataset = self._load_dataset(d_input)
        self.eval_dataset = None
        if self.d_eval is not None:
            self.eval_dataset = self._load_dataset(self.d_eval)
            self.str_eval = str_eval
        self.fs = self.raw_dataset.info['sfreq']

        # Resample if needed
        self.resampled = False
        if self.new_fs is not None:
            if self.new_fs != self.fs:
                print("\nResample from {} Hz to {} Hz".format(self.fs, self.new_fs))
                self.raw_dataset.resample(self.new_fs)

                self.orig_fs = self.fs
                self.orig_raw_dataset = self._load_dataset(d_input, verbose=False)
                self.resampled = True

                self.epoched_orig_cleaned_dataset = None
                self.orig_cleaned_dataset = None

        self.standardized_dataset = None
        self.epoched_standardized_dataset = None
        self.epoch_rejected = False
        self.vec_idx_good_epochs = None

        self.epoched_cleaned_dataset = None
        self.cleaned_dataset = None
        self.rms_results = {}

        self.cv_mode = cv_mode
        if cv_mode:
            self.vec_xs = None
            self.vec_ys = None
            self.mat_idx_slice = None

            self.vec_epoched_orig_cleaned_dataset = None
            self.vec_orig_cleaned_dataset = None
            self.vec_epoched_cleaned_dataset = None
            self.vec_cleaned_dataset = None

            if num_fold is not None:
                self.num_fold = num_fold
            else:
                self.num_fold = int(np.round(1 / self.per_test))

        else:
            self.xs = None
            self.ys = None
            self.vec_idx_slice = None

        # TODO: delete the following snippet (since the user doesn't have any motion data)
        # TODO: put the channel names in a meta file somewhere
        self.raw_dataset.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])
        if self.resampled:
            self.orig_raw_dataset.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])

    @staticmethod
    def _load_dataset(d_run, **kwargs):
        """
        Private method for loading in a single run of data from a single subject

        :param pathlb.Path d_run: pathlib object containing the absolute path to the single run of data
        :param **kwargs: other keyword arguments for mne.io.read_raw_eeglab()

        :return: mne.io.RawArray object holding the raw data
        """

        if d_run.suffix == '.set' and 'all' not in d_run.stem and d_run.stat().st_size > 0:
            run_dataset = mne.io.read_raw_eeglab(str(d_run), preload=True, **kwargs)

            return run_dataset
        else:
            raise RuntimeError("Invalid dataset")

    def prepare_dataset(self):
        """
        Perform standardization, epoching and epoch rejection on the current dataset

        :return:
        """
        # standardize the dataset first
        self._standardize_dataset()

        # then perform epoching and epoch rejection
        self._epoch_dataset()

    def _standardize_dataset(self):
        """
        Perform standardization on the object
        """

        standardized_dataset, ecg_stats, eeg_stats = self._standardize_raw_dataset(self.raw_dataset)

        self.standardized_dataset = standardized_dataset
        self.ecg_stats = ecg_stats
        self.eeg_stats = eeg_stats

    @staticmethod
    def _standardize_raw_dataset(raw_dataset):
        """
        Performs renormalization of the raw dataset by whitening each channel (subtracting the mean of each channel and
        then divide by the standard deviation (std) of each channel

        :param mne.io.RawArray raw_dataset: object that contains the unstandardized raw data

        :return: a tuple (standardized_dataset, ecg_stats, eeg_stats), where standardized_dataset is the object
            that contains standardized data by channels, ecg_stats is a list [ecg_mean, ecg_std] and eeg_stats
            is a list [[eeg_mean1, eeg_mean2,...], [eeg_std1, eeg_std2,...]]
        """

        # obtain the data numpy array and information structure from the MNE Raw object
        data = raw_dataset.get_data()
        info = raw_dataset.info

        # obtain the number of the channel that holds the ECG channel and the channel index
        ecg_ch = info['ch_names'].index('ECG')
        target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

        # used for reverting back to original data later
        ecg_mean = np.mean(data[ecg_ch, :])
        ecg_std = np.std(data[ecg_ch, :])
        eeg_mean = np.mean(data[target_ch, :], axis=1)
        eeg_std = np.std(data[target_ch, :], axis=1)

        # standardize the data
        standardized_raw_data = np.zeros(data.shape)
        for i in range(data.shape[0]):
            ds = data[i, :] - np.mean(data[i, :])
            ds /= np.std(ds)
            standardized_raw_data[i, :] = ds

        # create another mne object
        standardized_raw_dataset = mne.io.RawArray(standardized_raw_data, info, verbose=False)

        ecg_stats = [ecg_mean, ecg_std]
        eeg_stats = [eeg_mean, eeg_std]

        return standardized_raw_dataset, ecg_stats, eeg_stats

    @staticmethod
    def _unstandardize_ecg_data(standardized_ecg_data, ecg_stats):
        """
        unstandardize the ecg data

        :param np.ndarray standardized_ecg_data: standardized ecg data with shape (n_samples,) or (1, n_samples, 1)
        :param list ecg_stats: a list [ecg_mean, ecg_std]

        :return: unstandardized ecg data with same shape as input
        """

        unstandardized_ecg_data = standardized_ecg_data * ecg_stats[1] + ecg_stats[0]

        return unstandardized_ecg_data

    @staticmethod
    def _unstandardize_eeg_data(standardized_ecg_data, eeg_stats):
        """
        unstandardize the eeg data

        :param np.ndarray standardized_ecg_data: standardized eeg data with shape (n_channels, n_samples)
        :param list eeg_stats: a list [[eeg_mean1, eeg_mean2,...], [eeg_std1, eeg_std2,...]]

        :return: unstandardized eeg data with same shape as input
        """

        # Create empty array same size as input
        unstandardized_eeg_data = np.zeros(standardized_ecg_data.shape)

        # Loop through the channels of the input
        for i in range(standardized_ecg_data.shape[0]):
            # For each channel, perform the renormalization
            unstandardized_eeg_data[i, :] = standardized_ecg_data[i, :] * eeg_stats[1][i] + eeg_stats[0][i]

        return unstandardized_eeg_data

    def _epoch_dataset(self):
        """
        performs epoching of the dataset
        """

        if self.standardized_dataset is None:
            raise RuntimeError("Need to run standardization first")

        # perform epoching and epoch rejection
        self.epoched_standardized_dataset, self.vec_idx_good_epochs, \
            self.epoch_rejected = self._perform_epoching(self.standardized_dataset, self.len_epoch, self.mad_threshold)

        # extract the equivalent epochs from the two other raw datasets
        self.epoched_raw_dataset = self._extract_good_epochs(self.raw_dataset, self.len_epoch,
                                                             self.vec_idx_good_epochs)

        if self.resampled:
            self.epoched_orig_raw_dataset = self._extract_good_epochs(self.orig_raw_dataset, self.len_epoch,
                                                                      self.vec_idx_good_epochs)

        if self.eval_dataset is not None:
            self.epoched_eval_dataset = self._extract_good_epochs(self.eval_dataset, self.len_epoch,
                                                                  self.vec_idx_good_epochs)

    @staticmethod
    def _perform_epoching(raw_dataset, len_epoch, mad_threshold):
        """
        Perform epoching of a raw dataset and perform rejection of outlier epochs based on mean absolute deviation

        :param mne.io.RawArray raw_dataset: object holding the dataset for which epoching operation is to be performed,
            used for obtaining the length in time of the original recording
        :param int/float len_epoch: length of each epoch in seconds
        :param mad_threshold: # times the mean absolute deviation to set the threshold for MAD-based epoch rejection

        :return: a tuple (epoched_dataset, vec_idx_good_epochs, epoch_rejected) where epoched_dataset is an
            mne.EpochsArray object that holds all epochs that passed epoch rejection, vec_idx_good_epochs is a list
            that contains the indices of all epochs that passed the rejection in the original epoched dataset, and
            epoch_rejected is a boolean flag that indicates whether or not any epochs where actually rejected
        """

        # obtain the number of samples in the original recording and the sampling rate
        len_recording = raw_dataset.get_data().shape[1]
        fs = raw_dataset.info['sfreq']

        # constructing events of fixed interval apart
        constructed_events, tmax = DefaultDataset._construct_epoch_events(len_recording, fs, len_epoch)

        # perform epoching using the MNE package functionality and splits the dataset into time windows of equal length
        orig_epoched_dataset = mne.Epochs(raw_dataset, constructed_events, tmin=0, tmax=tmax,
                                          baseline=None, reject=None, verbose=False)
        epoch_rejected = False

        # perform epoch rejection
        vec_idx_good_epochs = DefaultDataset._perform_epoch_rejection(orig_epoched_dataset, mad_threshold)

        # check if anything actually got rejected
        if len(vec_idx_good_epochs) != orig_epoched_dataset.get_data().shape[0]:
            epoch_rejected = True

            good_epocehd_data = orig_epoched_dataset.get_data()[vec_idx_good_epochs, :, :]
            epoched_dataset = mne.EpochsArray(good_epocehd_data, orig_epoched_dataset.info, baseline=None,
                                              reject=None, verbose=False)

            return epoched_dataset, vec_idx_good_epochs, epoch_rejected

        return orig_epoched_dataset, vec_idx_good_epochs, epoch_rejected

    @staticmethod
    def _extract_good_epochs(raw_dataset, len_epoch, vec_idx_good_epochs):
        """
        Extract epochs that passed the rejection test

        :param mne.io.RawArray raw_dataset: object holding the dataset for which epoching operation is to be performed,
            used for obtaining the length in time of the original recording
        :param int/float len_epoch: length of each epoch in seconds
        :param numpy.ndarray vec_idx_good_epochs: list of epochs that passed the MAD-based rejection test

        :return: an mne.EpochsArray object that holds the epoched dataset where epochs equivalent to those that passed
            the rejection test are extracted
        """

        if vec_idx_good_epochs is None:
            raise RuntimeError("Need to perform rejection on the standardized data first")

        # obtain the number of samples in the original recording and the sampling rate
        len_recording = raw_dataset.get_data().shape[1]
        fs = raw_dataset.info['sfreq']

        # constructing events of fixed interval apart
        constructed_events, tmax = DefaultDataset._construct_epoch_events(len_recording, fs, len_epoch)

        # perform epoching using the MNE package functionality and splits the dataset into time windows of equal length
        orig_epoched_dataset = mne.Epochs(raw_dataset, constructed_events, tmin=0, tmax=tmax,
                                          baseline=None, verbose=False)

        # Then simply extract the epochs that are good using good_idx
        epoched_data = orig_epoched_dataset.get_data()[vec_idx_good_epochs, :, :]
        epoched_dataset = mne.EpochsArray(epoched_data, orig_epoched_dataset.info, baseline=None,
                                          reject=None, verbose=False)

        return epoched_dataset

    @staticmethod
    def _perform_epoch_rejection(epoched_dataset, mad_threshold):
        """
        Function for performing the mean absolute deviation (MAD) based epoch rejection

        :param mne.Epochs epoched_dataset: object holding epoched dataset for which MAD-based epoch rejection is
            to be performed
        :param int mad_threshold: # times the mean absolute deviation to set the threshold for MAD-based epoch rejection

        :return: indices of epochs that passed the epoch rejection test
        """

        # note that abs_epoched_data is of shape (n_epochs, n_channel, n_sample)
        abs_epoched_data = np.absolute(epoched_dataset.get_data())
        info = epoched_dataset.info
        ecg_ch = info['ch_names'].index('ECG')
        target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

        # Compute the ratio of each epoch's absolute value across all channels over its MAD
        vec_mabs_eeg = np.mean(abs_epoched_data[:, target_ch, :], axis=(1, 2))
        vec_eeg_norm = (vec_mabs_eeg - np.median(vec_mabs_eeg)) / median_absolute_deviation(vec_mabs_eeg)

        # If the ratio is higher than the threshold then the epoch is rejected
        vec_idx_bad_epochs = np.arange(0, len(vec_eeg_norm), 1)[vec_eeg_norm > mad_threshold]
        vec_idx_good_epochs = np.delete(np.arange(0, epoched_dataset.get_data().shape[0], 1), vec_idx_bad_epochs)

        print("\nRejecting {} epochs out of a total of {}".format(len(vec_idx_bad_epochs), abs_epoched_data.shape[0]))
        print("{} epochs remaining...\n".format(len(vec_idx_good_epochs)))

        return vec_idx_good_epochs

    @staticmethod
    def _construct_epoch_events(len_recording, fs, len_epoch):
        """
        Create events of fixed duration apart to split the original time series into time windows (epochs) of equal
        length

        :param int len_recording: number of samples in the recording
        :param int fs: sampling rate of the data
        :param int len_epoch: length of each epoch in seconds
        :return: a tuple (constructed_events, tmax), where constructed_events is a numpy array of shape (n_events, 3)
            containing the latency, duration and tag of fixed interval events for mne and tmax is the number of samples
            in each epoch
        """

        # create the empty numpy array to hold the events, of shape (floor(time/duration), 3)
        constructed_events = np.zeros(shape=(int(np.floor(len_recording / fs) / len_epoch), 3), dtype=int)

        # populate the constructed_events created with the starting index of each time window
        # the numbers 0 and 1 are for marking the event as fake for MNE
        # fake events are of custom latency, 0 duration and tag 999
        for i in range(0, int(np.floor(len_recording / fs)) - len_epoch, len_epoch):
            ix = i / len_epoch
            constructed_events[int(ix)] = np.array([i * fs, 0, 999])

        n_events = len(range(0, int(np.floor(len_recording / fs)) - len_epoch, len_epoch))
        if n_events < constructed_events.shape[0]:
            constructed_events = constructed_events[:n_events, :]

        # Delete the last sample to make the length of the time window even
        tmax = len_epoch - 1 / fs

        return constructed_events, tmax

    def split_dataset(self):
        if not self.cv_mode:
            self.xs, self.ys, self.vec_idx_slice \
                = DefaultDataset._generate_train_valid_test(self.epoched_standardized_dataset, self.per_training,
                                                            self.per_valid, self.random_seed)
        else:
            self.vec_xs, self.vec_ys, self.mat_idx_slice \
                = DefaultDataset._generate_train_valid_test_cv(self.epoched_standardized_dataset, self.per_valid,
                                                               self.num_fold, self.random_seed)

    @staticmethod
    def _generate_train_valid_test(epoched_dataset, per_training, per_valid, random_seed):
        """
        Generate the training, validation and test epoch data from the mne.EpochsArray object based on the training set
            ratio and validation set ratio and random seed provided

        :param mne.EpochsArray epoched_dataset: object where epoched_dataset.get_data() is the  normalized
            epoched data and is of the form of (epoch, channel, data)
        :param float per_training: percentage of training set epochs
        :param float per_valid: percentage of validation set epochs
        :param int random_seed: random seed used for the splitting of the dataset

        :return: a tuple (xs, ys, vec_ix_slice), where xs is the list containing all the ECG data in the form of
            [x_train, x_validation, x_test] and each has shape (epoch, channel, data), ys is a list containing all the
            corrupted EEG data in the form of [y_train, y_validation, y_test], and each has shape
            (epoch, channel, data), and vec_ix_slice is a list in the form of [vec_ix_slice_training,
            vec_ix_slice_validation, vec_ix_slice_test], where each element contains the indices of epochs in the
            original dataset belonging to the training, validation and test set respectively
        """

        # Obtain the data and the index of the ECG channel from the MNE Epoch object
        # Note that normalized_data is in the form (epoch, channel, data)
        epoched_data = epoched_dataset.get_data()
        ch_ecg = epoched_dataset.info['ch_names'].index('ECG')

        # Obtain the total number of epochs
        num_epochs = epoched_data.shape[0]

        # Temporarily set the random seed so that the data will be split in the same way
        with temp_seed(random_seed):
            # compute the number of epochs in each set
            num_training = int(np.round(num_epochs * per_training))
            num_valid = int(np.round(num_epochs * per_valid))

            # now compute the indices
            vec_idx = np.random.permutation(num_epochs)
            vec_idx_training = vec_idx[:num_training]
            vec_idx_valid = vec_idx[num_training: num_training + num_valid]
            vec_idx_test = vec_idx[num_training + num_valid:]

            # test for overlap
            overlap_1 = np.intersect1d(vec_idx_training, vec_idx_valid)
            overlap_2 = np.intersect1d(vec_idx_valid, vec_idx_test)
            overlap_3 = np.intersect1d(vec_idx_training, vec_idx_test)
            if len(overlap_1) > 0 or len(overlap_2) > 0 or len(overlap_3) > 0:
                raise RuntimeError("Overlap between training, validation and test sets")

            # obtain the epochs in each set
            epoched_data_training = epoched_data[vec_idx_training, :, :]
            epocehd_data_valid = epoched_data[vec_idx_valid, :, :]
            epocehd_data_test = epoched_data[vec_idx_test, :, :]

            # Obtain the ECG data in each set
            x_train = epoched_data_training[:, ch_ecg, :]
            x_validation = epocehd_data_valid[:, ch_ecg, :]
            x_test = epocehd_data_test[:, ch_ecg, :]

            # Obtain the EEG data in each set
            y_train = np.delete(epoched_data_training, ch_ecg, axis=1)
            y_validation = np.delete(epocehd_data_valid, ch_ecg, axis=1)
            y_test = np.delete(epocehd_data_test, ch_ecg, axis=1)

        # Package everything together into a list
        xs = [x_train, x_validation, x_test]
        ys = [y_train, y_validation, y_test]
        vec_idx_slice = [vec_idx_training, vec_idx_valid, vec_idx_test]

        return xs, ys, vec_idx_slice

    @staticmethod
    def _split_epoched_dataset(epoched_dataset, vec_idx_slice):
        """
        Split the mne.io_ops.Mne object holding the cleaned EEG data into the same set of training, validation and test
        epochs as during the training

        :param mne.EpochsArray epoched_dataset: object that holds the epoched data, note that the data held has the
            form (epoch, channel, data)
        :param vec_idx_slice: list in the form of [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test],
            where each element contains the indices of epochs in the original dataset belonging to the
            training, validation and test set respectively

        :return: a list in the form of [epoched_dataset_training, epoched_dataset_validation, epoched_dataset_test],
            where each is an mne.io_ops.Mne object that holds epoched data belonging to
            the specified set during training
        """

        # Get the data and info object first
        epoched_data = epoched_dataset.get_data()
        info = epoched_dataset.info

        # Get the training, validation and test data
        epoched_data_training = epoched_data[vec_idx_slice[0], :, :]
        epoched_data_validation = epoched_data[vec_idx_slice[1], :, :]
        epoched_data_test = epoched_data[vec_idx_slice[2], :, :]

        epoched_dataset_training = mne.EpochsArray(epoched_data_training, info, verbose=False)
        epoched_dataset_validation = mne.EpochsArray(epoched_data_validation, info, verbose=False)
        epoched_dataset_test = mne.EpochsArray(epoched_data_test, info, verbose=False)

        vec_epoched_dataset = [epoched_dataset_training, epoched_dataset_validation, epoched_dataset_test]

        return vec_epoched_dataset

    # TODO: maybe change per_validation to be same as 1/num_fold
    @staticmethod
    def _generate_train_valid_test_cv(epoched_dataset, per_valid, num_fold, random_seed):
        """
        Generate the training, validation and test epoch data from the MNE Epoch object based on the number of folds
            (related to test set ratio) and validation set ratio and random seed provided in a cross validation manner

        :param mne.EpochsArray epoched_dataset: object where epoched_dataset.get_data() is the  normalized
            epoched data and is of the form of (epoch, channel, data)
        :param float per_valid: percentage of validation set epochs
        :param int num_fold: number of cross validation folds
        :param int random_seed: random seed used for the splitting of the dataset

        :return a tuple (vec_xs, vec_ys, mat_ix_slice), where vec_xs is a list containing all the ECG data in
            the form of [fold1, fold2, ...] and each fold is in the form of [x_train, x_validation, x_test] and each
            has shape (epoch, data), where vec_ys is a list containing all the corrupted EEG data in the form of
            [fold1, fold2, ...] and each fold is in the form [y_train, y_validation, y_test] and each has
            shape (epoch, channel, data), and where mat_ix_slice is a list in the form of [fold1, fold2, ...], where
            each fold is of the form of [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test], where
            each element contains the indices of epochs in the original dataset belonging to the training, validation
            and test set respectively
        """

        # Obtain the data and the index of the ECG channel from the MNE Epoch object
        # Note that normalized_data is in the form (epoch, channel, data)
        epoched_data = epoched_dataset.get_data()
        ch_ecg = epoched_dataset.info['ch_names'].index('ECG')

        # Obtain the total number of epochs
        num_epochs = epoched_data.shape[0]

        # Temporarily set the random seed so that the data will be split in the same way
        with temp_seed(random_seed):
            # Split everything into int(ceil(1/per_fold)) number of folds of roughly equal sizes
            vec_idx_epoch = np.random.permutation(num_epochs)
            mat_idx_slice_test = np.array_split(vec_idx_epoch, num_fold)

            # Define the empty arrays to hold the variables
            mat_idx_slice = []
            vec_xs = []
            vec_ys = []

            # Loop through each fold and determine the validation set and training set for each fold according to
            # defined percentages
            for i in range(len(mat_idx_slice_test)):
                vec_idx_test = mat_idx_slice_test[i]
                if not np.all(np.isin(vec_idx_test, vec_idx_epoch)):
                    raise Exception('Erroneous CV fold splitting')
                epocehd_data_test = epoched_data[vec_idx_test, :, :]

                # Obtain the indices that correspond to training + validation set and permute it
                vec_idx_tv = np.setdiff1d(vec_idx_epoch, vec_idx_test)
                permuted_vec_tv = vec_idx_tv[np.random.permutation(len(vec_idx_tv))]

                # Obtain the validation epochs
                num_valid = int(np.round(num_epochs * per_valid))
                vec_idx_valid = permuted_vec_tv[:num_valid]
                epocehd_data_valid = epoched_data[vec_idx_valid, :, :]

                # Obtain the training epochs
                vec_idx_training = permuted_vec_tv[num_valid:]
                epocehd_data_training = epoched_data[vec_idx_training, :, :]

                # test for overlap
                overlap_1 = np.intersect1d(vec_idx_training, vec_idx_valid)
                overlap_2 = np.intersect1d(vec_idx_valid, vec_idx_test)
                overlap_3 = np.intersect1d(vec_idx_training, vec_idx_test)
                if len(overlap_1) > 0 or len(overlap_2) > 0 or len(overlap_3) > 0:
                    raise RuntimeError("Overlap between training, validation and test sets")

                # Obtain the xs and the ys
                x_training = epocehd_data_training[:, ch_ecg, :]
                x_validation = epocehd_data_valid[:, ch_ecg, :]
                x_test = epocehd_data_test[:, ch_ecg, :]

                y_training = np.delete(epocehd_data_training, ch_ecg, axis=1)
                y_validation = np.delete(epocehd_data_valid, ch_ecg, axis=1)
                y_test = np.delete(epocehd_data_test, ch_ecg, axis=1)

                # Package everything into a single list
                xs = [x_training, x_validation, x_test]
                ys = [y_training, y_validation, y_test]
                vec_idx_slice = [vec_idx_training, vec_idx_valid, vec_idx_training]

                # Append those to the outer lists holding everything
                vec_xs.append(xs)
                vec_ys.append(ys)
                mat_idx_slice.append(vec_idx_slice)

        return vec_xs, vec_ys, mat_idx_slice

    # TODO: reconsider the _ variables here.... as well as the standardized data...
    def clean_dataset(self, model, vec_callbacks=None):
        """
        generate the cleaned datasets using provided model and callback objetcs

        :param tensorflow.keras.Model/tensorflow.keras.Sequential model: keras model that was trained
        :param list vec_callbacks: (optional) a list containing the early stopping object used during training
            (only relevant if TF 2.X is used)
        """

        if self.resampled:
            self.epoched_orig_cleaned_dataset, self.orig_cleaned_dataset, _, _ \
                = self._clean_dataset(model=model, vec_callbacks=vec_callbacks,
                                      orig_raw_dataset=self.orig_raw_dataset,
                                      standardized_dataset=self.standardized_dataset,
                                      raw_dataset=self.raw_dataset,
                                      resampled=self.resampled,
                                      ecg_stats=self.ecg_stats, eeg_stats=self.eeg_stats,
                                      len_epoch=self.len_epoch,
                                      vec_idx_good_epochs=self.vec_idx_good_epochs)
        else:
            _, _, self.epoched_cleaned_dataset, self.cleaned_dataset = \
                self._clean_dataset(model=model, vec_callbacks=vec_callbacks,
                                    standardized_dataset=self.standardized_dataset,
                                    raw_dataset=self.raw_dataset,
                                    resampled=self.resampled,
                                    ecg_stats=self.ecg_stats, eeg_stats=self.eeg_stats,
                                    len_epoch=self.len_epoch,
                                    vec_idx_good_epochs=self.vec_idx_good_epochs)

        # clean up the standardized data instance variables to save memory
        self.standardized_dataset = None
        self.epoched_standardized_dataset = None

    # TODO: reconsider the _ variables here.... as well as the standardized data...
    def clean_dataset_cv(self, vec_model, vec_callbacks=None):
        """
        generate the cleaned datasets using provided model and callback objetcs

        :param list vec_model: list of keras model that was trained
        :param list vec_callbacks: (optional) a list containing the early stopping object used during training
            (only relevant if TF 2.X is used)
        """

        if self.resampled:
            vec_epoched_orig_cleaned_dataset = []
            vec_orig_cleaned_dataset = []

            for i in range(len(vec_model)):
                epoched_orig_cleaned_dataset, orig_cleaned_dataset, _, _ \
                    = self._clean_dataset(model=vec_model[i], vec_callbacks=vec_callbacks,
                                          orig_raw_dataset=self.orig_raw_dataset,
                                          standardized_dataset=self.standardized_dataset,
                                          raw_dataset=self.raw_dataset,
                                          resampled=self.resampled,
                                          ecg_stats=self.ecg_stats, eeg_stats=self.eeg_stats,
                                          len_epoch=self.len_epoch,
                                          vec_idx_good_epochs=self.vec_idx_good_epochs)

                vec_epoched_orig_cleaned_dataset.append(epoched_orig_cleaned_dataset)
                vec_orig_cleaned_dataset.append(orig_cleaned_dataset)

            self.vec_epoched_orig_cleaned_dataset = vec_epoched_orig_cleaned_dataset
            self.vec_orig_cleaned_dataset = vec_orig_cleaned_dataset
        else:
            vec_epoched_cleaned_dataset = []
            vec_cleaned_dataset = []

            for i in range(len(vec_model)):
                _, _, epoched_cleaned_dataset, cleaned_dataset = \
                    self._clean_dataset(model=vec_model[i], vec_callbacks=vec_callbacks,
                                        standardized_dataset=self.standardized_dataset,
                                        raw_dataset=self.raw_dataset,
                                        resampled=self.resampled,
                                        ecg_stats=self.ecg_stats, eeg_stats=self.eeg_stats,
                                        len_epoch=self.len_epoch,
                                        vec_idx_good_epochs=self.vec_idx_good_epochs)
                vec_epoched_cleaned_dataset.append(epoched_cleaned_dataset)
                vec_cleaned_dataset.append(cleaned_dataset)

            self.vec_epoched_cleaned_dataset = vec_epoched_cleaned_dataset
            self.vec_cleaned_dataset = vec_cleaned_dataset

        # clean up the standardized data instance variables to save memory
        self.standardized_dataset = None
        self.epoched_standardized_dataset = None

    @staticmethod
    def _clean_dataset(model, vec_callbacks, standardized_dataset, raw_dataset, resampled,
                       ecg_stats, eeg_stats, len_epoch, vec_idx_good_epochs, orig_raw_dataset=None):
        """
        Generates the unstandardized ECG and predicted BCG time series

        :param tensorflow.keras.Model/tensorflow.keras.Sequential model: keras model that was trained
        :param list vec_callbacks: a list containing the early stopping object used during training
        :param mne.io.RawArray standardized_dataset: the object that contains standardized data by channels
        :param mne.io.RawArray: raw_dataset: the object that holds
        :param bool resampled: whether or not the dataset was resampled
        :param list ecg_stats: ecg_stats: list in the form of [mean_ECG, std_ECG]
        :param list eeg_stats: input list in the form of [[eeg_ch1_mean, eeg_ch2_mean, ...],
            [eeg_ch1_std, eeg_ch2_std, ...]]
        :param int len_epoch: length of each epoch in seconds
        :param np.ndarray vec_idx_good_epochs: indices of epochs that passed the epoch rejection test
        :param mne.io.RawArray orig_raw_dataset: (optional) object that holds the unstandardized data from the original
            dataset (only relevant when the dataset is resampled)

        :return a tuple (epoched_orig_cleaned_dataset, orig_cleaned_dataset, epoched_cleaned_dataset, cleaned_dataset),
            where epoched_orig_cleaned_dataset and orig_cleaned_dataset are the epoched and time series version
            of cleaned data interpolated to original sampling rate and epoched_cleaned_dataset, cleaned_dataset
            are epoched and time series version of cleaned data with the resampled sampling rate or a tuple
            (None, None, epoched_cleaned_dataset, cleaned_dataset) is no resampling was performed
        """

        # Obtain the normalized raw data and the info object holding the channel information
        standardized_data = standardized_dataset.get_data()
        info = standardized_dataset.info

        # Obtain the index of the ECG channel
        ch_ecg = info['ch_names'].index('ECG')

        # Obtain the indices of all the EEG channels
        ch_eeg = np.delete(np.arange(0, len(info['ch_names']), 1), ch_ecg)

        # Obtain the standardized ECG and EEG data
        # get the ECG data and perform reshape so that the shape works with Keras
        standardized_ecg_data = standardized_data[ch_ecg, :].reshape(1, standardized_data.shape[1], 1)

        # get the EEG data, no need to do any transformation here, note that data in the form (channel, data)
        standardized_eeg_data = standardized_data[ch_eeg, :]

        # Predict the BCG data in all EEG channels, note that since Keras generates data in the form of
        # (1, data, channel), and a transpose is needed at the end to convert to channel-major format
        if int(tf.__version__[0]) > 1:
            predicted_bcg_data = model.predict(x=standardized_ecg_data, callbacks=vec_callbacks, verbose=0)

        else:
            predicted_bcg_data = model.predict(x=standardized_ecg_data, verbose=0)
        predicted_bcg_data = np.transpose(
            predicted_bcg_data.reshape(predicted_bcg_data.shape[1], predicted_bcg_data.shape[2]))

        # Obtain the cleaned EEG data
        standardized_cleaned_eeg_data = standardized_eeg_data - predicted_bcg_data

        # Undo the normalization
        ecg_data = DefaultDataset._unstandardize_ecg_data(standardized_ecg_data, ecg_stats)
        cleaned_eeg_data = DefaultDataset._unstandardize_eeg_data(standardized_cleaned_eeg_data, eeg_stats)

        # Check if the normalization is performed normally
        orig_ecg_data = raw_dataset.get_data()[ch_ecg, :].reshape(1, raw_dataset.get_data().shape[1], 1)
        if not np.allclose(ecg_data, orig_ecg_data):
            raise Exception('Normalization failed during prediction')

        # reshape to make dimension correct
        ecg_data = ecg_data.reshape(-1)

        # If performed normally, then generate an mne.io_ops.RawArray object holding the cleaned data
        cleaned_data = np.insert(cleaned_eeg_data, ch_ecg, ecg_data, axis=0)
        cleaned_dataset = mne.io.RawArray(cleaned_data, info, verbose=False)

        # Obtain the data from the ground truth dataset that corresponds to epochs that passed the MAD rejection
        # and that are used in training the model
        epoched_cleaned_dataset = DefaultDataset._extract_good_epochs(cleaned_dataset, len_epoch, vec_idx_good_epochs)

        if resampled:
            # Interpolate the dataset and perform the same operation
            orig_cleaned_dataset = interpolate_raw_dataset(cleaned_dataset, orig_raw_dataset)
            epoched_orig_cleaned_dataset = DefaultDataset._extract_good_epochs(orig_cleaned_dataset, len_epoch,
                                                                               vec_idx_good_epochs)

            return epoched_orig_cleaned_dataset, orig_cleaned_dataset, epoched_cleaned_dataset, cleaned_dataset

        return None, None, epoched_cleaned_dataset, cleaned_dataset

    def evaluate_dataset(self, mode='test'):
        """
        Evaluate the performance of the model compared to raw and optional evaluation dataset and package all results
        into a dictionary

        :param str mode: either 'train', 'valid' or 'test', indicating which set to extract RMS value and
        power ratio from
        """

        if self.resampled:
            vec_epoched_raw_dataset = DefaultDataset._split_epoched_dataset(self.epoched_orig_raw_dataset,
                                                                            self.vec_idx_slice)

            vec_epoched_cleaned_dataset = DefaultDataset._split_epoched_dataset(self.epoched_orig_cleaned_dataset,
                                                                                self.vec_idx_slice)
        else:
            vec_epoched_raw_dataset = DefaultDataset._split_epoched_dataset(self.epoched_raw_dataset,
                                                                            self.vec_idx_slice)

            vec_epoched_cleaned_dataset = DefaultDataset._split_epoched_dataset(self.epoched_cleaned_dataset,
                                                                                self.vec_idx_slice)

        if self.eval_dataset is not None:
            vec_epoched_eval_dataset = DefaultDataset._split_epoched_dataset(self.epoched_eval_dataset,
                                                                             self.vec_idx_slice)

            vec_rms_set = compute_rms(self.idx_run, vec_epoched_raw_dataset, vec_epoched_cleaned_dataset,
                                      vec_epoched_eval_dataset=vec_epoched_eval_dataset,
                                      str_eval=self.str_eval, mode=mode, cfg=self.cfg)
        else:
            vec_rms_set = compute_rms(self.idx_run, vec_epoched_raw_dataset, vec_epoched_cleaned_dataset,
                                      mode=mode, cfg=self.cfg)

        self.rms_results[mode] = vec_rms_set

    def evaluate_dataset_cv(self, idx_fold, mode='test'):
        """
        Evaluate the performance of the model compared to raw and optional evaluation dataset and package all results
        into a dictionary for the cross validation mode

        :param int idx_fold: index of the fold (0 indexing)
        :param str mode: either 'train', 'valid' or 'test', indicating which set to extract RMS value and
        power ratio from
        """

        if self.resampled:
            vec_epoched_raw_dataset_tvt = DefaultDataset._split_epoched_dataset(self.epoched_orig_raw_dataset,
                                                                                self.mat_idx_slice[idx_fold])

            vec_epoched_cleaned_dataset_tvt = \
                DefaultDataset._split_epoched_dataset(self.vec_epoched_orig_cleaned_dataset[idx_fold],
                                                      self.mat_idx_slice[idx_fold])
        else:
            vec_epoched_raw_dataset_tvt = DefaultDataset._split_epoched_dataset(self.epoched_raw_dataset,
                                                                                self.mat_idx_slice[idx_fold])

            vec_epoched_cleaned_dataset_tvt = \
                DefaultDataset._split_epoched_dataset(self.vec_epoched_cleaned_dataset[idx_fold],
                                                      self.mat_idx_slice[idx_fold])

        if self.eval_dataset is not None:
            vec_epoched_eval_dataset_tvt = DefaultDataset._split_epoched_dataset(self.epoched_eval_dataset,
                                                                                 self.mat_idx_slice[idx_fold])

            vec_rms_set = compute_rms(self.idx_run, vec_epoched_raw_dataset_tvt, vec_epoched_cleaned_dataset_tvt,
                                      vec_epoched_eval_dataset=vec_epoched_eval_dataset_tvt,
                                      str_eval=self.str_eval, mode=mode, cfg=self.cfg)
        else:
            vec_rms_set = compute_rms(self.idx_run, vec_epoched_raw_dataset_tvt, vec_epoched_cleaned_dataset_tvt,
                                      mode=mode, cfg=self.cfg)

        if idx_fold not in self.rms_results:
            fold_rms_results = {mode: vec_rms_set}
            self.rms_results[idx_fold] = fold_rms_results
        else:
            self.rms_results[idx_fold][mode] = vec_rms_set

    def save_dataset(self, p_output, f_output, overwrite=False, idx_fold=None, **kwargs):
        """
        Save the processed dataset in Neuromag .fif format

        :param pathlib.Path p_output: directory to save the dataset into
        :param str f_output: filename of the saved dataset
        :param boolean overwrite: whether or not to overwrite any existing files
        :param int idx_fold: (optional) index of the current fold (0 indexing)
        :param kwargs: other arguments that are accepted by mne.io.Raw.save() function
        """

        # make output directory if not exist already
        if idx_fold is not None:
            p_output_curr = p_output / 'fold{}'.format(idx_fold)
        else:
            p_output_curr = p_output
        p_output_curr.mkdir(parents=True, exist_ok=True)

        if idx_fold is not None:
            if self.resampled:
                self.vec_orig_cleaned_dataset[idx_fold].save(fname=str(p_output_curr / f_output),
                                                             overwrite=overwrite, **kwargs)
            else:
                self.vec_cleaned_dataset[idx_fold].save(fname=str(p_output_curr / f_output),
                                                        overwrite=overwrite, **kwargs)
        else:
            if self.resampled:
                self.orig_cleaned_dataset.save(fname=str(p_output_curr / f_output), overwrite=overwrite, **kwargs)
            else:
                self.cleaned_dataset.save(fname=str(p_output_curr / f_output), overwrite=overwrite, **kwargs)

    def save_data(self, p_output, f_output, overwrite=False, idx_fold=None, **kwargs):
        """
        Save the processed data in MATLAB .mat format

        :param pathlib.Path p_output: directory to save the dataset into
        :param str f_output: filename of the saved dataset
        :param boolean overwrite: whether or not to overwrite any existing files
        :param int idx_fold: (optional) index of the current fold (0 indexing)
        :param kwargs: other keyword arguments accepted by the sio.savemat() function
        """

        # make output directory if not exist already
        if idx_fold is not None:
            p_output_curr = p_output / 'fold{}'.format(idx_fold)
        else:
            p_output_curr = p_output
        p_output_curr.mkdir(exist_ok=True, parents=True)

        # Obtain the data
        if idx_fold is not None:
            if self.resampled:
                cleaned_data = self.vec_orig_cleaned_dataset[idx_fold].get_data()
            else:
                cleaned_data = self.vec_cleaned_dataset[idx_fold].get_data()

        else:
            if self.resampled:
                cleaned_data = self.orig_cleaned_dataset.get_data()
            else:
                cleaned_data = self.cleaned_dataset.get_data()

        # Save the dataset
        if overwrite or (p_output_curr / f_output).exists() is False:
            sio.savemat(str(p_output_curr / f_output), {'data': cleaned_data}, **kwargs)


if __name__ == '__main__':
    """ used for debugging """
