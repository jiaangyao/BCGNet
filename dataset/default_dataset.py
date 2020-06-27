import mne
import numpy as np
from scipy.stats import median_absolute_deviation
from settings import get_str_proc
from utils import temp_seed

"""
Each dataset object contains a single run of data from a single subject loaded by mne
"""


class DefaultDataset:
    # TODO: check whether cfg is correctly used here and if it's correctly documented in the docstring
    # TODO: obtain the fields from the cfg object
    def __init__(self, d_input, len_epoch=3, mad_threshold=5, new_fs=None, per_training=0.7, per_valid=0.15,
                 per_test=0.15, random_seed=1997, cv_mode=False, num_fold=None, cfg=None):
        """
        Load in the dataset and resample if needed

        :param pathlib.Path d_input: pathlib object containing absolute path to the single run of data
        :param int new_fs: (optional) new desired sampling rate
        :param object cfg: configuration

        """

        self.cfg = cfg

        # Load in the dataset
        self.raw_dataset = self._load_dataset(d_input)
        self.fs = int(self.raw_dataset.info['sfreq'])
        self.resampled = False

        self.len_epoch = len_epoch
        self.mad_threshold = mad_threshold

        self.standardized_dataset = None
        self.epoched_standardized_dataset = None
        self.epoch_rejected = False
        self.vec_idx_good_epochs = None

        if per_training + per_valid + per_test > 1:
            raise RuntimeError("Total percentage greater than 1")

        self.per_training = per_training
        self.per_valid = per_valid
        self.per_test = per_test
        self.random_seed = random_seed

        self.cv_mode = cv_mode
        if cv_mode:
            self.vec_xs = None
            self.vec_ys = None
            self.mat_idx_slice = None

            if cv_mode is not None:
                self.num_fold = num_fold
            else:
                self.num_fold = int(np.round(1 / self.per_test))

        else:
            self.xs = None
            self.ys = None
            self.vec_idx_slice = None

        # Resample if needed
        if new_fs is not None:
            if new_fs != self.fs:
                print("\nResample from {} Hz to {} Hz".format(self.fs, new_fs))
                self.raw_dataset.resample(new_fs)

                self.orig_fs = self.fs
                self.orig_raw_dataset = self._load_dataset(d_input, verbose=False)
                self.resampled = True

        # TODO: delete the following snippet (since the user doesn't have any motion data)
        # TODO: put the channel names in a meta file somewhere
        if get_str_proc() == 'proc_full':
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

        :return: a tuple (standardized_raw_dataset, ecg_stats, eeg_stats), where standardized_raw_dataset is the object
            that contains data by channels, ecg_stats is a list [ecg_mean, ecg_std] and eeg_stats is a list
            [[eeg_mean1, eeg_mean2,...], [eeg_std1, eeg_std2,...]]
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

    def _epoch_dataset(self):
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

    @staticmethod
    def _perform_epoching(raw_dataset, len_epoch, mad_threshold):
        """
        Perform epoching of a raw dataset and perform rejection of outlier epochs based on mean absolute deviation

        :param mne.io.RawArray raw_dataset: object holding the dataset for which epoching operation is to be performed,
            used for obtaining the length in time of the original recording
        :param len_epoch: length of each epoch in seconds
        :param mad_threshold: # times the mean absolute deviation to set the threshold for MAD-based epoch rejection

        :return: processed epoched dataset
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
        :param int len_epoch: length of each epoch in seconds
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
            self.xs, self.ys, \
                self.vec_idx_slice = DefaultDataset._generate_train_valid_test(self.epoched_standardized_dataset,
                                                                               self.per_training, self.per_valid,
                                                                               self.random_seed)

        else:
            self.vec_xs, self.vec_ys, \
                self.mat_idx_slice = DefaultDataset._generate_train_valid_test_cv(self.epoched_standardized_dataset,
                                                                                  self.per_valid, self.num_fold,
                                                                                  self.random_seed)

    # TODO: fix the documentation
    @staticmethod
    def _generate_train_valid_test(epoched_dataset, per_training, per_valid, random_seed):
        """
        Generate the training, validation and test epoch data from the MNE Epoch object based on the training set
            ratio and validation set ratio and random seed provided

        :param mne.EpochsArray epoched_dataset: object where epoched_dataset.get_data() is the  normalized
            epoched data and is of the form of (epoch, channel, data)
        :param float per_training: percentage of training set epochs
        :param float per_valid: percentage of validation set epochs
        :param int random_seed: random seed used for the splitting of the dataset

        :return:


        :return: xs: list containing all the ECG data in the form of [x_train, x_validation, x_test], where each element
            is of the form (epoch, channel, data)
        :return: ys: list containing all the corrupted EEG data in the form of [y_train, y_validation, y_test], where each
            element is of the form (epoch, channel, data)
        :return: vec_ix_slice: list in the form of [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test],
            where each element contains the indices of epochs in the original dataset belonging to the training, validation
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

    # TODO: fix the documentation
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


        :return: vec_xs: list containing all the ECG data in the form of [fold1, fold2, ...], where each fold is in the form
            of [x_train, x_validation, x_test] and each element is of the form (epoch, data)
        :return: vec_ys: list containing all the corrupted EEG data in the form of [fold1, fold2, ...] where each fold is
            in the form [y_train, y_validation, y_test] and each element is of the form (epoch, channel, data)
        :return: mat_ix_slice: list in the form of [fold1, fold2, ...], where each fold is of the form of
            [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test], where each element contains the indices of
            epochs in the original dataset belonging to the training, validation and test set respectively
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
                vec_idx_training = vec_idx_tv[num_valid:]
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

    def evaluate_dataset(self):
        pass

    # TODO: fix this later
    @staticmethod
    def _split_epoched_dataset(epoched_dataset, vec_ix_slice):
        """
        Split the mne.io_ops.Mne object holding the cleaned EEG data into the same set of training, validation and test
        epochs as during the training

        :param epoched_dataset: mne.io_ops.Mne object that holds the epoched data, note that the data held has the form
            (epoch, channel, data)
        :param vec_ix_slice: list in the form of [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test],
            where each element contains the indices of epochs in the original dataset belonging to the training, validation
            and test set respectively

        :return: vec_epoched_dataset_set: list in the form of [epoched_dataset_training, epoched_dataset_validation,
            epoched_dataset_test], where each is an mne.io_ops.Mne object that holds epoched data belonging to the specified set
            during training
        """

        # Get the data and info object first
        epoched_data = epoched_dataset.get_data()
        info = epoched_dataset.info

        # Get the training, validation and test data
        epoched_data_training = epoched_data[vec_ix_slice[0], :, :]
        epoched_data_validation = epoched_data[vec_ix_slice[1], :, :]
        epoched_data_test = epoched_data[vec_ix_slice[2], :, :]

        epoched_dataset_training = mne.EpochsArray(epoched_data_training, info)
        epoched_dataset_validation = mne.EpochsArray(epoched_data_validation, info)
        epoched_dataset_test = mne.EpochsArray(epoched_data_test, info)

        vec_epoched_dataset_set = [epoched_dataset_training, epoched_dataset_validation, epoched_dataset_test]

        return vec_epoched_dataset_set


if __name__ == '__main__':
    """ used for debugging """
    from pathlib import Path
    import settings

    settings.init(Path.home(), Path.home())  # Call only once
    d_ga_removed = Path('/home/jyao/Local/working_eegbcg/proc_full/proc_rs/sub11/sub11_r01_rs.set')
    dataset = DefaultDataset(d_ga_removed, new_fs=500)
    dataset.prepare_dataset()
    dataset.split_dataset()

    print('nothing')