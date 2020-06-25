import mne
import numpy as np
from scipy.stats import median_absolute_deviation
from settings import get_str_proc

"""
Each dataset object contains a single run of data from a single subject loaded by mne
"""


class DefaultDataset:
    # TODO: check whether cfg is correctly used here and if it's correctly documented in the docstring
    # TODO: obtain the fields from the cfg object
    def __init__(self, d_input, len_epoch=3, mad_threshold=5, new_fs=None, cfg=None):
        """
        Load in the dataset and resample if needed

        :param pathlib.Path d_input: pathlib object containing absolute path to the object
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

    def evaluate_dataset(self):
        pass


if __name__ == '__main__':
    """ used for debugging """
    from pathlib import Path
    import settings

    settings.init(Path.home(), Path.home())  # Call only once
    d_ga_removed = Path('/home/jyao/Local/working_eegbcg/proc_full/proc_rs/sub11/sub11_r01_rs.set')
    dataset = DefaultDataset(d_ga_removed, new_fs=500)
    dataset.prepare_dataset()

    print('nothing')