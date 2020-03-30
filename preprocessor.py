import scipy.stats as stats
from sp_normalization import *
from collections import namedtuple
from utils.context_management import suppress_stdout
from settings import rs_path, obs_path
import numpy as np
import mne
import settings
from options import test_opt

Opt = namedtuple('Opt', ['input_feature', 'output_features',
                         'd_features', 't_epoch', 'generate',
                         'fs_ds', 'p_training', 'p_validation',
                         'p_evaluation'])


def opt_default():
    """
    This is a function in feature_extractor.py with default settings
    not sure if there is a type in the channel settings in MNE, but if
    so that would be the easiest…:

    :return:
    """
    return Opt(
        # if the feature_type opt are None, then you can specify manually,
        # e.g. opt.input_feature = [0, 1, 2] or, opt.input_feature =
        # [‘Fz’, ‘Cz’] etc.
        input_feature=['ecg'],
        output_features=['eeg'],
        d_features=settings.d_root / 'Local/working_eegbcg/proc_bcgnet/features/',
        t_epoch=2,
        # generate=generate_ws_features,  # train and test within subject.
        # To test across subject, or test within run we define new functions
        # here some extensions might fit neatly within generate_ws_features,
        # for some we might need entirely new functions specified here.
        fs_ds=100,  # frequency at which to downsample (this gets inverted
        # at the end of the pipeline)
        p_training=0,
        p_validation=0.15,
        p_evaluation=0.85
    )


def preprocessing(dataset_dir, duration, threshold, n_downsampling, flag_use_motion_data):
    """
    Performs all the preprocessing for the data in the RNN processing pipeline

    NOTE: Here MNE package is used to load the original dataset in EEGLAB format and is used as a structure that holds
    all processed data and the info (which contains channel information) to avoid hard coding the channel index
    corresponding to the ECG channel or hard coding the sampling frequency of the data

    NOTE: here there is no need to pad the data since for RNN the output has the same shape as the input

    :param dataset_dir: pathlib.Path object pointing to the source, which is a EEGlab format containing a single run
        from a single subject of shape (64, n_time_stamps)
    :param duration: duration of each epoch
    :param threshold: multiples of mean absolute deviation (MAD) within each epoch for thresholding outliers
    :param n_downsampling: factor of downsampling, if 1 then no downsampling is performed
    :param flag_use_motion_data: flag for whether or not motion data was used

    :return: normalized_epoched_raw_dataset: mne.EpochArray object containing the whitened epoched data
    :return: normalized_raw_dataset: mne Raw object with whitened data
    :return: epoched_raw_dataset: mne.EpochArray object containing the raw epoched data
    :return: raw_dataset: mne.RawArray object with raw data
    :return: orig_sr_epoched_raw_dataset: mne.EpochArray object with data epoched using raw data with original
        sampling rate
    :return: orig_sr_raw_dataset: mne.RawArray object with raw data where downsampling is not performed
    :return: ecg_stats: list containing the mean and std for the ECG channel, shape (1, 2)
    :return: eeg_stats: list containing the mean and std for the EEG channels, shape (63, 2)
    :return: good_idx: list containing the epochs that passed the epoch rejection, used later in prediction step
    """

    print('\n\nStarting to load the data\n')
    # Loading the raw input in EEGLAB format and downsampling it
    raw_dataset = mne.io.read_raw_eeglab(dataset_dir, preload=True, stim_channel=False)

    if n_downsampling != 1:
        fs_orig = raw_dataset.info['sfreq']
        fs = fs_orig / n_downsampling
        raw_dataset.resample(fs)

    # Load the raw dataset again but don't downsample it this time
    orig_sr_raw_dataset = mne.io.read_raw_eeglab(dataset_dir, preload=True, stim_channel=False)

    # Get rid of the motion data channels if flag not set to true
    if not flag_use_motion_data:
        raw_dataset.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])
        orig_sr_raw_dataset.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])

    print('\n\nPerforming normalization\n')
    # Performs normalization by whitening each channel
    normalized_raw_dataset, ecg_stats, eeg_stats = normalize_raw_data(raw_dataset)

    print('\n\nPartitioning the data into 3s epochs\n')
    # Perform epoch rejection by threshold * MAD
    normalized_epoched_raw_dataset, good_idx = dataset_epoch(dataset=normalized_raw_dataset, duration=duration,
                                                             epoch_rejection=True, threshold=threshold,
                                                             raw_dataset=raw_dataset)

    # Epoch the unnormalized raw for later uses also
    epoched_raw_dataset = dataset_epoch(dataset=raw_dataset, duration=duration, epoch_rejection=False,
                                        good_idx=good_idx)

    # Epoch the unnormalized raw with original sampling rate for later uses also
    orig_sr_epoched_raw_dataset = dataset_epoch(dataset=orig_sr_raw_dataset,
                                                duration=duration, epoch_rejection=False,
                                                good_idx=good_idx)

    return normalized_epoched_raw_dataset, normalized_raw_dataset, epoched_raw_dataset, raw_dataset, \
           orig_sr_epoched_raw_dataset, orig_sr_raw_dataset, ecg_stats, eeg_stats, good_idx


def preprocessing_mr(str_sub, vec_run_id, duration, threshold, n_downsampling, flag_use_motion_data):
    """
    Wrapper function using the preprocessing function for single runs to load multiple runs from the same subject

    :param str_sub: string for the subject
    :param vec_run_id: list containing the indices for al the runs to be analyzed
    :param duration: duration of each epoch
    :param threshold: multiples of mean absolute deviation (MAD) within each epoch for thresholding outliers
    :param n_downsampling: factor of downsampling, if 1 then no downsampling is performed
    :param flag_use_motion_data: flag for whether or not motion data was used

    :return: vec_normalized_epoched_raw_dataset: list containing mne.EpochArray objects where each object contains
        the whitened epoched data from a single run
    :return: vec_normalized_raw_dataset: list containing mne.RawArray objects where each object contains whitened data
        from a single run

    :return: vec_epoched_raw_dataset: list containing mne.EpochArray objects where each object contains the raw
        epoched data from a single run
    :return: vec_raw_dataset: list containing mne.RawArray objects where each object contains raw data from a single run

    :return: vec_orig_sr_epoched_raw_dataset: list containing mne.EpochArray objects where each object contains the
        epoched data with the original sampling rate from a single run
    :return: vec_orig_sr_raw_dataset: list containing mne.RawArray objects with raw data where each run contains raw
        data where downsampling is not performed

    :return: vec_ecg_stats: list of lists where each sublist stores the mean and std for the ECG channel from a single
        run, in the form of [ecg_stats1, ecg_stats2, ...], where each sublist is in the form of [mean, std]
    :return: vec_eeg_stats: list of lists where each sublist contains the mean and std for the EEG channels, in the form
        of [eeg_stats1, eeg_stats2,...], where each sublist is in the form [mean_all_channels, std_channels]
    :return: vec_good_idx: list of list where each sublist contain the epochs that passed the epoch rejection for a
        single run, used later in prediction step
    """

    # Load the data for each run and pack everything into lists
    vec_normalized_epoched_raw_dataset = []
    vec_normalized_raw_dataset = []
    vec_epoched_raw_dataset = []
    vec_raw_dataset = []
    vec_orig_sr_epoched_raw_dataset = []
    vec_orig_sr_raw_dataset = []
    vec_ecg_stats = []
    vec_eeg_stats = []
    vec_good_idx = []

    for run_id in vec_run_id:
        # Path setup
        p_rs, f_rs = rs_path(str_sub, run_id)
        pfe_rs = str(p_rs / f_rs)

        # Load, normalize and epoch the raw dataset
        normalized_epoched_raw_dataset, normalized_raw_dataset, epoched_raw_dataset, raw_dataset, \
        orig_sr_epoched_raw_dataset, orig_sr_raw_dataset, \
        ecg_stats, eeg_stats, good_idx = preprocessing(dataset_dir=pfe_rs, duration=duration,
                                                       threshold=threshold,
                                                       n_downsampling=n_downsampling,
                                                       flag_use_motion_data=flag_use_motion_data)

        vec_normalized_epoched_raw_dataset.append(normalized_epoched_raw_dataset)
        vec_normalized_raw_dataset.append(normalized_raw_dataset)
        vec_epoched_raw_dataset.append(epoched_raw_dataset)
        vec_raw_dataset.append(raw_dataset)
        vec_orig_sr_epoched_raw_dataset.append(orig_sr_epoched_raw_dataset)
        vec_orig_sr_raw_dataset.append(orig_sr_raw_dataset)
        vec_ecg_stats.append(ecg_stats)
        vec_eeg_stats.append(eeg_stats)
        vec_good_idx.append(good_idx)

    return vec_normalized_epoched_raw_dataset, vec_normalized_raw_dataset, vec_epoched_raw_dataset, vec_raw_dataset, \
           vec_orig_sr_epoched_raw_dataset, vec_orig_sr_raw_dataset, vec_ecg_stats, vec_eeg_stats, vec_good_idx


def dataset_epoch(dataset, duration, epoch_rejection, threshold=None, raw_dataset=None, good_idx=None):
    """
    A single wrapper function that performs all epoching related operations

    Option 1: (MAD-based epoch rejection on raw data)
        performs mean absolute deviation (MAD) based epoch rejection on the dataset based on a threshold that's
        specified by the user

    Option 2: (Extraction of equivalent epochs)
        after the MAD-based epoch rejection is performed on the raw dataset, for comparison purposes, often we want
        to extract the same epochs from the ground truth dataset. Then using good_idx that was previously generated
        by the same function, this function can extract equivalent epochs from the clean or ground truth dataset
        so that comparison can be made later

    :param dataset: MNE Raw object that holds normalized data waiting to be epoched
    :param duration: desired length of time windows (epochs) to split the data
    :param epoch_rejection: boolean for whether to perform MAD-based epoch rejection or to extract equivalent epochs
        based on good_idx
    :param threshold: # times the mean absolute deviation to set the threshold for MAD-based epoch rejection
    :param raw_dataset: needed if performing the MAD-based epoch rejection
    :param good_idx: index of epochs that passed MAD-based epoch rejection, needed for extracting equivalent epochs
        from the ground truth dataset

    Option 1:
    :return: epoched_dataset: MNE Epoch object that holds data from all epochs that passed the MAD-based epoch
        rejection test as well as the information structure. epoched_dataset.get_data() has the form
        (epoch, channel, data)
    :return: good_idx: the list containing indices of epochs that passed the MAD-based epoch rejection test

    Option 2:
    :return: epoched_dataset: MNE Epoch object that holds the data from all epochs equivalent to those that passed the
        MAD-based test performed on the raw data, in the form of (epoch, channel, data)

    """

    # Obtain the information from the MNE Raw object and obtain the sampling rate
    info = dataset.info
    fs = info['sfreq']

    # Constructing events of duration seconds
    constructed_events, tmax = epoch_events(dataset, fs, duration)

    # Performing epoching of the input dataset based on the constructed events
    old_epoched_dataset = mne.Epochs(dataset, constructed_events, tmin=0, tmax=tmax, baseline=None)

    if epoch_rejection:
        # If choosing option 1 and want to perform MAD-based epoch rejection

        # Epoch rejection based on median absolute deviation of mean of absolute values for individual epochs

        # Obtain the index of epochs that are rejected, delete from the list of all epochs and then create a new
        # MNE Epoch object holding the data from all good epochs and info structure
        reject_idx = mad_rejection(raw_dataset, threshold, fs, duration)

        print('\nRejecting {} epochs'.format(len(reject_idx)))
        good_idx = np.delete(np.arange(0, old_epoched_dataset.get_data().shape[0], 1), reject_idx)
        good_data = old_epoched_dataset.get_data()[good_idx, :, :]
        epoched_dataset = mne.EpochsArray(good_data, old_epoched_dataset.info)

        # return both the MNE Epoch object and the list of epochs that passed the test
        return epoched_dataset, good_idx

    else:
        # If choosing option 2 and want to extract epochs that passed MAD-based epoch rejection performed on the raw
        # data

        # Then simply extract the epochs that are good using good_idx
        epoched_data = old_epoched_dataset.get_data()[good_idx, :, :]
        epoched_dataset = mne.EpochsArray(epoched_data, old_epoched_dataset.info)

        return epoched_dataset


def epoch_events(dataset, fs, duration):
    """
    Create events of fixed duration apart to split the original time series into time windows (epochs) of equal length

    :param dataset: MNE Raw object holding the dataset for which epoching operation is to be performed, used for
        obtaining the length in time of the original recording
    :param fs: sampling rate of the dataset
    :param duration: desired duration of each time window (epoch)

    :return: constructed_events: numpy list containing the sample index corresponding to the start of each time window
    :return: tmax: length of each time window in samples
    """

    # Obtain the length in samples of the original recording
    total_time_stamps = dataset.get_data().shape[1]

    # Create the empty numpy array to hold the events, of shape (floor(time/duration), 3)
    constructed_events = np.zeros(shape=(int(np.floor(total_time_stamps / fs) / duration), 3), dtype=int)

    # Populate the constructed_events created with the starting index of each time window
    # The numbers 0 and 1 are for marking the event as fake for MNE
    for i in range(0, int(np.floor(total_time_stamps / fs)) - duration, duration):
        ix = i / duration
        constructed_events[int(ix)] = np.array([i * fs, 0, 1])

    n_events = len(range(0, int(np.floor(total_time_stamps / fs)) - duration, duration))
    if n_events < constructed_events.shape[0]:
        constructed_events = constructed_events[:n_events, :]

    # Delete the last sample to make the length of the time window even
    tmax = duration - 1 / fs

    return constructed_events, tmax


def mad_rejection(dataset, threshold, fs, duration):
    """
    Function for performing the mean absolute deviation (MAD) based epoch rejection

    :param dataset: MNE Raw object holding the input dataset for which MAD-based epoch rejection is to be
        performed
    :param threshold: # times the mean absolute deviation to set the threshold for MAD-based epoch rejection
    :param fs: desired sampling rate of the dataset
    :param duration: duration of each time window

    :return: vec_bad_epochs_ix: indices for epochs that failed the MAD-based epoch rejection
    """

    # Suppress the standard output to avoid populating the terminal/output window with too much info
    with suppress_stdout():
        # Obtain the sampling rate of dataset and if different from desired sampling rate then resample to that s.r.
        srate = dataset.info['sfreq']
        if srate != fs:
            dataset.resample(fs)

        # Obtain the index of the ECG channel
        info = dataset.info
        ecg_ch = info['ch_names'].index('ECG')

        # Obtain the indices of all the EEG channels
        target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

        # Obtain the constructed events
        constructed_events, tmax = epoch_events(dataset, fs, duration)

        # Perform epoching using the MNE package functionality and splits the dataset into time windows of equal length
        epoched_dataset = mne.Epochs(dataset, constructed_events, tmin=0, tmax=tmax, baseline=None)

        # Obtain the absolute value of the data, of form (epoch, channel, data)
        data_abs = np.absolute(epoched_dataset.get_data())

    # Compute the ratio of each epoch's absolute value across all channels over its MAD
    vec_mabs_eeg = np.mean(data_abs[:, target_ch, :], axis=(1, 2))
    vec_eeg_norm = (vec_mabs_eeg - np.median(vec_mabs_eeg)) / stats.median_absolute_deviation(vec_mabs_eeg)

    # If the ratio is higher than the threshold then the epoch is rejected
    vec_bad_epochs_ix = np.arange(0, len(vec_eeg_norm), 1)[vec_eeg_norm > threshold]

    return vec_bad_epochs_ix


def normalize_raw_data(raw_dataset):
    """
    Performs renormalization of the raw dataset by whitening each channel (subtracting the mean of each channel and
    then divide by the standard deviation (std) of each channel

    :param raw_dataset: MNE Raw Object that contains the unnormalized raw data

    :return: normalized_raw
    :return: ecg_stats
    :return: eeg_stats
    """

    # Obtain the data numpy array and information structure from the MNE Raw object
    data = raw_dataset.get_data()
    info = raw_dataset.info

    # Obtain the number of the channel that holds the ECG channel and the channel index
    ecg_ch = info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # used for reverting back to original data later
    ecg_mean = np.mean(data[ecg_ch, :])
    ecg_std = np.std(data[ecg_ch, :])
    eeg_mean = np.mean(data[target_ch, :], axis=1)
    eeg_std = np.std(data[target_ch, :], axis=1)

    normalized_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        ds = data[i, :] - np.mean(data[i, :])
        ds /= np.std(ds)
        normalized_data[i, :] = ds

    normalized_raw = mne.io.RawArray(normalized_data, info)

    ecg_stats = [ecg_mean, ecg_std]
    eeg_stats = [eeg_mean, eeg_std]

    return normalized_raw, ecg_stats, eeg_stats


def renormalize(data, stats, flag_multi_ch, flag_time_series):
    """
    Performs renormalization operations that undo the whitening of the data in the proprocessing steps
    In particular, data is renormalized by via the formula (channel of data) * (channel std) + (channel mean)

    :param data: input numpy array that can be (either in the form of (epoch, data) in the case of ECG input or in the
        form of (channel, epoch, data) in the case of EEG input) for epoched data, or (can be in the form of (data,) in
        the case of ECG input or in the form of (channel, data) in the case of EEG input) in the case of time series
        input
    :param stats: input list either in the form of [ecg_mean, ecg_std] or in the form of
        [[eeg_ch1_mean, eeg_ch2_mean, ...], [eeg_ch1_std, eeg_ch2_std, ...]]
    :param flag_multi_ch: input boolean variable that specifies if input is ECG or EEG (0 for ECG, 1 for EEG)
    :param flag_time_series: input boolean variable that specifies if input is in the form of epochs (epoch, [channel],
        data) or in the form of time series ([channel], data) (0 for epoched data, 1 for time series)

    :return: data_renorm: renormalized data either in the form (epoch, data) or (epoch, channel, data)
    """

    # If the data is in the form of time series
    if flag_time_series:
        if not flag_multi_ch:
            # If the input is ECG, then renormalization can be simply done by multiplying data by std (stats[0])
            # and then add by mean (stats[1])
            data_renorm = data * stats[1] + stats[0]

        else:
            # If the input is EEG, then first create an empty array of the same shape as the input and then perform
            # normalization channel by channel

            # Create empty array same size as input
            data_renorm = np.zeros(data.shape)

            # Loop through the channels of the input
            for i in range(data.shape[0]):
                # For each channel, perform the renormalization
                data_renorm[i, :] = data[i, :] * stats[1][i] + stats[0][i]

    # If the data is in the form of epoched data
    else:
        if not flag_multi_ch:
            # If the input is ECG, then renormalization can be simply done by multiplying data by std (stats[0])
            # and then add by mean (stats[1])
            data_renorm = data * stats[1] + stats[0]

        else:
            # If the input is EEG, then first create an empty array of the same shape as the input and then perform
            # normalization channel by channel

            # Create empty array same size as input
            data_renorm = np.zeros(data.shape)

            # Loop through the channels of the input
            for i in range(data.shape[0]):
                # For each channel, perform the renormalization
                data_renorm[i, :, :] = data[i, :, :] * stats[1][i] + stats[0][i]

    return data_renorm


if __name__ == '__main__':
    """ used for debugging """
    from pathlib import Path
    settings.init(Path.home(), Path.home())  # Call only once
    str_sub = 'sub11'
    run_id = 1
    opt_user = test_opt(None)

    # Path setup
    p_rs, f_rs = rs_path(str_sub, run_id)
    p_obs, f_obs = obs_path(str_sub, run_id)

    pfe_rs = str(p_rs / f_rs)
    pfe_obs = str(p_obs / f_obs)

    """
    Preparing the dataset
    """
    # Load, normalize and epoch the raw dataset
    normalized_epoched_raw_dataset, normalized_raw_dataset, epoched_raw_dataset,\
    raw_dataset, orig_sr_epoched_raw_dataset, orig_sr_raw_dataset, \
    ecg_stats, eeg_stats, good_idx = preprocessing(dataset_dir=pfe_rs,
                                                   duration=opt_user.epoch_duration,
                                                   threshold=opt_user.mad_threshold,
                                                   n_downsampling=opt_user.n_downsampling,
                                                   flag_use_motion_data=opt_user.use_motion_data)
