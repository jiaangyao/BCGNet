import mne
import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal

from tabulate import tabulate


def interpolate_raw_dataset(dataset, orig_raw_dataset):
    """
    Interpolate the downsampled dataset to the original sampling rate

    :param mne.io.RawArray dataset: object holding the downsampled dataset
    :param mne.io.RawArray orig_raw_dataset: object holding the raw dataset with the original sampling rate

    :return: mne.io.RawArray object holding the interpolated dataset
    """

    # obtain the time stamps, data and the info object from the dataset
    ts = dataset.times
    data = dataset.get_data()

    # obtain the original time stamps
    orig_ts = orig_raw_dataset.times

    # perform interpolation
    interpolator = interpolate.PchipInterpolator(ts, data, axis=1, extrapolate=True)
    interpolated_data = interpolator(orig_ts, extrapolate=True)
    info = orig_raw_dataset.info

    # substitute the ECG data with original ECG data
    interpolated_dataset = mne.io.RawArray(interpolated_data, info, verbose=False)

    return interpolated_dataset


def compute_psd(dataset):
    """
    Compute the power spectral density of given EEG data averaged for all channels and across all epochs

    :param mne.EpochsArray dataset: input objects holding the epoched data

    :return: a tuple (f_avg_eeg, pxx_avg_eeg), where f_avg_eeg is the channel average of frequencies
        at which the PSD was computed for all channels and pxx_avg_eeg is channel average power spectral of
        all channels
    """

    # obtain the data first. Note that here a transpose is needed to convert the data to the form
    # (channel, epoch, data) and also need to convert to micro Volts
    data = np.transpose(dataset.get_data(), axes=(1, 0, 2)) * 1e6
    info = dataset.info
    fs = info['sfreq']

    # Obtain the indices for ECG and all EEG channels
    ch_ecg = info['ch_names'].index('ECG')
    ch_eeg = np.delete(np.arange(0, len(info['ch_names']), 1), ch_ecg)

    # Obtain the EEG data
    eeg_data = data[ch_eeg, :, :]

    # have the list for holding all the variables
    f_eeg = []
    pxx_eeg = []

    # Loop through the channels first
    for i in range(eeg_data.shape[0]):

        # Declare empty list to append the PSD and corresponding frequency calculated from each epoch
        f_avg_eeg_ch = []
        pxx_avg_eeg_ch = []

        # Then loop through the epochs
        for j in range(eeg_data.shape[1]):
            # Compute the power spectral density of the EEG data
            f_eeg_i, pxx_eeg_i = signal.welch(eeg_data[i, j, :], fs, nperseg=int(data.shape[-1]))

            # Append to the list
            f_avg_eeg_ch.append(f_eeg_i)
            pxx_avg_eeg_ch.append(pxx_eeg_i)

        # np.stack(..., axis=0) transforms PSD into the form (epoch, PSD)
        # then taking the mean down axis 0  gives the mean across the epochs
        f_avg_eeg_ch = np.mean(np.stack(f_avg_eeg_ch, axis=0), axis=0)
        pxx_avg_eeg_ch = np.mean(np.stack(pxx_avg_eeg_ch, axis=0), axis=0)

        # Append to the bigger list
        f_eeg.append(f_avg_eeg_ch)
        pxx_eeg.append(pxx_avg_eeg_ch)

    # np.stack(..., axis=0) transforms PSD into the form (channel, PSD)
    f_eeg = np.stack(f_eeg, axis=0)
    pxx_eeg = np.stack(pxx_eeg, axis=0)

    # compute the mean across the channels also
    f_avg_eeg = np.mean(f_eeg, axis=0)
    pxx_avg_eeg = np.mean(pxx_eeg, axis=0)

    return f_avg_eeg, pxx_avg_eeg


def compute_band_power(f_eeg, pxx_eeg, cutoff_low, cutoff_high):
    """
    Compute the total band power in a frequency band defined by [cutoff_low, cutoff_high]

    :param np.ndarray f_eeg: frequencies at which the power spectral density was computed
    :param np.ndarray pxx_eeg: the power spectral density at frequencies in f_eeg
    :param int/float cutoff_low: the lower cutoff frequency of the frequency band
    :param int/float cutoff_high: the higher cutoff frequency of the frequency band

    :return: total band power in a frequency band defined by [cutoff_low, cutoff_high]
    """

    band_idx = (f_eeg >= cutoff_low) & (f_eeg <= cutoff_high)
    band_power = np.sum(pxx_eeg[band_idx])

    return band_power


def tabulate_band(vec_f_avg_set, vec_pxx_avg_set, vec_str_dataset, cutoff_low, cutoff_high, str_band):
    """

    :param list vec_f_avg_set: list containing the frequency from multiple dataset, note first has to be raw
    :param list vec_pxx_avg_set: list containing PSD from multiple dataset, note first has to be raw
    :param list vec_str_dataset: list containing the dataset type from multiple dataset, first has to be raw
    :param int/float cutoff_low: the lower cutoff frequency of the frequency band
    :param int/float cutoff_high: the higher cutoff frequency of the frequency band
    :param str str_band: name of the current frequency band
    """

    if not vec_str_dataset[0].lower() == 'raw':
        raise RuntimeError('Erroneous packaging of datasets')

    # obtain band power from raw dataset first
    f_avg_raw_set = vec_f_avg_set[0]
    pxx_avg_raw_set = vec_pxx_avg_set[0]
    band_raw = compute_band_power(f_avg_raw_set, pxx_avg_raw_set, cutoff_low, cutoff_high)
    band_table = []

    # Compute the power all the other datasets
    for i in range(1, len(vec_f_avg_set)):
        band_dataset_i = compute_band_power(vec_f_avg_set[i], vec_pxx_avg_set[i], cutoff_low, cutoff_high)
        band_ratio_dataset_i = band_dataset_i / band_raw
        band_table.append([vec_str_dataset[i], band_dataset_i, band_ratio_dataset_i])

    print('Results for {} band'.format(str_band))
    print(tabulate(band_table, headers=['Type', 'Total Power', 'Ratio to Raw']) + '\n')


def tabulate_band_power_reduction(epoched_raw_dataset_set, epoched_cleaned_dataset_set, epoched_eval_dataset_set=None,
                                  str_eval=None, cfg=None):
    """
    Compute the power in each frequency band of interest and compute the power ratio

    :param mne.EpochsArray epoched_raw_dataset_set: object holding the epoched data from the raw dataset,
        note that the data is in the form of (epoch, channel, data)
    :param mne.EpochsArray epoched_cleaned_dataset_set: object holding the epoched data from the BCGNet-cleaned
        dataset
    :param mne.EpochsArray epoched_eval_dataset_set: (optional) dataset used for comparing performance provided by
        the user
    :param str str_eval: (optional) name of the method for the evaluation dataset
    :param cfg: configuration file containing the band definitions
    """

    # Compute the power in each frequency band
    cutoff_low_delta = cfg.cutoff_low_delta
    cutoff_high_delta = cfg.cutoff_high_delta

    cutoff_low_theta = cfg.cutoff_low_theta
    cutoff_high_theta = cfg.cutoff_high_theta

    cutoff_low_alpha = cfg.cutoff_low_alpha
    cutoff_high_alpha = cfg.cutoff_high_alpha

    # Compute the mean PSD across all channels
    f_avg_raw_set, pxx_avg_raw_set = compute_psd(epoched_raw_dataset_set)
    f_avg_cleaned_set, pxx_avg_cleaned_set = compute_psd(epoched_cleaned_dataset_set)

    vec_f_avg_set = [f_avg_raw_set, f_avg_cleaned_set]
    vec_pxx_avg_set = [pxx_avg_raw_set, pxx_avg_cleaned_set]
    vec_str_dataset = ['raw', 'BCGNet']

    if epoched_eval_dataset_set is not None:
        f_avg_eval_set, pxx_avg_eval_set = compute_psd(epoched_eval_dataset_set)

        vec_f_avg_set = [f_avg_raw_set, f_avg_eval_set, f_avg_cleaned_set]
        vec_pxx_avg_set = [pxx_avg_raw_set, pxx_avg_eval_set, pxx_avg_cleaned_set]
        vec_str_dataset = ['raw', str_eval, 'BCGNet']

    # Compute the power in delta band
    tabulate_band(vec_f_avg_set, vec_pxx_avg_set, vec_str_dataset, cutoff_low_delta, cutoff_high_delta, 'Delta')

    # Compute the power in theta band
    tabulate_band(vec_f_avg_set, vec_pxx_avg_set, vec_str_dataset, cutoff_low_theta, cutoff_high_theta, 'Theta')

    # Compute the power in alpha band
    tabulate_band(vec_f_avg_set, vec_pxx_avg_set, vec_str_dataset, cutoff_low_alpha, cutoff_high_alpha, 'Alpha')


def compute_rms_epoched_dataset(epoched_raw_dataset_set, epoched_cleaned_dataset_set,
                                epoched_eval_dataset_set=None):
    """
    Compute the RMS value based on epoched dataset (i.e. from a specific group such as test set during training)

    :param mne.io.RawArray epoched_raw_dataset_set: object holding the epoched data from the raw dataset, note
        that the data is in the form of (epoch, channel, data)
    :param mne.io.RawArray epoched_cleaned_dataset_set: object holding the epoched data from the BCGNet-cleaned
        dataset
    :param mne.io.RawArray epoched_eval_dataset_set: (optional) dataset used for comparing performance provided by
        the user

    :return: a tuple rms_raw, rms_eval_eeg_data_set, rms_cleaned of RMS values from raw evaluation and cleaned datasets
        respectively if evaluation set is provided and a tuple rms_raw, None, rms_cleaned if evaluation set
        is not provided
    """

    # Obtain the set data for all three datasets and change unit to micro V
    epoched_raw_data_set = epoched_raw_dataset_set.get_data() * 1e6
    epoched_cleaned_data_set = epoched_cleaned_dataset_set.get_data() * 1e6

    # obtain the index of the EEG channel
    ch_ecg = epoched_raw_dataset_set.info['ch_names'].index('ECG')
    ch_eeg = np.delete(np.arange(0, len(epoched_raw_dataset_set.info['ch_names']), 1), ch_ecg)

    # extract the EEG data
    epoched_raw_eeg_data_set = epoched_raw_data_set[:, ch_eeg, :]
    epoched_cleaned_eeg_data_set = epoched_cleaned_data_set[:, ch_eeg, :]

    # Compute the RMS for all data
    rms_raw_eeg_data_set = np.sqrt(np.square(epoched_raw_eeg_data_set).mean())
    rms_cleaned_eeg_data_set = np.sqrt(np.square(epoched_cleaned_eeg_data_set).mean())

    if epoched_eval_dataset_set is not None:
        epoched_eval_data_set = epoched_eval_dataset_set.get_data() * 1e6
        epoched_eval_eeg_data_set = epoched_eval_data_set[:, ch_eeg, :]
        rms_eval_eeg_data_set = np.sqrt(np.square(epoched_eval_eeg_data_set).mean())

        # return the RMS value
        vec_rms_set = rms_raw_eeg_data_set, rms_eval_eeg_data_set, rms_cleaned_eeg_data_set

        return vec_rms_set

    # return the RMS value
    vec_rms_set = rms_raw_eeg_data_set, None, rms_cleaned_eeg_data_set

    return vec_rms_set


def compute_rms(idx_run, vec_epoched_raw_dataset, vec_epoched_cleaned_dataset,  vec_epoched_eval_dataset=None,
                str_eval=None, mode='test', cfg=None):
    """
    Split epoched dataset and compute the RMS value.

    :param int idx_run: the current run
    :param list vec_epoched_raw_dataset: a list in the form of [training, valid, test] from the raw dataset
    :param list vec_epoched_cleaned_dataset: a list in the form of [training, valid, test] from the
        BCGNet-cleaned dataset
    :param list vec_epoched_eval_dataset: (optional) a list in the form of [training, valid, test] from the
        dataset used for comparing performance provided by the user
    :param str str_eval: (optional) name of the method for the evaluation dataset
    :param str mode: either 'train', 'valid' or 'test', indicating which set to extract RMS value and
        power ratio from

    :return: a tuple of RMS values from raw, evaluation and cleaned dataset if eval is given and from raw, and
        cleaned dataset if not
    """

    if mode == 'test':
        idx_set = 2
    elif mode == 'valid':
        idx_set = 1
    elif mode == 'train':
        idx_set = 0
    else:
        raise NotImplementedError

    epoched_raw_dataset_set = vec_epoched_raw_dataset[idx_set]
    epoched_cleaned_dataset_set = vec_epoched_cleaned_dataset[idx_set]

    # Computing the RMS using the test set
    if vec_epoched_eval_dataset is not None:
        epoched_eval_dataset_set = vec_epoched_eval_dataset[idx_set]
        vec_rms_set = compute_rms_epoched_dataset(epoched_raw_dataset_set=epoched_raw_dataset_set,
                                                  epoched_eval_dataset_set=epoched_eval_dataset_set,
                                                  epoched_cleaned_dataset_set=epoched_cleaned_dataset_set)

        print("RMS VALUES: RUN {}, {} SET".format(idx_run, mode.upper()))
        print("RMS Raw: {}".format(vec_rms_set[0]))
        print("RMS {}: {}".format(str_eval, vec_rms_set[1]))
        print("RMS BCGNet: {}".format(vec_rms_set[2]))

        # Compute the reduction in each power band
        print("\nFREQUENCY BAND POWER REDUCTION: RUN {}".format(idx_run))
        tabulate_band_power_reduction(epoched_raw_dataset_set=epoched_raw_dataset_set,
                                      epoched_eval_dataset_set=epoched_eval_dataset_set,
                                      epoched_cleaned_dataset_set=epoched_cleaned_dataset_set,
                                      str_eval=str_eval, cfg=cfg)

    else:
        vec_rms_set = compute_rms_epoched_dataset(epoched_raw_dataset_set, epoched_cleaned_dataset_set)

        print("RMS VALUES: RUN {}, {} SET".format(idx_run, mode.upper()))
        print("RMS Raw: {}".format(vec_rms_set[0]))
        print("RMS BCGNet: {}".format(vec_rms_set[2]))

        # Compute the reduction in each power band
        print("\nFREQUENCY BAND POWER REDUCTION: RUN {}".format(idx_run))
        tabulate_band_power_reduction(epoched_raw_dataset_set, epoched_cleaned_dataset_set, cfg=cfg)
    print('\n'.rjust(41, '_'))

    return vec_rms_set



