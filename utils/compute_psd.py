import scipy.signal as signal
import numpy as np
from tabulate import tabulate


# TODO: merge into a single function to avoid potential confusion
def compute_channel_psd(dataset):
    """
    Compute the power spectral density of given EEG data averaged for all channels and across all epochs

    :param dataset: input mne.EpochArray objects holding the epoched data

    :return: f_avg_eeg: frequency at which the psd was computed
    :return: Pxx_avg_eeg: power spectral density at frequencies in f_avg_eeg
    """

    # obtain the data first. Note that here a transpose is needed to convert the data to the form (channel, epoch, data)
    # and also need to convert to micro Volts
    data = np.transpose(dataset.get_data(), axes=(1, 0, 2)) * 1e6

    # Obtain the info object and the indices for ECG and all EEG channels
    info = dataset.info
    ecg_ch = info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # Obtain the EEG data
    eeg_data = data[target_ch, :, :]

    # Obtain the sampling rate and define list for holding the variables
    fs = info['sfreq']
    f_avg_eeg = []
    Pxx_avg_eeg = []

    # Loop through the channels first
    for i in range(eeg_data.shape[0]):

        # Declare empty list to append the PSD and corresponding frequency calculated from each epoch
        f_avg_eeg_ch = []
        Pxx_avg_eeg_ch = []

        # Then loop through the epochs
        for j in range(eeg_data.shape[1]):
            # Compute the power spectral density of the EEG data
            f_eeg_i, Pxx_eeg_i = signal.welch(eeg_data[i, j, :], fs, nperseg=int(data.shape[-1]))

            # Append to the list
            f_avg_eeg_ch.append(f_eeg_i)
            Pxx_avg_eeg_ch.append(Pxx_eeg_i)

        # np.stack(..., axis=0) transforms PSD into the form (epoch, PSD)
        # then taking the mean down axis 0  gives the mean across the epochs
        f_avg_eeg_ch = np.mean(np.stack(f_avg_eeg_ch, axis=0), axis=0)
        Pxx_avg_eeg_ch = np.mean(np.stack(Pxx_avg_eeg_ch, axis=0), axis=0)

        # Append to the bigger list
        f_avg_eeg.append(f_avg_eeg_ch)
        Pxx_avg_eeg.append(Pxx_avg_eeg_ch)

    # np.stack(..., axis=0) transforms PSD into the form (channel, PSD)
    f_avg_eeg = np.stack(f_avg_eeg, axis=0)
    Pxx_avg_eeg = np.stack(Pxx_avg_eeg, axis=0)

    return f_avg_eeg, Pxx_avg_eeg


def compute_mean_psd(dataset):
    """
    Compute the power spectral density of given EEG data averaged across all channels and across all epochs

    :param dataset: input mne.EpochArray objects holding the epoched data

    :return: f_avg_eeg: frequency at which the psd was computed
    :return: Pxx_avg_eeg: power spectral density at frequencies in f_avg_eeg
    """

    # obtain the data first. Note that here a transpose is needed to convert the data to the form (channel, epoch, data)
    # and also need to convert to micro Volts
    data = np.transpose(dataset.get_data(), axes=(1, 0, 2)) * 1e6

    # Obtain the info object and the indices for ECG and all EEG channels
    info = dataset.info
    ecg_ch = info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # Obtain the EEG data
    eeg_data = data[target_ch, :, :]

    # Obtain the sampling rate and define list for holding the variables
    fs = info['sfreq']
    f_avg_eeg = []
    Pxx_avg_eeg = []

    # Loop through the channels first
    for i in range(eeg_data.shape[0]):

        # Declare empty list to append the PSD and corresponding frequency calculated from each epoch
        f_avg_eeg_ch = []
        Pxx_avg_eeg_ch = []

        # Then loop through the epochs
        for j in range(eeg_data.shape[1]):
            # Compute the power spectral density of the EEG data
            f_eeg_i, Pxx_eeg_i = signal.welch(eeg_data[i, j, :], fs, nperseg=int(data.shape[-1]))

            # Append to the list
            f_avg_eeg_ch.append(f_eeg_i)
            Pxx_avg_eeg_ch.append(Pxx_eeg_i)

        # np.stack(..., axis=0) transforms PSD into the form (epoch, PSD)
        # then taking the mean down axis 0  gives the mean across the epochs
        f_avg_eeg_ch = np.mean(np.stack(f_avg_eeg_ch, axis=0), axis=0)
        Pxx_avg_eeg_ch = np.mean(np.stack(Pxx_avg_eeg_ch, axis=0), axis=0)

        # Append to the bigger list
        f_avg_eeg.append(f_avg_eeg_ch)
        Pxx_avg_eeg.append(Pxx_avg_eeg_ch)

    # np.stack(..., axis=0) transforms PSD into the form (channel, PSD)
    # then taking the mean down axis 0  gives the mean across the channels
    f_avg_eeg = np.mean(np.stack(f_avg_eeg, axis=0), axis=0)
    Pxx_avg_eeg = np.mean(np.stack(Pxx_avg_eeg, axis=0), axis=0)

    return f_avg_eeg, Pxx_avg_eeg


def compute_band_power(f_eeg, Pxx_eeg, cutoff_low, cutoff_high):
    """
    Compute the total band power in a frequency band defined by [cutoff_low, cutoff_high]

    :param f_eeg: frequencies at which the power spectral density was computed
    :param Pxx_eeg: the power spectral density at frequencies in f_eeg
    :param cutoff_low: the lower cutoff frequency of the frequency band
    :param cutoff_high: the higher cutoff frequency of the frequency band

    :return: band_power: total band power in a frequency band defined by [cutoff_low, cutoff_high]
    """

    band_idx = (f_eeg >= cutoff_low) & (f_eeg <= cutoff_high)
    band_power = np.sum(Pxx_eeg[band_idx])

    return band_power


def tabulate_band_power_reduction(epoched_raw_dataset_set, epoched_cleaned_dataset_set):
    """
    Compute the power in each frequency band of interest and compute the power ratio

    :param epoched_raw_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the raw dataset, note that the
        data is in the form of (epoch, channel, data)
    :param epoched_cleaned_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the BCGNet-cleaned dataset
    """

    # obtain the info object and indices for the ECG and all EEG channels
    info = epoched_raw_dataset_set.info
    ecg_ch = info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # Compute the mean PSD across all channels
    f_avg_raw_set, Pxx_avg_raw_set = compute_mean_psd(epoched_raw_dataset_set)
    f_avg_cleaned_set, Pxx_avg_cleaned_set = compute_mean_psd(epoched_cleaned_dataset_set)

    # Compute the power in each frequency band
    cutoff_low_delta = 0.5
    cutoff_high_delta = 4

    cutoff_low_theta = 4
    cutoff_high_theta = 8

    cutoff_low_alpha = 8
    cutoff_high_alpha = 13

    # Compute the power in delta band
    delta_raw = compute_band_power(f_avg_raw_set, Pxx_avg_raw_set, cutoff_low_delta, cutoff_high_delta)
    delta_cleaned = compute_band_power(f_avg_cleaned_set, Pxx_avg_cleaned_set, cutoff_low_delta, cutoff_high_delta)

    delta_ratio_raw_cleaned = delta_cleaned / delta_raw

    delta_table = [['BCGNet', delta_cleaned, delta_ratio_raw_cleaned]]

    print('Results for Delta band')
    print(tabulate(delta_table, headers=['Type', 'Total Power', 'Ratio to BCE']))

    # Compute the power in theta band
    theta_raw = compute_band_power(f_avg_raw_set, Pxx_avg_raw_set, cutoff_low_theta, cutoff_high_theta)
    theta_cleaned = compute_band_power(f_avg_cleaned_set, Pxx_avg_cleaned_set, cutoff_low_theta, cutoff_high_theta)

    theta_ratio_raw_cleaned = theta_cleaned / theta_raw

    theta_table = [['BCGNet', theta_cleaned, theta_ratio_raw_cleaned]]

    print('\n\nResults for Theta band')
    print(tabulate(theta_table, headers=['Type', 'Total Power', 'Ratio to BCE']))

    # Compute the power in alpha band
    alpha_raw = compute_band_power(f_avg_raw_set, Pxx_avg_raw_set, cutoff_low_alpha, cutoff_high_alpha)
    alpha_cleaned = compute_band_power(f_avg_cleaned_set, Pxx_avg_cleaned_set, cutoff_low_alpha, cutoff_high_alpha)

    alpha_ratio_raw_cleaned = alpha_cleaned / alpha_raw

    alpha_table = [['BCGNet', alpha_cleaned, alpha_ratio_raw_cleaned]]

    print('\n\nResults for Alpha band')
    print(tabulate(alpha_table, headers=['Type', 'Total Power', 'Ratio to BCE']))
