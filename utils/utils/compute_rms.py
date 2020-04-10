import numpy as np
import mne


def compute_rms_epoched_dataset(epoched_raw_dataset_set, epoched_obs_dataset_set, epoched_cleaned_dataset_set):
    """
    Compute the RMS value based on epoched dataset (i.e. from a specific group such as test set during training)

    :param epoched_raw_dataset_set: mne.io_ops.RawArray object holding the epoched data from the raw dataset, note that the
        data is in the form of (epoch, channel, data)
    :param epoched_obs_dataset_set: mne.io_ops.RawArray object holding the epoched data from the OBS-cleaned dataset
    :param epoched_cleaned_dataset_set: mne.io_ops.RawArray object holding the epoched data from the BCGNet-cleaned dataset

    :return:
    """

    # Obtain the set data for all three datasets and change unit to micro V
    epoched_raw_data_set = epoched_raw_dataset_set.get_data() * 1e6
    epoched_obs_data_set = epoched_obs_dataset_set.get_data() * 1e6
    epoched_cleaned_data_set = epoched_cleaned_dataset_set.get_data() * 1e6

    # obtain the index of the EEG channel
    ecg_ch = epoched_raw_dataset_set.info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(epoched_raw_dataset_set.info['ch_names']), 1), ecg_ch)

    # extract the EEG data
    epoched_raw_eeg_data_set = epoched_raw_data_set[:, target_ch, :]
    epoched_obs_eeg_data_set = epoched_obs_data_set[:, target_ch, :]
    epoched_cleaned_eeg_data_set = epoched_cleaned_data_set[:, target_ch, :]

    # Compute the RMS for all data
    rms_raw_eeg_data_set = np.sqrt(np.square(epoched_raw_eeg_data_set).mean())
    rms_obs_eeg_data_set = np.sqrt(np.square(epoched_obs_eeg_data_set).mean())
    rms_cleaned_eeg_data_set = np.sqrt(np.square(epoched_cleaned_eeg_data_set).mean())

    # return the RMS value
    vec_rms_set = [rms_raw_eeg_data_set, rms_obs_eeg_data_set, rms_cleaned_eeg_data_set]

    return vec_rms_set
