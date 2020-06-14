import numpy as np

from dataset_splitter import split_epoched_dataset_mr
from utils.compute_psd import tabulate_band_power_reduction


def compute_rms(vec_run_id, vec_orig_sr_epoched_raw_dataset, vec_orig_sr_epoched_cleaned_dataset, mr_ten_ix_slice):
    """
    Split epoched dataset and compute the RMS value.

    :param vec_run_id: TODO
    :param vec_orig_sr_epoched_raw_dataset: TODO
    :param vec_orig_sr_epoched_cleaned_dataset: TODO
    :param mr_ten_ix_slice: TODO
    :return: mat_rms_test: TODO
    """
    # Obtain the equivalent test epochs used during training from raw, OBS-cleaned and BCGNet-cleaned data
    mat_epoched_raw_dataset = split_epoched_dataset_mr(vec_orig_sr_epoched_raw_dataset, mr_ten_ix_slice)
    mat_epoched_cleaned_dataset = split_epoched_dataset_mr(vec_orig_sr_epoched_cleaned_dataset, mr_ten_ix_slice)

    vec_orig_sr_epoched_raw_dataset_test = []
    vec_orig_sr_epoched_cleaned_dataset_test = []
    mat_rms_test = []

    for i in range(len(vec_run_id)):
        vec_epoched_raw_dataset = mat_epoched_raw_dataset[i]
        vec_epoched_cleaned_dataset = mat_epoched_cleaned_dataset[i]

        orig_sr_epoched_raw_dataset_test = vec_epoched_raw_dataset[-1]
        orig_sr_epoched_cleaned_dataset_test = vec_epoched_cleaned_dataset[-1]

        # Computing the RMS using the test set
        vec_rms_test = _compute_rms_epoched_dataset_(orig_sr_epoched_raw_dataset_test,
                                                     orig_sr_epoched_cleaned_dataset_test)

        vec_orig_sr_epoched_raw_dataset_test.append(orig_sr_epoched_raw_dataset_test)
        vec_orig_sr_epoched_cleaned_dataset_test.append(orig_sr_epoched_cleaned_dataset_test)
        mat_rms_test.append(vec_rms_test)

    print("\n#############################################")
    print("#                  Results                  #")
    print("#############################################\n")

    for i in range(len(vec_run_id)):
        vec_rms_test = mat_rms_test[i]
        orig_sr_epoched_raw_dataset_test = vec_orig_sr_epoched_raw_dataset_test[i]
        orig_sr_epoched_cleaned_dataset_test = vec_orig_sr_epoched_cleaned_dataset_test[i]

        print("RMS VALUES: RUN {}".format(vec_run_id[i]))
        print("RMS Raw: {}".format(vec_rms_test[0]))
        print("RMS BCGNet: {}".format(vec_rms_test[1]))

        # Compute the reduction in each power band
        print("\nFREQUENCY BAND POWER REDUCTION: RUN {}".format(vec_run_id[i]))
        tabulate_band_power_reduction(orig_sr_epoched_raw_dataset_test,
                                      orig_sr_epoched_cleaned_dataset_test)
        print('\n\n\n')

    return mat_rms_test


def _compute_rms_epoched_dataset_(epoched_raw_dataset_set, epoched_cleaned_dataset_set):
    """
    Compute the RMS value based on epoched dataset (i.e. from a specific group such as test set during training)

    :param epoched_raw_dataset_set: mne.io_ops.RawArray object holding the epoched data from the raw dataset, note that the
        data is in the form of (epoch, channel, data)
    :param epoched_cleaned_dataset_set: mne.io_ops.RawArray object holding the epoched data from the BCGNet-cleaned dataset

    :return: vec_rms_set: [rms_raw_eeg_data_set, rms_cleaned_eeg_data_set]
    """

    # Obtain the set data for all three datasets and change unit to micro V
    epoched_raw_data_set = epoched_raw_dataset_set.get_data() * 1e6
    epoched_cleaned_data_set = epoched_cleaned_dataset_set.get_data() * 1e6

    # obtain the index of the EEG channel
    ecg_ch = epoched_raw_dataset_set.info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(epoched_raw_dataset_set.info['ch_names']), 1), ecg_ch)

    # extract the EEG data
    epoched_raw_eeg_data_set = epoched_raw_data_set[:, target_ch, :]
    epoched_cleaned_eeg_data_set = epoched_cleaned_data_set[:, target_ch, :]

    # Compute the RMS for all data
    rms_raw_eeg_data_set = np.sqrt(np.square(epoched_raw_eeg_data_set).mean())
    rms_cleaned_eeg_data_set = np.sqrt(np.square(epoched_cleaned_eeg_data_set).mean())

    # return the RMS value
    vec_rms_set = [rms_raw_eeg_data_set, rms_cleaned_eeg_data_set]

    return vec_rms_set
