import numpy as np
import mne
from sp.sp_normalization import renormalize
from sp.sp_preprocessing import dataset_epoch
from utils.interpolate_dataset import *


def predict_time_series(model, callbacks_, normalized_raw_dataset, raw_dataset, orig_sr_raw_dataset, ecg_stats,
                        eeg_stats, duration, good_idx):
    """
    Generates the renormalized ECG and predicted BCG time series

    :param model: Keras model that was trained
    :param callbacks_: early stopping objects that were used in training
    :param normalized_raw_dataset: mne.io_ops.RawArray object that holds the normalized data
    :param raw_dataset: mne.io_ops.RawArray object that holds the unnormalized raw data
    :param orig_sr_raw_dataset: mne.io_ops.RawArray object that holds the original sampling rate
        unnormalized raw data
    :param ecg_stats: ecg_stats: list in the form of [mean_ECG, std_ECG]
    :param eeg_stats: input list in the form of [[eeg_ch1_mean, eeg_ch2_mean, ...], [eeg_ch1_std, eeg_ch2_std, ...]]
        for undoing the initial normalization of the data
    :param duration: duration of each epoch
    :param good_idx: list containing the epochs that passed the epoch rejection from the preprocessing function

    :return: orig_sr_epoched_cleaned_dataset: mne.EpochArray object that holds the interpolated and epoched
        cleaned (unnormalized) data
    :return: orig_sr_cleaned_dataset: mne.io.RawArray object that holds the interpolated cleaned (unnormalized) data
    :return: epoched_cleaned_dataset: mne.EpochArray object that holds the epoched cleaned (unnormalized) data
    :return: cleaned_dataset: mne.io.RawArray object that holds the cleaned (unnormalized) data
    """

    # Obtain the normalized raw data and the info object holding the channel information
    normalized_raw_data = normalized_raw_dataset.get_data()
    info = normalized_raw_dataset.info

    # Obtain the index of the ECG channel
    ecg_ch = info['ch_names'].index('ECG')

    # Obtain the indices of all the EEG channels
    target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # Obtain the normalized ECG and EEG data
    # get the ECG data and perform reshape so that the shape works with Keras
    normalized_ECG_data = normalized_raw_data[ecg_ch, :].reshape(1, normalized_raw_data.shape[1], 1)

    # get the EEG data, no need to do any transformation here, note that data in the form (channel, data)
    normalized_EEG_data = normalized_raw_data[target_ch, :]

    # Predict the BCG data in all EEG channels, note that since Keras generates data in the form of (1, data, channel),
    # a transpose is needed at the end to convert to channel-major format

    # TODO: note that in TF2.0, callbacks can also be passed back here
    predicted_BCG_data = model.predict(x=normalized_ECG_data, verbose=0)
    predicted_BCG_data = np.transpose(predicted_BCG_data.reshape(predicted_BCG_data.shape[1], predicted_BCG_data.shape[2]))

    # Obtain the cleaned EEG data
    normalized_cleaned_EEG_data = normalized_EEG_data - predicted_BCG_data

    # Undo the normalization
    ECG_data = renormalize(normalized_ECG_data, ecg_stats, flag_multi_ch=False, flag_time_series=True)
    clenaed_EEG_data = renormalize(normalized_cleaned_EEG_data, eeg_stats, flag_multi_ch=True, flag_time_series=True)

    # Check if the normalization is performed normally
    ECG_data_orig = raw_dataset.get_data()[ecg_ch, :].reshape(1, raw_dataset.get_data().shape[1], 1)
    if not np.allclose(ECG_data, ECG_data_orig):
        raise Exception('Normalization failed during prediction')

    # reshape to make dimension correct
    ECG_data = ECG_data.reshape(-1)

    # If performed normally, then generate an mne.io_ops.RawArray object holding the cleaned data
    cleaned_data = np.insert(clenaed_EEG_data, ecg_ch, ECG_data, axis=0)
    cleaned_dataset = mne.io.RawArray(cleaned_data, info)

    # Obtain the data from the ground truth dataset that corresponds to epochs that passed the MAD rejection and that
    # are used in training the model
    epoched_cleaned_dataset = dataset_epoch(dataset=cleaned_dataset, duration=duration,
                                            epoch_rejection=False, good_idx=good_idx)

    # Interpolate the dataset and perform the same operation
    orig_sr_cleaned_dataset = interpolate_raw_dataset(cleaned_dataset, orig_sr_raw_dataset)
    orig_sr_epoched_cleaned_dataset = dataset_epoch(dataset=orig_sr_cleaned_dataset, duration=duration,
                                                    epoch_rejection=False, good_idx=good_idx)

    return orig_sr_epoched_cleaned_dataset, orig_sr_cleaned_dataset, epoched_cleaned_dataset, cleaned_dataset


def predict_time_series_mr(model, callbacks_, vec_normalized_raw_dataset, vec_raw_dataset, vec_orig_sr_raw_dataset,
                           vec_ecg_stats, vec_eeg_stats, duration, vec_good_idx):
    """
    Wrapper function using predict_time_series to generate prediction for all runs from the same subject

    :param model: Keras model that was trained
    :param callbacks_: early stopping objects that were used in training
    :param vec_normalized_raw_dataset: list containing mne.RawArray objects where each object contains whitened data
        from a single run
    :param vec_raw_dataset: list containing mne.RawArray objects where each object contains raw data from a single run
    :param vec_orig_sr_raw_dataset: list containing mne.RawArray objects with raw data where each run contains raw
        data where downsampling is not performed
    :param vec_ecg_stats: list of lists where each sublist stores the mean and std for the ECG channel from a single
        run, in the form of [ecg_stats1, ecg_stats2, ...], where each sublist is in the form of [mean, std]
    :param vec_eeg_stats: list of lists where each sublist contains the mean and std for the EEG channels, in the form
        of [eeg_stats1, eeg_stats2,...], where each sublist is in the form [mean_all_channels, std_channels]
    :param duration: duration of each epoch
    :param vec_good_idx: list of list where each sublist contain the epochs that passed the epoch rejection for a
        single run, used later in prediction step

    :return: vec_orig_sr_epoched_cleaned_dataset: list holding mne.EpochArray objects that holds the interpolated and
        epoched cleaned (unnormalized) data from a single run
    :return: vec_orig_sr_cleaned_dataset: list holding mne.io.RawArray objects that holds the interpolated cleaned
        (unnormalized) data from a single run
    :return: vec_epoched_cleaned_dataset: list holding mne.EpochArray objects that holds the epoched cleaned
        (unnormalized) data from a single run
    :return: vec_cleaned_dataset: list holding mne.io.RawArray object that holds the cleaned (unnormalized) data from
        a single run
    """

    vec_orig_sr_epoched_cleaned_dataset = []
    vec_orig_sr_cleaned_dataset = []
    vec_epoched_cleaned_dataset = []
    vec_cleaned_dataset = []

    for i in range(len(vec_normalized_raw_dataset)):
        normalized_raw_dataset = vec_normalized_raw_dataset[i]
        raw_dataset = vec_raw_dataset[i]
        orig_sr_raw_dataset = vec_orig_sr_raw_dataset[i]
        ecg_stats = vec_ecg_stats[i]
        eeg_stats = vec_eeg_stats[i]
        good_idx = vec_good_idx[i]

        orig_sr_epoched_cleaned_dataset, orig_sr_cleaned_dataset, \
            epoched_cleaned_dataset, cleaned_dataset = predict_time_series(model, callbacks_,
                                                                           normalized_raw_dataset, raw_dataset,
                                                                           orig_sr_raw_dataset, ecg_stats,
                                                                           eeg_stats, duration, good_idx)

        vec_orig_sr_epoched_cleaned_dataset.append(orig_sr_epoched_cleaned_dataset)
        vec_orig_sr_cleaned_dataset.append(orig_sr_cleaned_dataset)
        vec_epoched_cleaned_dataset.append(epoched_cleaned_dataset)
        vec_cleaned_dataset.append(cleaned_dataset)

    return vec_orig_sr_epoched_cleaned_dataset, vec_orig_sr_cleaned_dataset, vec_epoched_cleaned_dataset, \
        vec_cleaned_dataset

def predict_validation_epochs(model, callbacks_, x_validation, y_validation, eeg_stats):
    """
    Generate the renormalized corrupted validation set and renormalized cleaned validation set of EEG data for
    RNN models

    :param model: CNN Keras model that was trained
    :param callbacks_: early stopping objects that were used in training
    :param x_validation: ECG data from the validation set in the form (epoch, data, 1)
    :param y_validation: EEG data from the validation set (epoch, data, channel)
    :param eeg_stats: input list either in the form of [[eeg_ch1_mean, eeg_ch2_mean, ...], [eeg_ch1_std, eeg_ch2_std, ...]]
        for undoing the initial normalization of the data

    :return: y_validation_renorm: renormalized corrupted EEG array with unit of micro Volt, and in the form of
        (channel, epoch, data)
    :return: y_validation_cleaned_renorm: renormalized cleaned EEG array with unit of micro Volt, and in the form of
        (channel, epoch, data)
    """

    # Predict the BCG in all EEG channels using ECG as input, note that it currently has the form (epoch, data, channel)
    y_validation_pred = model.predict(x=x_validation, callbacks=callbacks_, verbose=0)

    # Transform the data to the form of (channel, epoch, data) by doing the transpose
    y_validation_cleaned = np.transpose(y_validation - y_validation_pred, axes=(2, 0, 1))
    y_validation_reshape = np.transpose(y_validation, axes=(2, 0, 1))

    # Perform renormalization and subsequently multiply by 1e6 to convert unit to micro Volts
    y_validation_cleaned_renorm = renormalize(y_validation_cleaned, eeg_stats, flag_multi_ch=True,
                                              flag_time_series=False)
    y_validation_cleaned_renorm = y_validation_cleaned_renorm * 1e6

    # Perform renormalization and subsequently multiply by 1e6 to convert unit to micro Volts
    y_validation_renorm = renormalize(y_validation_reshape, eeg_stats, flag_multi_ch=True, flag_time_series=False)
    y_validation_renorm = y_validation_renorm * 1e6

    return y_validation_renorm, y_validation_cleaned_renorm


def predict_test_epochs(model, callbacks_, x_test, y_test, eeg_stats):
    """
    Generate the renormalized corrupted validation set and renormalized cleaned validation set of EEG data for
    RNN models

    :param model: CNN Keras model that was trained
    :param callbacks_: early stopping objects that were used in training
    :param x_test: ECG data from the test set in the form (epoch, data, 1)
    :param y_test: EEG data from the test set (epoch, data, channel)
    :param eeg_stats: input list either in the form of [[eeg_ch1_mean, eeg_ch2_mean, ...], [eeg_ch1_std, eeg_ch2_std, ...]]
        for undoing the initial normalization of the data

    :return: y_validation_renorm: renormalized corrupted EEG array with unit of micro Volt, and in the form of
        (channel, epoch, data)
    :return: y_test_cleaned_renorm: renormalized cleaned EEG array with unit of micro Volt, and in the form of
        (channel, epoch, data)
    """

    # Predict the BCG in all EEG channels using ECG as input, note that it currently has the form (epoch, data, channel)
    y_test_pred = model.predict(x=x_test, callbacks=callbacks_, verbose=0)

    # Transform the data to the form of (channel, epoch, data) by doing the transpose
    y_test_cleaned = np.transpose(y_test - y_test_pred, axes=(2, 0, 1))
    y_test_reshape = np.transpose(y_test, axes=(2, 0, 1))

    # Perform renormalization and subsequently multiply by 1e6 to convert unit to micro Volts
    y_test_cleaned_renorm = renormalize(y_test_cleaned, eeg_stats, flag_multi_ch=True, flag_time_series=False)
    y_test_cleaned_renorm = y_test_cleaned_renorm * 1e6

    # Perform renormalization and subsequently multiply by 1e6 to convert unit to micro Volts
    y_test_renorm = renormalize(y_test_reshape, eeg_stats, flag_multi_ch=True, flag_time_series=False)
    y_test_renorm = y_test_renorm * 1e6

    return y_test_renorm, y_test_cleaned_renorm