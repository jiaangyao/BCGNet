import numpy as np
import matplotlib.pyplot as plt
from utils.compute_psd import *


def plot_training_history(m, p_figure):
    """
    Plotting the training loss and validation loss for CNN and RNN training

    :param m: keras history object
    :param p_figure: pathlib.Path object holding the root directory to save the figure to
    """
    plt.figure(figsize=(6, 5), num=11)
    epochs = len(m.epoch)
    n_epoch = np.arange(1, epochs + 1)
    plt.plot(n_epoch, m.history['loss'], label='loss')
    plt.plot(n_epoch, m.history['val_loss'], label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')

    fig = plt.gcf()
    fig.savefig(p_figure / 'training_history.png', format='png')
    plt.close(fig)


def plot_random_epoch(epoched_raw_dataset_set, epoched_obs_dataset_set, epoched_cleaned_dataset_set, vec_ix_slice_set,
                      p_figure, opt):
    """
    Plotting the prediction of an architecture on a random epoch.

    NOTE: for the RNN input, no padding was applied

    :param epoched_raw_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the raw dataset, note that the
        data is in the form of (epoch, channel, data)
    :param epoched_obs_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the OBS-cleaned dataset
    :param epoched_cleaned_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the BCGNet-cleaned dataset
    :param vec_ix_slice_set: indices of the epochs in the original dataset
    :param p_figure: pathlib.Path object that holds the path to save the figures to
    :param opt: option object that contains all the parameters for training
    """

    # Obtain the info and the indices for ECG and all EEG channels
    info = epoched_raw_dataset_set.info
    ecg_ch = info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # Obtain the ECG data from all the epochs, note that the data is the form of (epoch, data)
    ecg_data_set = epoched_raw_dataset_set.get_data()[:, ecg_ch, :] * 1e6

    # Similarly, obtain the EEG data from all three datasets, note that these data are in the form of (epoch, channel,
    # data)
    raw_eeg_data_set = epoched_raw_dataset_set.get_data()[:, target_ch, :] * 1e6
    obs_eeg_data_set = epoched_obs_dataset_set.get_data()[:, target_ch, :] * 1e6
    cleaned_eeg_data_set = epoched_cleaned_dataset_set.get_data()[:, target_ch, :] * 1e6

    # Obtain the time corresponding to each sample
    t = np.arange(0, ecg_data_set.shape[1], 1) / info['sfreq']

    # Select a number of channels
    vec_channel_selection = np.random.permutation(raw_eeg_data_set.shape[1])
    for ix in np.arange(0, opt.training_figure_num):

        # Select a random channel and epoch
        ch = vec_channel_selection[ix]
        ep = np.random.randint(0, raw_eeg_data_set.shape[0])

        # Obtain the name of the corresponding channel
        # Since ECG channel is taken out from the y_validation sets, need to add 1 if the index of the channel picked is
        # larger than the index of the ECG channel
        if ch < info['ch_names'].index('ECG'):
            ch_name = info['ch_names'][ch]
            ch_num = ch + 1
        else:
            ch_name = info['ch_names'][ch + 1]
            ch_num = ch + 2

        # Add one to let index of epoch start from 1 in the titles
        ep_num = vec_ix_slice_set[ep] + 1

        # Generating the plot
        plt.figure(figsize=(8, 10))
        plt.suptitle('Prediction in Channel {}, Epoch {}'.format(ch_name, ep_num),
                     fontweight='bold')
        plt.subplot(311)
        plt.title('Original ECG waveform')
        plt.plot(t, ecg_data_set[ep, :], 'C0')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude ($\mu$V)')
        plt.xlim([0, 3.1])

        plt.subplot(312)
        plt.title('Predicted BCG waveform')
        plt.plot(t, raw_eeg_data_set[ep, ch, :] - cleaned_eeg_data_set[ep, ch, :], 'C4')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude ($\mu$V)')
        plt.xlim([0, 3.1])

        plt.subplot(313)
        plt.title('BCG, OBS and BCGNet')
        plt.plot(t, raw_eeg_data_set[ep, ch, :], 'C1', label='BCE')
        plt.plot(t, obs_eeg_data_set[ep, ch, :], 'C2', label='OBS')
        plt.plot(t, cleaned_eeg_data_set[ep, ch, :], 'C3', label='BCGNet')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude ($\mu$V)')
        plt.xlim([0, 3.1])

        plt.legend(loc='upper right', frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        fig = plt.gcf()
        p_figure_curr = p_figure / 'test_set_epoch'
        p_figure_curr.mkdir(parents=True, exist_ok=True)

        fig.savefig(p_figure_curr / 'test_set_epoch_ch{}_ep{}.png'.format(ch_num, ep_num), format='png')
        plt.close(fig)

    # NOW: Always generate plots from a fixed epoch for debugging purposes
    # Select a random channel and epoch
    ch = 0
    ep = 3

    ch_name = info['ch_names'][ch]
    ch_num = ch

    # Add one to let index of epoch start from 1 in the titles
    ep_num = vec_ix_slice_set[ep] + 1

    # Generating the plot
    plt.figure(figsize=(6, 8), num=11)
    plt.suptitle('Prediction in Channel {}, Epoch {}'.format(ch_name, ep_num),
                 fontweight='bold')
    plt.subplot(311)
    plt.title('Original ECG waveform')
    plt.plot(t, ecg_data_set[ep, :], 'C0')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ($\mu$V)')
    plt.xlim([0, 3.1])

    plt.subplot(312)
    plt.title('Predicted BCG waveform')
    plt.plot(t, raw_eeg_data_set[ep, ch, :] - cleaned_eeg_data_set[ep, ch, :], 'C4')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ($\mu$V)')
    plt.xlim([0, 3.1])

    plt.subplot(313)
    plt.title('BCG, OBS and BCGNet')
    plt.plot(t, raw_eeg_data_set[ep, ch, :], 'C1', label='BCE')
    plt.plot(t, obs_eeg_data_set[ep, ch, :], 'C2', label='OBS')
    plt.plot(t, cleaned_eeg_data_set[ep, ch, :], 'C3', label='BCGNet')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ($\mu$V)')
    plt.xlim([0, 3.1])

    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    fig = plt.gcf()
    p_figure_curr = p_figure / 'test_set_epoch' / 'debug'
    p_figure_curr.mkdir(parents=True, exist_ok=True)

    fig.savefig(p_figure_curr / 'test_set_epoch_ch{}_ep{}.png'.format(ch_name, ep_num), format='png')
    plt.close(fig)


def plot_psd(epoched_raw_dataset_set, epoched_obs_dataset_set, epoched_cleaned_dataset_set, p_figure, opt):
    """
    Plot the PSD for chosen number of channels and the summary PSD by default

    :param epoched_raw_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the raw dataset, note that the
        data is in the form of (epoch, channel, data)
    :param epoched_obs_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the OBS-cleaned dataset
    :param epoched_cleaned_dataset_set: mne.io_ops.EpochArray object holding the epoched data from the BCGNet-cleaned dataset
    :param p_figure: pathlib.Path object holding the path to the directory for saving the figures
    :param opt: option object for holding all the parameter settings
    """

    # obtain the info object and indices for the ECG and all EEG channels
    info = epoched_raw_dataset_set.info
    ecg_ch = info['ch_names'].index('ECG')
    target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # Compute the mean PSD across all channels
    f_avg_raw_set, Pxx_avg_raw_set = compute_mean_psd(epoched_raw_dataset_set)
    f_avg_obs_set, Pxx_avg_obs_set = compute_mean_psd(epoched_obs_dataset_set)
    f_avg_cleaned_set, Pxx_avg_cleaned_set = compute_mean_psd(epoched_cleaned_dataset_set)

    # Compute the PSD for all channels
    f_raw_set, Pxx_raw_set = compute_channel_psd(epoched_raw_dataset_set)
    f_obs_set, Pxx_obs_set = compute_channel_psd(epoched_obs_dataset_set)
    f_cleaned_set, Pxx_cleaned_set = compute_channel_psd(epoched_cleaned_dataset_set)

    # Plotting the power spectral density
    plt.figure(figsize=(6, 6))
    plt.title('Average Power Spectral Density Across all channels')
    plt.semilogy(f_avg_raw_set, Pxx_avg_raw_set, 'C1-', label='BCE')
    plt.semilogy(f_avg_obs_set, Pxx_avg_obs_set, 'C2--', label='OBS')
    plt.semilogy(f_avg_cleaned_set, Pxx_avg_cleaned_set, 'C3--', label='BCGNet')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'PSD ($\mu V^2/Hz)$')

    plt.xlim([0, 30])
    plt.ylim([1e-4, 1e4])
    plt.legend(loc='upper right')

    fig = plt.gcf()
    p_figure_curr = p_figure / 'test_set_psd' / 'summary'
    p_figure_curr.mkdir(parents=True, exist_ok=True)

    fig.savefig(p_figure_curr / 'test_set_psd_across_ch.png', format='png')
    plt.close(fig)

    # NOW: Always generate plots from a fixed epoch for debugging purposes
    # Select a random channel and epoch
    ch = 0
    ep = 3
    ch_name = info['ch_names'][ch]
    ch_num = ch
    ep_num = ep + 1

    ecg_data_set = epoched_raw_dataset_set.get_data()[:, ecg_ch, :] * 1e6
    raw_eeg_data_set = epoched_raw_dataset_set.get_data()[:, target_ch, :] * 1e6
    obs_eeg_data_set = epoched_obs_dataset_set.get_data()[:, target_ch, :] * 1e6
    cleaned_eeg_data_set = epoched_cleaned_dataset_set.get_data()[:, target_ch, :] * 1e6

    # Obtain the time corresponding to each sample
    t = np.arange(0, ecg_data_set.shape[1], 1) / info['sfreq']

    # Generating the plot
    plt.figure(figsize=(6, 8), num=11)
    plt.suptitle('Prediction in Channel {}, Epoch {}'.format(ch_name, ep_num),
                 fontweight='bold')
    plt.subplot(311)
    plt.title('Original ECG waveform')
    plt.plot(t, ecg_data_set[ep, :], 'C0')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ($\mu$V)')
    plt.xlim([0, 3.1])

    plt.subplot(312)
    plt.title('Predicted BCG waveform')
    plt.plot(t, raw_eeg_data_set[ep, ch, :] - cleaned_eeg_data_set[ep, ch, :], 'C4')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ($\mu$V)')
    plt.xlim([0, 3.1])

    plt.subplot(313)
    plt.title('BCG, OBS and BCGNet')
    plt.plot(t, raw_eeg_data_set[ep, ch, :], 'C1', label='BCE')
    plt.plot(t, obs_eeg_data_set[ep, ch, :], 'C2', label='OBS')
    plt.plot(t, cleaned_eeg_data_set[ep, ch, :], 'C3', label='BCGNet')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude ($\mu$V)')
    plt.xlim([0, 3.1])

    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    fig = plt.gcf()
    p_figure_curr = p_figure / 'test_set_psd' / 'debug'
    p_figure_curr.mkdir(parents=True, exist_ok=True)

    fig.savefig(p_figure_curr / 'test_set_epoch_debug.png', format='png')
    plt.close(fig)

    # Select a number of channels
    vec_channel_selection = np.random.permutation(epoched_raw_dataset_set.get_data()[:, target_ch, :].shape[1])
    for ix in np.arange(0, opt.training_figure_num):
        # Select a random channel and epoch
        ch = vec_channel_selection[ix]

        # Obtain the name of the corresponding channel
        # Since ECG channel is taken out from the y_validation sets, need to add 1 if the index of the channel picked is
        # larger than the index of the ECG channel
        if ch < info['ch_names'].index('ECG'):
            ch_name = info['ch_names'][ch]
            ch_num = ch + 1
        else:
            ch_name = info['ch_names'][ch + 1]
            ch_num = ch + 2

        # Plotting the power spectral density
        plt.figure(figsize=(6, 6))
        plt.title('Power Spectral Density for channel {}'.format(ch_name))
        plt.semilogy(f_raw_set[ch, :], Pxx_raw_set[ch, :], 'C1-', label='BCE')
        plt.semilogy(f_obs_set[ch, :], Pxx_obs_set[ch, :], 'C2--', label='OBS')
        plt.semilogy(f_cleaned_set[ch, :], Pxx_cleaned_set[ch, :], 'C3--', label='BCGNet')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'PSD ($\mu V^2/Hz)$')

        plt.xlim([0, 30])
        plt.ylim([1e-4, 1e4])
        plt.legend(loc='upper right')

        fig = plt.gcf()
        p_figure_curr = p_figure / 'test_set_psd'
        p_figure_curr.mkdir(parents=True, exist_ok=True)

        fig.savefig(p_figure_curr / 'test_set_psd_ch{}.png'.format(ch_name), format='png')
        plt.close(fig)
