import mne
import glob
import os
import sys
from contextlib import contextmanager
import numpy as np
import scipy.stats as stats


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def single_subject_mabs(dataset, threshold):
    with suppress_stdout():
        srate = dataset.info['sfreq']
        if srate != 100:
            fs = srate/5
            dataset.resample(fs)
        else:
            fs = srate

        if len(dataset.info['ch_names']) > 64:
            dataset.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])
        info = dataset.info
        ecg_ch = info['ch_names'].index('ECG')
        target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

        duration = 3
        total_time_stamps = dataset.get_data().shape[1]
        constructed_events =np.zeros(shape=(int(np.floor(total_time_stamps/fs)/duration), 3), dtype=int)

        for i in range(0, int(np.floor(total_time_stamps/fs))-duration, duration):
            ix = i/duration
            constructed_events[int(ix)] = np.array([i*fs, 0, 1])

        tmax = duration - 1/fs
        epoched_dataset = mne.Epochs(dataset, constructed_events, tmin=0, tmax=tmax)
        data_abs = np.absolute(epoched_dataset.get_data())

    vec_mabs_ecg = np.mean(data_abs[:, ecg_ch, :], axis=1)
    vec_mabs_eeg = np.mean(data_abs[:, target_ch, :], axis=(1, 2))

    vec_ecg_norm = (vec_mabs_ecg - np.median(vec_mabs_ecg)) / stats.median_absolute_deviation(vec_mabs_ecg)
    vec_eeg_norm = (vec_mabs_eeg - np.median(vec_mabs_eeg)) / stats.median_absolute_deviation(vec_mabs_eeg)

    vec_bad_epochs_ix = np.arange(0, len(vec_eeg_norm), 1)[vec_eeg_norm > threshold]

    return vec_bad_epochs_ix


if __name__ == '__main__':
    import settings

    p_rs, f_rs = settings.rs_path('sub11', 1)
    p_rs = p_rs.parents[0]
    vec_filename = glob.glob(str(p_rs / '**/sub*_r0[1-5]_rs.set'), recursive=True)

    vec_data = []
    vec_mabs_ecg = []
    vec_mabs_eeg = []

    with suppress_stdout():
        for i in vec_filename:
            rs_raw = mne.io.read_raw_eeglab(i, preload=True, stim_channel=False)
            srate = rs_raw.info['sfreq']
            fs = srate / 5
            rs_raw.resample(fs)
            rs_raw.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])
            ecg_ch = rs_raw.info['ch_names'].index('ECG')
            target_ch = np.delete(np.arange(0, len(rs_raw.info['ch_names']), 1), ecg_ch)

            duration = 3
            total_time_stamps = rs_raw.get_data().shape[1]
            constructed_events = np.zeros(shape=(int(np.floor(total_time_stamps / fs) / duration), 3), dtype=int)

            for i in range(0, int(np.floor(total_time_stamps / fs)) - duration, duration):
                ix = i / duration
                constructed_events[int(ix)] = np.array([i * fs, 0, 1])

            tmax = duration - 1 / fs
            # Epoching the data using the constructed event and plotting it
            epoched_data = mne.Epochs(rs_raw, constructed_events, tmin=0, tmax=tmax)
            data = epoched_data.get_data()
            vec_data.append(data)

            data_abs = np.absolute(data)
            ecg_abs = data_abs[:, ecg_ch, :]
            m_ecg_abs = np.mean(ecg_abs, axis=1)

            eeg_abs = data_abs[:, target_ch, :]
            m_eeg_abs = np.mean(eeg_abs, axis=(1, 2))

            vec_mabs_ecg.append(m_ecg_abs)
            vec_mabs_eeg.append(m_eeg_abs)

    vec_mabs_ecg = np.array(vec_mabs_ecg)
    vec_mabs_eeg = np.array(vec_mabs_eeg)

    vec_mabs_ecg = vec_mabs_ecg.reshape(-1)
    vec_mabs_eeg = vec_mabs_eeg.reshape(-1)

    vec_ecg_norm = (vec_mabs_ecg - np.median(vec_mabs_ecg)) / stats.median_absolute_deviation(vec_mabs_ecg)
    vec_eeg_norm = (vec_mabs_eeg - np.median(vec_mabs_eeg)) / stats.median_absolute_deviation(vec_mabs_eeg)

    vec_norm = np.array([vec_ecg_norm.reshape(-1), vec_eeg_norm.reshape(-1)])

    # vec_norm[:, np.logical_or(vec_norm[0, :] > 5, vec_norm[1, :] > 5)].shape[-1]/(len(vec_eeg_norm)) * 100

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.hist(vec_mabs_ecg, bins=20)
    plt.title('Historgram of ECG epoch mean absolute')

    plt.subplot(122)
    plt.hist(vec_mabs_eeg, bins=20)
    plt.title('Historgram of EEG epoch mean absolute')

    plt.show()
