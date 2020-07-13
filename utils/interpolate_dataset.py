import mne
import scipy.interpolate as interpolate


def interpolate_raw_dataset(dataset, orig_sr_raw_dataset):
    """
    Interpolate the downsampled dataset to the original sampling rate

    :param dataset: mne.io_ops.RawArray object holding the downsampled dataset
    :param orig_sr_raw_dataset: mne.io_ops.RawArray object holding the raw dataset with the original sampling rate

    :return: interpolated_dataset: mne.io_ops.RawArray object holding the interpolated dataset
    """

    # obtain the time stamps, data and the info object from the dataset
    ts = dataset.times
    data = dataset.get_data()

    # obtain the original time stamps
    orig_ts = orig_sr_raw_dataset.times

    # perform interpolation
    interpolator = interpolate.PchipInterpolator(ts, data, axis=1, extrapolate=True)
    interpolated_data = interpolator(orig_ts, extrapolate=True)
    info = orig_sr_raw_dataset.info

    # substitute the ECG data with original ECG data
    interpolated_dataset = mne.io.RawArray(interpolated_data, info, verbose=False)

    return interpolated_dataset


def interpolate_epoched_dataset(epoched_dataset, orig_sr_epoched_raw_dataset):
    """
    Interpolate the downsampled epoched dataset to the original sampling rate

    :param epoched_dataset: mne.io_ops.EpochArray object holding the downsampled dataset
    :param orig_sr_epoched_raw_dataset: mne.io_ops.EpochArray object holding the raw dataset with the original sampling
        rate

    :return: interpolated_epoched_dataset: mne.io_ops.EpochArray object holding the interpolated dataset
    """
    # obtain the time stamps, data and the info object from the dataset
    ts = epoched_dataset.times
    data = epoched_dataset.get_data()

    # obtain the original time stamps
    orig_ts = orig_sr_epoched_raw_dataset.times

    # perform interpolation
    interpolator = interpolate.PchipInterpolator(ts, data, axis=2, extrapolate=True)
    interpolated_epoched_data = interpolator(orig_ts, extrapolate=True)
    info = orig_sr_epoched_raw_dataset.info

    # substitute the ECG data with original ECG data
    interpolated_epoched_dataset = mne.EpochsArray(interpolated_epoched_data, info)

    return interpolated_epoched_dataset
