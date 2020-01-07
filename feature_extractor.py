import mne
from pathlib import Path
from epoch_rejection import single_subject_mabs
from collections import namedtuple, defaultdict
import contextlib
import numpy as np
import settings
import hash_opt as ho
import mne.io


def opt_default():
    # this is a function in feature_extractor.py with default settings
    # not sure if there is a type in the channel settings in MNE, but if
    # so that would be the easiest…:
    Opt = namedtuple('Opt', ['input_feature', 'output_features',
                             'd_features', 't_epoch', 'generate', 'fs_ds'])

    return Opt(
        # if the feature_type opt are None, then you can specify manually,
        # e.g. opt.input_feature = [0, 1, 2] or, opt.input_feature =
        # [‘Fz’, ‘Cz’] etc.
        input_feature=['ecg'],
        output_features=['eeg'],
        d_features = 'settings.d_root/proc_bcgnet/features/',
        t_epoch = 2,
        generate = generate_ws_features,  # train and test within subject.
        # To test across subject, or test within run we define new functions
        # here some extensions might fit neatly within generate_ws_features,
        # for some we might need entirely new functions specified here.
        fs_ds = 100 # frequency at which to downsample (this gets inverted
        # at the end of the pipeline)
    )


def generate(d_mne, opt):
    # point here is to convert a standardised MNE structure to an epoched X, Y
    # for training, test and validate

    # Assert that of  [opt.output_features, opt.output_feature_type], exactly
    # one is none, do same for input
    d_features = opt.generate(d_mne, opt)  # so that we can easily switch it out
    return d_features


def generate_ws_features(d_mne, opt):
    # n.b. This is the code that generates features to train and test within
    # subject… different functions would need to be defined for different
    # required feature sets. I suggest we start with this one, then see how
    # we go….
    #  d_mne = Path(opt.d_mne)
    # d_features = Path(opt.d_features)
    # h = ho.dict_to_hash(ho.namedtuple_to_dict(opt), exclusions=None)
    # d_features = d_features / str(h)  # (I suggest a hash of opt, combined
    # with some useful human readable stuff)

    # raw_fname = '/home/yida/Local/working_eegbcg/test_output/sub12_r02_rs_raw.fif'
    # # Setup for reading the raw data
    # raw = mne.io.read_raw_fif(raw_fname)

    d_mne = Path('/home/yida/Local/working_eegbcg/test_output')
    dict_data = defaultdict(dict)  # a structure that contains organised files. E.g.:
    # dict_files[‘subject01][0] = [0, 1, 2]  # i.e. [subject][session][run]

    # load in all the data - reason for this is that we -might- need to do
    # single run processing based on all runs (data should be small, but we
    # can change this structure if it gets problematic)
    for sub in d_mne.iterdir():
        for run in (d_mne / sub).iterdir():
            # load each run with MNE and store in dict_data which should be
            # similar structure to dic_files
            dict_data[sub.stem][run.stem] = mne.io.read_raw_fif(str(run))

    for sub in dict_data.keys():
        for run in dict_data[sub].keys():
            data = dict_data[sub][run]
            data.load_data()
            data_resampled = data_resample(data, opt.fs_ds)
            rs_data = data_resample(data_resampled, opt.fs_ds)\
                          .get_data()[64:, :]
            rs_removed_raw = data_resampled.drop_channels(\
                ['t0', 't1', 't2', 'r0', 'r1', 'r2'])

            normalized_raw, ecg_mean, ecg_std, eeg_mean, eeg_std \
                = normalize_raw_data_multi_ch(rs_removed_raw)

            # data_normalized = data_normalize(data_resampled)

            # Adding the motion data back into the eeg data
            rs_renorm = normalize_rs_data_multi_ch(rs_data, opt.fs_ds)
            normalized_raw.add_channels([rs_renorm], force_update_info=True)

            normalized_raw = modify_motion_data_with_bcg(normalized_raw, opt)

            epoched_data, good_ix = dataset_epoch(dataset=normalized_raw,
                                                  duration=3,
                                                  epoch_rejection=True,
                                                  threshold=5,
                                                  raw_dataset=data_resampled)

            normalized_raw.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])
            epoched_data.drop_channels(['t0', 't1', 't2', 'r0', 'r1', 'r2'])

            # store back into dict_data
            # dict_data[sub][run] = data_epoch(data_normalized, t)
            dict_data[sub][run] = epoched_data

            # we also want to store the epoch_indeces! This is important
            # so that later if we choose to apply the model on original
            # data, we can reconstruct what was train/test/validate.
            # Easiest would be as a list of lists[ [from0, to0],
            # [from1, to1], … etc.]

    # maybe come up with some epoch rejection criteria here (maybe not, whatever)

   #  for sub in dict_data:
   #      for session in dict_data[sub]:
   #          for run in dict_data[sub][session]:
   #              data = dict_data[sub][session][run]
   #              # stretch, apply epoch rejection criterion
   #              # use input_feature_type or input_feature to convert data to X and y
   #              # contantate all Xs and all ys (separately)
   #
   #   # split X, y and epoch_indeces into train/test/validate by using opt spec
   #   # save X, y, opt to d_features into a neat package to be loaded… ttv in next module will then generate an arch per package
   #
   #  return d_features


@contextlib.contextmanager
def temp_seed(seed):
    # see bcg_net.py for the use of this (it’s to recreate epoching with
    # same seed)
    state = np.random.get_state()
    np.random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)

def modify_motion_data_with_bcg(rs_set, opt, shift=None):
    opt_local = opt
    # if not opt_local.multi_sub and opt_local.use_rs_data and opt_local.multi_ch:
    # if opt_local.use_rs_data and opt_local.multi_ch:
    data = rs_set.get_data()
    info = rs_set.info

    # makes me very nervous!
    eeg_data = data[0:64, :]

    # This length has to be 6 or else...
    electrode_list = ['F5', 'F6', 'P5', 'P6', 'TP9', 'TP10']  # also ugly place to put this

    bcg_input = np.zeros((len(electrode_list), np.shape(data)[1]))
    for ix, electrode in enumerate(electrode_list):
        bcg_input[ix, :] = data[info['ch_names'].index(electrode), :].reshape(1, -1)
        if shift:
            bcg_input[ix, :] = np.roll(bcg_input[ix, :], shift)  # need to check directions
            # not taking care of the edges - meh, this is just proof of concept

    modified_data = np.append(eeg_data, bcg_input, axis=0)
    modified_rs_raw = mne.io.RawArray(modified_data, info)

    return modified_rs_raw

# Normalizing the motion data
def normalize_rs_data_multi_ch(rs_data, fs):
    rs_info = mne.create_info(['t0', 't1', 't2', 'r0', 'r1', 'r2'], fs,
                              ['misc', 'misc', 'misc', 'misc', 'misc', 'misc'])
    transformational_data = rs_data[0:3, :]
    rotational_data = rs_data[3:, :]

    # transformational_data_renorm = np.append([[0], [0], [0]], np.diff(transformational_data, axis=1) * fs * 1e6, axis=1)
    transformational_data_renorm = transformational_data * 1e8
    # used to be 1e9 and derivatives
    rotational_data_renorm = rotational_data * 1e7
    # used to be 1e8

    rs_data_renorm = np.insert(transformational_data_renorm, 3, rotational_data_renorm, axis=0)
    rs_renorm = mne.io.RawArray(rs_data_renorm, rs_info)
    return rs_renorm


# Performing epoching on the raw data set that's provided
def dataset_epoch(dataset, duration, epoch_rejection, threshold=None, raw_dataset=None, good_ix=None):
    # Constructing events of duration 10s
    info = dataset.info
    fs = info['sfreq']

    total_time_stamps = dataset.get_data().shape[1]
    constructed_events = np.zeros(shape=(int(np.floor(total_time_stamps/fs)/duration), 3), dtype=int)

    for i in range(0, int(np.floor(total_time_stamps/fs))-duration, duration):
        ix = i/duration
        constructed_events[int(ix)] = np.array([i*fs, 0, 1])

    tmax = duration - 1/fs

    # Epoching the data using the constructed event and plotting it
    old_epoched_data = mne.Epochs(dataset, constructed_events, tmin=0, tmax=tmax)

    if epoch_rejection:
        # Epoch rejection based on median absolute deviation of mean of absolute values for individual epochs
        ix = single_subject_mabs(raw_dataset, threshold)
        good_ix = np.delete(np.arange(0, old_epoched_data.get_data().shape[0], 1), ix)
        good_data = old_epoched_data.get_data()[good_ix, :, :]
        epoched_data = mne.EpochsArray(good_data, old_epoched_data.info)

        return epoched_data, good_ix
    else:
        epoched_data = old_epoched_data.get_data()[good_ix, :, :]
        return epoched_data


def data_epoch(data, t):
    # mne based epoching
    return mne.Epochs(data, t, tmin=0, tmax=3)


def data_normalize(data):

    return data


def data_resample(data, fs_ds):
    # see bcg_net.py
    # although I am not sure if we want this here! Maybe it should be in
    # feature_extraction
    data.resample(fs_ds)
    return data


# Normalize the data by subtracting the mean and then dividing it by its std for multiple channels
def normalize_raw_data_multi_ch(raw_data, target_ch=[]):
    # Assuming that raw data is an mne Raw object
    data = raw_data.get_data()
    info = raw_data.info
    ecg_ch = info['ch_names'].index('ECG')
    if not target_ch:
        target_ch = np.delete(np.arange(0, len(info['ch_names']), 1), ecg_ch)

    # used for reverting back to original data later
    ecg_mean = np.mean(data[ecg_ch, :])
    ecg_std = np.std(data[ecg_ch, :])
    eeg_mean = np.mean(data[target_ch, :], axis=1)
    eeg_std = np.std(data[target_ch, :], axis=1)

    normalizedData = np.zeros(data.shape)
    for i in range(data.shape[0]):
        ds = data[i, :] - np.mean(data[i, :])
        ds /= np.std(ds)
        normalizedData[i, :] = ds

    normalized_raw = mne.io.RawArray(normalizedData, info)
    return normalized_raw, ecg_mean, ecg_std, eeg_mean, eeg_std


if __name__ == '__main__':
    """ used for debugging """
    # Person = namedtuple('Person', 'name age gender d_features')
    # opt = Person(name='John', age=45, gender='male', d_features = 'settings.d_root/proc_bcgnet/features/')
    #
    # f = generate_ws_features('d_mne', opt)

    generate_ws_features(None, opt_default())


