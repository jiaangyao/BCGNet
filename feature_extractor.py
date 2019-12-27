import mne
from pathlib import Path
from collections import namedtuple
import contextlib
import numpy as np
import settings


def opt_default():
    # this is a function in feature_extractor.py with default settings
    # not sure if there is a type in the channel settings in MNE, but if
    # so that would be the easiest…:
    Opt = namedtuple('Opt', ['input_feature_type', 'output_feature_type',
                             'input_feature', 'output_features',
                             'd_features', 't_epoch', 'generate', 'fs_ds'])

    return Opt(
        input_feature_type = ['ecg'],
        output_feature_type = ['eeg'], # if the feature_type opt are None,
        # then you can specify manually, e.g. opt.input_feature = [0, 1, 2]
        # or, opt.input_feature = [‘Fz’, ‘Cz’] etc.
        input_feature = None,
        output_features = None,
        d_features =  settings.d_root / 'proc_bcgnet/features/',
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
    d_features  =  opt.generate(d_mne, opt)  # so that we can easily switch it out
    return d_features


def generate_ws_features(d_mne, opt):
   # n.b. This is the code that generates features to train and test within
   # subject… different functions would need to be defined for different
   # required feature sets. I suggest we start with this one, then see how
   # we go….
    d_mne = Path(opt.d_mne)
    d_features = Path(opt.d_features)
    d_features = d_features / 'something_unique_based_on_opt'  # (I suggest
   # a hash of opt, combined with some useful human readable stuff)
#
#     dict_files= …  # a structure that contains organised files. E.g.:
#     # dict_files[‘subject01][0] = [0, 1, 2]  # i.e. [subject][session][run]
#
#    # load in all the data - reason  for this is that we -might- need to do single run processing based on all runs (data should be small, but we can change this structure if it gets problematic)
#     for sub in dict_files:
#         for session in dict_files[sub]
#             for run in dict_files[sub][session]
#                 # load each run with MNE and store in dict_data which should be similar structure to dic_fiiles
#
#     for sub in dict_data:
#         for session in dict_data[sub]:
#             for run in dict_data[sub][session]:
#                 data = dict_data[sub][session][run]
#                 # data_resample(data, opt.fs_ds)
#                 # data_noramlize
#                 # data_epoch(data, t)
#                 # store back into dict_data
#                 # we also want to store the epoch_indeces! This is important so that later if we choose to apply the model on original data, we can reconstruct what was train/test/validate. Easiest would be as a list of lists[ [from0, to0], [from1, to1], … etc.]
#
# # maybe come up with some epoch rejection criteria here (maybe not, whatever)
#
#     for sub in dict_data:
#         for session in dict_data[sub]:
#             for run in dict_data[sub][session]:
#                 data = dict_data[sub][session][run]
#                 # stretch, apply epoch rejection criterion
#                 # use input_feature_type or input_feature to convert data to X and y
#                 # contantate all Xs and all ys (separately)
#
#      # split X, y and epoch_indeces into train/test/validate by using opt spec
#      # save X, y, opt to d_features into a neat package to be loaded… ttv in next module will then generate an arch per package

    return d_features


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


def data_resample(data, fs_ds):
    # see bcg_net.py
    # although I am not sure if we want this here! Maybe it should be in
    # feature_extraction
    data.resample(fs_ds)
    return data


def data_epoch(data, t):
    # mne based epoching
    return mne.Epochs(data, t, tmin=0, tmax=3)


if __name__ == '__main__':
    """ used for debugging """



