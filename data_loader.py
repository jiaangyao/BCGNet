# My thought with how to deal with different experimental structures/data types
# etc. is: we assume that everyone has ‘subject/session_xx/??????%02d???.file’
# or ‘subject/??????%02d???.file’, the user can specify a regex for the run in
# opt (we put a sensible default), and if their structure is really different
# they need to define their own convert_to_mne and load_eeglab….
# No matter what the input is, we output into a standardised format which is
# formatted: #‘subject/subject_session%02d/subject_session%02d_run%02d.mne_extension’
# This is good for our sanity when writing feature_extractor.py

import mne
from pathlib import Path
from collections import namedtuple
import scipy.io
import settings
import re


def opt_default():
    # opt should be a namedtuple so syntax needs changing!
    Opt = namedtuple('Opt', ['load_type', 'd_mne', 'run_regex', 'overwrite'])

    return Opt(
        load_type = load_eeglab,  # function handle to the function that we
        # want to use to load the actual data. To make it easily extensible.
        d_mne = settings.d_root / 'proc_bcgnet/mne/',  # if I was writing
        # this myself, I would use environment variables to specify d_root,
        # maybe there are opinions on this...
        run_regex = '?', # capture subject name, session and run
        overwrite = False
    )


def convert_to_mne(d_ga_removed, opt):
    # put everything into a standard mne format.
    d_save = Path(opt.d_mne)
    d_ga_removed = d_save  # some regex operation using

    for f_input in d_ga_removed:
        f_output = f_input # some regex operations on f_input using
        # opt.run_regex (might need its own function)
        f_output = d_save / f_output
        # check if  f_output exists… if it does not (or opt.overwrite is false)
        # then do following:
        if not f_output.is_file() or opt.overwrite:
            data = load_eeglab(f_input)
            data_save(data, f_output)

    return


def data_save(data, f_output):
    # I am thinking scipy.io.savemat (so it should also be matlab readable)
    # we want to save data and also opt - together (we could put opt in data)
    scipy.io.savemat(str(f_output), {'data': data})


def load_eeglab(f_input):  # see opt.load_type
    # See bcg_net.py I guess…
    data = mne.io.read_raw_eeglab(str(f_input), preload=True, stim_channel=False)
    # make sure that channel names are OK and channel types (e.g. they have ‘eeg’ or ‘ecg’ type)
    return data

  
if __name__ == '__main__':
    """ used for debugging """
    


