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
        d_mne = 'mne', # settings.d_root / 'proc_bcgnet/mne/',  # if I was writing
        # this myself, I would use environment variables to specify d_root,
        # maybe there are opinions on this...
        run_regex = '?', # capture subject name, session and run
        overwrite = False
    )


def convert_to_mne(d_ga_removed, opt):
    # put everything into a standard mne format.
    # d_save = Path(opt.d_mne)
    # d_ga_removed = d_save  # some regex operation using
    d_output = Path('/home/yida/Local/working_eegbcg/test_output/')

    for sub in d_ga_removed.iterdir():
        for run in (d_ga_removed / sub).iterdir():
            if run.suffix == '.set' and not 'all' in run.stem\
                    and run.stat().st_size > 0:
                out_dir = d_output / sub.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / (run.stem + '_raw.fif')

                if not out_file.is_file() or opt.overwrite:
                    data = load_eeglab(run)
                    data.opt = opt
                    data_save(data, out_file, overwrite=opt.overwrite)


def data_save(data, f_output, overwrite=False):
    # I am thinking scipy.io.savemat (so it should also be matlab readable)
    # we want to save data and also opt - together (we could put opt in data)
    # scipy.io.savemat(str(f_output), {'data': data})
    data.save(str(f_output), overwrite=overwrite)


def load_eeglab(f_input):  # see opt.load_type
    # See bcg_net.py I guess…
    data = mne.io.read_raw_eeglab(str(f_input), preload=True, stim_channel=False)
    # make sure that channel names are OK and channel types (e.g. they have ‘eeg’ or ‘ecg’ type)
    return data

  
if __name__ == '__main__':
    """ used for debugging """
    d_ga_removed = Path('/home/yida/Local/working_eegbcg/proc_full/proc_rs/')
    convert_to_mne(d_ga_removed, opt_default())
