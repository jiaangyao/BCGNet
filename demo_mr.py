"""
demo.py - the custom script a user modifies (highest level). I guess we also want
a non-python interface version of this using argparse. Take a look at
cluster/cluster_functions.py for an example of how to do this.
"""
from datetime import datetime
from pathlib import Path
import settings
import options
import preprocessor
import dataset_splitter
import training
import ttv
from utils.compute_rms import compute_rms

settings.init(Path.home(), Path.home())  # Call only once

# Parameters
str_sub = 'sub11'
run_id = [1, 2, 3, 4, 5]
opt = options.test_opt(None)
str_arch = 'gru_arch_general4'

# Preprocess
# Load, normalize and epoch the raw dataset from all runs
vec_normalized_epoched_raw_dataset, vec_normalized_raw_dataset, vec_epoched_raw_dataset, vec_raw_dataset, \
vec_orig_sr_epoched_raw_dataset, vec_orig_sr_raw_dataset, \
vec_ecg_stats, vec_eeg_stats, vec_good_idx \
 = preprocessor.preprocess_subject_mr(str_sub, run_id, str_arch, opt)
