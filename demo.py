"""
demo.py - the custom script a user modifies (highest level). I guess we also want
a non-python interface version of this using argparse. Take a look at
cluster/cluster_functions.py for an example of how to do this.
"""
from pathlib import Path
import settings
import options
import preprocessor
import dataset_splitter
import training
import ttv

settings.init(Path.home(), Path.home())  # Call only once

# Parameters
str_sub = 'sub11'
run_id = 1
opt = options.test_opt(None)
str_arch = 'gru_arch_general4'

# Preprocess
normalized_epoched_raw_dataset, normalized_raw_dataset, epoched_raw_dataset, \
raw_dataset, orig_sr_epoched_raw_dataset, orig_sr_raw_dataset, \
ecg_stats, eeg_stats, good_idx = preprocessor.preprocess_subject(str_sub=str_sub, run_id=run_id, opt=opt)

# Split data
xs, ys, vec_ix_slice = dataset_splitter.generate_train_valid_test(normalized_epoched_raw_dataset, opt=opt)

# Obtain the training and validation generators
training_generator = training.Defaultgenerator(xs[0], ys[0], batch_size=opt.batch_size, shuffle=True)
validation_generator = training.Defaultgenerator(xs[1], ys[1], batch_size=opt.batch_size, shuffle=True)

# Train and fit
model, callbacks_, m, epochs = ttv.train(training_generator, validation_generator, opt=opt,
                                         str_arch=str_arch)
ttv.predict(model, callbacks_, normalized_raw_dataset, raw_dataset, orig_sr_raw_dataset, ecg_stats, eeg_stats, opt,
            good_idx)
