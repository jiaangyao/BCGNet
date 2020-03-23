"""
demo.py - the custom script a user modifies (highest level). I guess we also want
a non-python interface version of this using argparse. Take a look at
cluster/cluster_functions.py for an example of how to do this.
"""
import bcg_net
import data_loader
import feature_extractor
import ttv
import settings
from pathlib import Path


settings.init(Path.home(), Path.home())  # Call only once

opt_data_loader = data_loader.opt_default()
opt_data_loader.something = 'something_else'
d_mne = data_loader.convert_to_mne(settings.d_root, opt_data_loader)

opt_feature_extractor = feature_extractor.opt_default()
opt_feature_extractor.something = 'something_else'
d_features = feature_extractor.generate(d_mne, opt_feature_extractor)

opt_ttv = ttv.opt_default()
opt_ttv.something = 'something_else'
d_model = ttv.train(d_features, opt_ttv)  # train the model
bcg = ttv.predict(d_features, d_model, opt_ttv)  # evaluate the model on all data
eeg = ttv.clean(d_features, d_model, d_mne, opt_ttv)  # upsample bcg back to eeg sampling rate and subtract

# then we can write some helper stuff to do something with the output




