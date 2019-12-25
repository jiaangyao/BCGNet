Directory structure

project_name = something % user forced to specify (?)

experiment_id = ? %something that identifies parameters of the experiment uniquely (e.g. which architecture? What features?)

d_root = wherever %user forced to specify project directory for output and intermediate processing
 % settable location for source of GA removed data, for GA removed data. Already high-pass filtered at 0.1Hz? Downsampled to 500Hz?:

d_data = $d_root$/proc/ga_rm % settable data location (this can be our default, but I guess user should be forced to specify full path)

d_mne = $d_root$/proc_bcgnet/mne/ %data_loader.py output goes here

d_features = $d_root$/proc_bcgnet/features/$experiment_id$ %feature_extractor.py output goes here
