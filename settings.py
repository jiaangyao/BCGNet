# Directory structure

def init(_project_name, _d_root):
    global project_name
    project_name = _project_name # user forced to specify

    global experiment_id
    experiment_id = '' # something that identifies parameters of the experiment uniquely (e.g. which architecture? What features?)

    global d_root
    d_root = _d_root # user forced to specify project directory for output and intermediate processing
    # settable location for source of GA removed data, for GA removed data. Already high-pass filtered at 0.1Hz? Downsampled to 500Hz?:

    global d_data
    d_data = d_root / 'proc/ga_rm' # settable data location (this can be our default, but I guess user should be forced to specify full path)

    global d_mne
    d_mne = d_root / 'proc_bcgnet/mne/' # data_loader.py output goes here

    global d_features
    d_features = d_root / 'proc_bcgnet/features/$experiment_id$' # feature_extractor.py output goes here