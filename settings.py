"""
Directory structure
"""
from pathlib import Path


class GitLabel():
    def __init__(self):
        try:
            self.label_ = subprocess.check_output(["git", "describe", "--tag"]).strip().decode("utf-8")
        except:
            self.label_ = 'vx.x.x'
            print('Not git tag found! Setting to {}\n'.format(self.label_))

    def __str__(self):
        return str(self.label_)

    @property
    def label(self):
        return self.label_

    @property
    def tag(self):
        return self.label_.split('-')[0]


def init(_project_name, _d_root):
    global project_name
    project_name = _project_name  # user forced to specify

    global experiment_id
    experiment_id = ''  # something that identifies parameters of the
    # experiment uniquely (e.g. which architecture? What features?)

    global d_root
    d_root = _d_root  # user forced to specify project directory for
    # output and intermediate processing settable location for source
    # of GA removed data, for GA removed data. Already high-pass
    # filtered at 0.1Hz? Downsampled to 500Hz?:

    global d_data
    d_data = d_root / 'proc' / 'ga_rm'  # settable data location (this
    # can be our default, but I guess user should be forced to specify
    # full path)

    global d_mne
    d_mne = d_root / 'proc_bcgnet' / 'mne/'  # data_loader.py output goes here

    # TODO: This one is outdated. To be deleted.
    global d_features
    d_features = d_root / 'proc_bcgnet' / 'features' / '$experiment_id$'
    # feature_extractor.py output goes here

    global git_label
    git_label = GitLabel()

    global home
    home = Path.home()

    global p_data
    p_data = home / 'Local'

    global str_experiment
    str_experiment = 'working_eegbcg'

    global str_proc
    str_proc = 'proc_full'

    global str_proc_rs
    str_proc_rs = 'proc_rs'

    global str_proc_obs
    str_proc_obs = 'proc_bcgobs'

    global str_proc_net
    str_proc_net = 'proc_net'

    global str_proc_net_cv
    str_proc_net_cv = 'proc_net_cv'

    global str_proc_bcgnet_mat
    str_proc_bcgnet_mat = 'proc_bcgnet_mat'

    global str_proc_bcgnet
    str_proc_bcgnet = 'proc_bcgnet'

    global str_proc_test_epochs
    str_proc_test_epochs = 'proc_test_epochs'

    global str_proc_cv_epochs
    str_proc_cv_epochs = 'proc_cv_epochs'

    global str_proc_sssr
    str_proc_sssr = 'single_sub'

    global str_proc_ssmr
    str_proc_ssmr = 'multi_run'

    global str_proc_msmr
    str_proc_msmr = 'multi_sub'

    global str_bash
    str_bash = 'bash'

    global str_log
    str_log = 'log'

    global t_arch
    t_arch = 'gru_arch'


def obs_path(str_sub, run_id):
    """
    Obtain the path to the OBS cleaned dataset and the correct filename

    :param str_sub: string for subject to run analysis on
    :param run_id: index of the run from the subject

    :return: p_bcg: pathlib.Path objects that holds the path to the OBS cleaned dataset
    :return: f_bcg: filename of the OBS cleaned dataset
    """

    p_bcg = p_data / str_experiment / str_proc / str_proc_obs / str_sub
    f_bcg = '{}_r0{}_rmbcg.set'.format(str_sub, run_id)
    return p_bcg, f_bcg


def rs_path(str_sub, run_id):
    """
    Obtain the path to the resampled dataset and the correct filename

    :param str_sub: string for subject to run analysis on
    :param run_id: index of the run from the subject

    :return: p_rs: pathlib.Path objects that holds the path to the resampled dataset
    :return: f_rs: filename of the resampled dataset
    """

    p_rs = p_data / str_experiment / str_proc / str_proc_rs / str_sub
    f_rs = '{}_r0{}_rs.set'.format(str_sub, run_id)
    return p_rs, f_rs
