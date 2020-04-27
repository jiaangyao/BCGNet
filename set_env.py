from pathlib import Path


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


home = Path.home()


if module_exists('dnc_set_env'):
    import dnc_set_env
    home = dnc_set_env.home()
elif home.exists():
    print('')  # default case
else:
    raise Exception('Local directory not found?')

p_data = home / 'Local'
str_experiment = 'working_eegbcg'
t_arch = 'gru_arch'


def ga_path(str_sub, run_id):
    p_ga = p_data.joinpath(str_experiment).joinpath('proc_full/proc_ga').joinpath((str_sub))
    f_ga = '{}_r0{}_rmga.set'.format(str_sub, run_id)
    return p_ga, f_ga


def rs_path(str_sub, run_id):
    p_rs = p_data.joinpath(str_experiment).joinpath('proc_full/proc_rs').joinpath((str_sub))
    f_rs = '{}_r0{}_rs.set'.format(str_sub, run_id)
    return p_rs, f_rs


def bcg_path(str_sub, run_id):
    p_bcg = p_data.joinpath(str_experiment).joinpath('proc_full/proc_bcgobs').joinpath((str_sub))
    f_bcg = '{}_r0{}_rmbcg.set'.format(str_sub, run_id)
    return p_bcg, f_bcg


def arch_path(str_sub, run_id, arch=None):
    import numpy as np
    import re

    if str_sub is None and run_id is None:
        p_arch = p_data.joinpath(str_experiment).joinpath('proc_full/proc_net')
        return p_arch

    if isinstance(str_sub, (list, tuple, np.ndarray)):
        str_sub_joined = ''.join(re.findall(r'\d+', x)[0] + '_' for x in str_sub)
        str_sub_joined = str_sub_joined[:-1]
        str_sub_local = 'sub{}'.format(str_sub_joined)

        if not isinstance(run_id, (list, tuple, np.ndarray)):
            p_arch = p_data.joinpath(str_experiment).joinpath('proc_full/proc_net').joinpath(str_sub_local).joinpath(t_arch).joinpath(arch).joinpath('r0{}'.format(run_id))
        else:
            run_id_str = ''.join(str(x) for x in run_id)
            p_arch = p_data.joinpath(str_experiment).joinpath('proc_full/proc_net').joinpath(str_sub_local).joinpath(t_arch).joinpath(arch).joinpath('r0{}'.format(run_id_str))
        return p_arch

    if not isinstance(run_id, (list, tuple, np.ndarray)):
        p_arch = p_data.joinpath(str_experiment).joinpath('proc_full/proc_net').joinpath(str_sub).joinpath(t_arch).joinpath(
        'r0{}'.format(run_id))
    else:
        run_id_str = ''.join(str(x) for x in run_id)
        p_arch = p_data.joinpath(str_experiment).joinpath('proc_full/proc_net').joinpath(str_sub).joinpath(t_arch).joinpath(
        'r0{}'.format(run_id_str))

    return p_arch


def hyperas_path(str_sub, run_id):
    p_hyperas = p_data.joinpath(str_experiment).joinpath('proc_full/proc_net/hyperas').joinpath(str_sub)
    f_hypeas = '{}_r0{}_hyperas.hyperopt'.format(str_sub, run_id)
    f_model = '{}_r0{}_hyperas.dat'.format(str_sub, run_id)
    return p_hyperas, f_hypeas, f_model


def conda_path_cpu():
    p_conda = home / 'miniconda3/envs/py-bcg/bin/python'
    return p_conda


def conda_path_gpu():
    p_conda = home / 'miniconda3/envs/py-bcg-gpu-custom/bin/python'
    return p_conda


def bash_path():
    p_bash = p_data.joinpath(str_experiment) / 'bash script'
    p_log = p_data.joinpath(str_experiment) / 'log'
    return p_bash, p_log


def output_path(str_sub, run_id, arch, mode=None, opt=None):
    opt_local = opt
    if not opt_local.multi_sub:
        if opt_local.multi_run:
            p_out = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('proc_bcgnet_mat').joinpath('multi_run').joinpath(arch).joinpath(str_sub)
            f_out = '{}_r0{}_bcgnet.mat'.format(str_sub, run_id)
        else:
            p_out = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('proc_bcgnet_mat').joinpath('single_run').joinpath(arch).joinpath(str_sub)
            f_out = '{}_r0{}_bcgnet.mat'.format(str_sub, run_id)

    else:
        p_out = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('proc_bcgnet_mat').joinpath(
            'multi_sub').joinpath(arch).joinpath(str_sub)
        if mode == 'pre-trained':
            f_out = '{}_r0{}_bcgnet_pre_trained.mat'.format(str_sub, run_id)
        else:
            f_out = '{}_r0{}_bcgnet.mat'.format(str_sub, run_id)
    return p_out, f_out

def bcg_output_path(str_sub, run_id, arch, opt=None):
    opt_local = opt
    if not opt_local.multi_sub:
        if opt_local.multi_run:
            p_out = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('proc_bcgnet_mat').joinpath('multi_run').joinpath(arch).joinpath(str_sub)
            f_out = '{}_r0{}_bcgnet_purebcg.mat'.format(str_sub, run_id)
        else:
            p_out = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('proc_bcgnet_mat').joinpath('single_run').joinpath(arch).joinpath(str_sub)
            f_out = '{}_r0{}_bcgnet_purebcg.mat'.format(str_sub, run_id)
    return p_out, f_out


def cleaned_dataset_path(str_sub, run_id, arch, trial_type):
    p_cleaned = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('proc_bcgnet').joinpath(trial_type).joinpath(arch).joinpath(str_sub)
    f_cleaned = '{}_r0{}_bcgnet.set'.format(str_sub, run_id)

    return p_cleaned, f_cleaned

def as_dataset_pretrained_path(str_sub, run_id, arch, trial_type):
    p_pretrained = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('proc_bcgnet').joinpath(trial_type).joinpath(arch).joinpath(str_sub)
    f_pretrained = '{}_r0{}_bcgnet_pre_trained.set'.format(str_sub, run_id)

    return p_pretrained, f_pretrained


def figure_path(fignum):
    p_figure = p_data.joinpath(str_experiment).joinpath('proc_full').joinpath('figures').joinpath('fig0{}'.format(fignum))

    return p_figure