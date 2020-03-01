from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers, callbacks, regularizers, optimizers
from tensorflow.python.keras import backend as K
from collections import namedtuple
import settings
import bcg_net_architecture
import bcg_net_architecture.arch0001
import bcg_net_architecture.gru_arch_general4
import dill
import numpy as np
import matplotlib as plt
import datetime
from pathlib import Path


Opt = namedtuple('Opt', ['input_feature', 'output_features',
                         'd_features', 't_epoch', 'generate',
                         'fs_ds', 'p_training', 'p_validation',
                         'p_evaluation'])


def opt_default():
    # this is a function in ttv.py with default settings
    Opt = namedtuple('Opt', ['epochs', 'es_min_delta', 'es_patience',
                             'early_stopping', 'resume', 'overwrite',
                             'validation', 'ttv_split', 'debug_mode',
                             'arch', 'extra_string'])

    return Opt(
        epochs = 2500,
        es_min_delta = 1e-5,
        es_patience = 25,  # How many times does the validation not increase
        early_stopping = True,
        resume = True,
        overwrite = False,
        validation = None,
        ttv_split = [0.7, 0.2, 0.1],  # train/test/validate split
        debug_mode = False,  # more output, also plots
        arch = bcg_net_architecture.arch0001,
        extra_string = ''
    )


def train(d_features, opt):
    # See bcg_net.py for inspiration.  The extra thing to consider is that
    # it would be nice to be able to switch between CNN and RNN which require
    # fundamentally different data restructuring {- maybe - this could go in
    # the architecture but probably there is some reason I have not thought
    # about it, for why this doesn’t work }. I have example code for this,
    # but may not be in bcg_net.

    file = open(d_features, 'rb')
    data_dict = dill.load(file)
    file.close()

    data = data_dict['data']
    opt_local = data_dict['opt']

    f_arch = opt_local.d_features / 'arch_epoch_{}_fs_{}'\
        .format(opt_local.t_epoch, opt_local.fs_ds)
    opt_def = opt_default()
    training_res = [None] * len(data)

    # for each package of X, y, opt in d_features, let’s train and test!
    #     load train, test and validation features, also opt_feature_extract
    #     do stuff
    #
    #      arch = opt.arch.create_arch(n_output, opt_feature_extract)
    #      arch_name = opt.arch.get_name()
    #
    #      f_arch = ?  # again some name and hash that depends on the opt.
    # It should probably be in a folder that has the same name as the d_features
    # folder (i.e. what I called something_unique_based_on_opt in
    # generate_ws_features) but with a file that is unique based on the arch.
    #
    # return


        # if f_arch exists, we load it, if it is finished training that’s it
        # if it hasn’t finished training, we finish training
        # if it doesn’t exist we start training

        # model = get_arch(arch, opt=opt_local)

    for i, subj_data in enumerate(data):
        x_train = subj_data['x_train']
        x_validation = subj_data['x_validation']
        x_test = subj_data['x_test']
        y_train = subj_data['y_train']
        y_validation = subj_data['y_validation']
        y_test = subj_data['y_test']
        vec_ix_slice_test = subj_data['vec_ix_slice_test']

        training_res[i] = train_sp(x_train, x_validation, y_train, y_validation,
                       arch='gru_arch_general4', overwrite=False,
                       held_out=False, opt=opt_def)

    return training_res


def predict():
    return


def clean():
    return


def train_sp(x_ev_train, x_ev_validation, y_ev_train, y_ev_validation,
             arch='gru_arch_000', overwrite=False, held_out=False, opt=None):
    opt_local = opt

    # if opt_local.f_arch is None:
    #     str_arch = 'net_{}_{}.dat'.format(arch, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    #     dir_arch = opt_local.p_arch.joinpath(Path(str_arch.split('.')[0]))
    #     dir_arch.mkdir(parents=True, exist_ok=True)
    #     opt_local.p_arch = dir_arch
    #     opt.f_arch = Path(str_arch)
    #     f_arch = opt_local.p_arch.joinpath(opt.f_arch)
    #
    # if isinstance(opt_local.f_arch, str):
    #     str_arch = opt_local.f_arch.split('.')[0]
    #     opt_local.p_arch = opt_local.p_arch.joinpath(str_arch)
    #     f_arch = opt_local.p_arch.joinpath(Path(opt_local.f_arch))

    d_features_local = Path('/home/yida/Local/working_eegbcg/proc_bcgnet/features/')
    f_arch = d_features_local / 'arch_epoch_{}'.format(opt_local.epochs)
    n_input, n_output = 64, 1
    model = bcg_net_architecture.gru_arch_general4.create_arch(n_input, n_output, opt_local)

    # Generator for the training set, the output is a tuple of the xs and ys
    def train_generator():
        ix = 0
        # if opt_local.use_rs_data:
        vec_ix = np.random.permutation(x_ev_train.shape[1])
        # else:
        #     vec_ix = np.random.permutation(x_ev_train.shape[0])

        while True:
            # if not opt_local.use_rs_data:
            # xs = x_ev_train[vec_ix[ix], :].reshape(1, -1, 1)
            # else:
            xs_full = np.transpose(x_ev_train[:, vec_ix[ix], :]).reshape(1, -1, 7)
            xs_ecg = xs_full[0, :, 0].reshape(1, -1, 1)
            xs_rs = xs_full[0, :, 1:].reshape(1, -1, 6)
            xs = [xs_ecg, xs_rs]

            # if not opt_local.multi_ch:
            #     ys = y_ev_train[vec_ix[ix], :].reshape(1, -1, 1)
            # else:
            ys = np.transpose(y_ev_train[:, vec_ix[ix], :]).reshape(1, -1, 63)
            ix += 1

            # if not opt_local.use_rs_data:
            # ix = ix % x_ev_train.shape[0]
            # else:
            ix = ix % x_ev_train.shape[1]
            yield xs, ys

    # Generator for the validation set, the output is a tuple of the xs and ys
    def validation_generator():
        ix = 0
        # if opt_local.use_rs_data:
        vec_ix = np.random.permutation(x_ev_validation.shape[1])
        # else:
            # vec_ix = np.random.permutation(x_ev_validation.shape[0])


        while True:
            # if not opt_local.use_rs_data:
            #     xs = x_ev_validation[vec_ix[ix], :].reshape(1, -1, 1)
            # else:
            xs_full = np.transpose(x_ev_validation[:, vec_ix[ix], :]).reshape(1, -1, 7)
            xs_ecg = xs_full[0, :, 0].reshape(1, -1, 1)
            xs_rs = xs_full[0, :, 1:].reshape(1, -1, 6)
            xs = [xs_ecg, xs_rs]

            # if not opt_local.multi_ch:
            #     ys = y_ev_validation[vec_ix[ix], :].reshape(1, -1, 1)
            # else:
            ys = np.transpose(y_ev_validation[:, vec_ix[ix], :]).reshape(1, -1, 63)
            ix += 1
            # if not opt_local.use_rs_data:
            #     ix = ix % x_ev_validation.shape[0]
            # else:
            ix = ix % x_ev_validation.shape[1]

            yield xs, ys

    if f_arch.exists() and not overwrite:
        # print('Loading {}'.format(f_arch))
        with open(str(f_arch), 'rb') as handle:
            net_settings = dill.load(handle)

        weights = net_settings['weights']

        if held_out:
            epochs_ori = net_settings['epochs']

        if 'finished' in net_settings:
            resume_overwrite = not (net_settings['finished'])
        else:
            resume_overwrite = False
        resume = opt_local.resume or resume_overwrite

        # you have to evaluate it before setting for some reason (maybe an older bug):
        model.set_weights(weights=weights)
    else:
        resume = False

    finished = True
    if not (f_arch.exists()) or resume or overwrite:
        print('Generating {}'.format(f_arch))
        if opt_local.epochs < 5:
            print('Only {} epochs selected... testing?'.format(opt_local.epochs))

        # if not opt_local.use_rs_data:
        steps_per_epoch_train = x_ev_train.shape[0]
        steps_per_epoch_validation = x_ev_validation.shape[0]
        # else:
        #     steps_per_epoch_train = x_ev_train.shape[1]
        #     steps_per_epoch_validation = x_ev_validation.shape[1]

        if opt_local.validation is None:
            model.fit_generator(train_generator(), steps_per_epoch=steps_per_epoch_train, epochs=opt_local.epochs,
                                verbose=2)
        else:
            if opt_local.early_stopping:
                # 'an absolute change of less than min_delta, will count as no improvement'
                callbacks_ = [callbacks.EarlyStopping(monitor='val_loss', min_delta=opt_local.es_min_delta,
                                                      patience=opt_local.es_patience, verbose=0, mode='min',
                                                      restore_best_weights=True)]
            else:
                callbacks_ = None

            m = model.fit_generator(train_generator(), steps_per_epoch=steps_per_epoch_train,
                                    epochs=opt_local.epochs, verbose=2,
                                    validation_data=validation_generator(),
                                    validation_steps=steps_per_epoch_validation,
                                    callbacks=callbacks_)

            # net_settings['finished']
            if np.max(m.epoch) + 1 >= opt_local.epochs:
                finished = False

        weights = model.get_weights()

        if resume:
            epochs = net_settings['epochs'] + len(m.history['loss'])  # don't like the structure here
        else:
            epochs = len(m.history['loss'])

        net_settings = {'weights': weights, 'arch': arch, 'epochs': epochs, 'finished': finished}

        # Don't like this code here
        if not resume:
            opt_local.p_arch.joinpath('TEp{}'.format(epochs)).mkdir(parents=True, exist_ok=True)
            opt.p_arch = opt_local.p_arch.joinpath('TEp{}'.format(epochs))
            opt.f_arch = f_arch
        else:
            if held_out:
                epochs = epochs_ori
            opt_local.p_arch.joinpath('TEp{}'.format(epochs)).mkdir(parents=True, exist_ok=True)
            opt.p_arch = opt_local.p_arch.joinpath('TEp{}'.format(epochs))
            opt.f_arch = f_arch

        with open(str(f_arch), 'wb') as handle:
            dill.dump(net_settings, handle)

        # Plotting the result of the training and saving the history file
        loss = m.history['loss']
        val_loss = m.history['val_loss']
        vec_epochs = range(1, len(loss) + 1)
        plt.figure(figsize=(6, 6))
        plt.plot(vec_epochs, loss, 'bo', label='Training loss')
        plt.plot(vec_epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        fig = plt.gcf()
        if not held_out:
            fig.savefig(opt.p_arch / 'TrVa_loss_TEp{}.svg'.format(epochs), format='svg')
            with open(opt.p_arch / 'history_rmse', 'wb') as filename:
                dill.dump(m.history, filename)
        else:
            fig.savefig(opt.p_arch / 'TrVa_loss_ho_TEp{}.svg'.format(epochs), format='svg')
            with open(opt.p_arch / 'history_rmse', 'rb') as handle:
                h1 = dill.load(handle)

            h1['history_held_out_sub'] = m.history
            with open(opt.p_arch / 'history_rmse', 'wb') as filename:
                dill.dump(h1, filename)

        return model


if __name__ == '__main__':
    """ used for debugging """
    d_features = '/home/yida/Local/working_eegbcg/proc_bcgnet/features/features.obj'
    train(d_features, None)

