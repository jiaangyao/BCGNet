from collections import namedtuple
import tensorflow as tf
import settings
import bcg_net_architecture
import bcg_net_architecture.arch0001
import bcg_net_architecture.gru_arch_general4
from preprocessor import preprocess_subject
from training import *
from predict import *
from options import test_opt
from dataset_splitter import generate_train_valid_test, _test_generate_train_valid_test_

Opt = namedtuple('Opt', ['input_feature', 'output_features',
                         'd_features', 't_epoch', 'generate',
                         'fs_ds', 'p_training', 'p_validation',
                         'p_evaluation'])


def opt_default():
    """
    This is a function in ttv.py with default settings.

    :return:
    """
    Opt = namedtuple('Opt', ['epochs', 'es_min_delta', 'es_patience',
                             'early_stopping', 'resume', 'overwrite',
                             'validation', 'ttv_split', 'debug_mode',
                             'arch', 'extra_string'])

    return Opt(
        epochs=2500,
        es_min_delta=1e-5,
        es_patience=25,  # How many times does the validation not increase
        early_stopping=True,
        resume=True,
        overwrite=False,
        validation=None,
        ttv_split=[0.7, 0.15, 0.15],  # train/test/validate split
        debug_mode=False,  # more output, also plots
        arch=bcg_net_architecture.arch0001,
        extra_string=''
    )


def train(training_generator, validation_generator, opt=test_opt(None), str_arch='gru_arch_general4'):
    """
    See bcg_net.py for inspiration.  The extra thing to consider is that
    it would be nice to be able to switch between CNN and RNN which require
    fundamentally different data restructuring {- maybe - this could go in
    the architecture but probably there is some reason I have not thought
    about it, for why this doesn’t work }. I have example code for this,
    but may not be in bcg_net.

    :param d_features:
    :param opt:
    :return:
    """
    # Tensorflow session configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    # Obtain the model and callback
    model = get_arch_rnn(str_arch, opt.lr)
    callbacks_ = get_callbacks_rnn(opt)

    # Fitting the model
    m = model.fit_generator(generator=training_generator, epochs=opt.epochs, verbose=2, callbacks=callbacks_,
                            validation_data=validation_generator)

    epochs = len(m.epoch)

    return model, callbacks_, m, epochs


def predict(model, callbacks_, normalized_raw_dataset, raw_dataset, orig_sr_raw_dataset, ecg_stats, eeg_stats, opt_user,
            good_idx):
    """

    :return:
    """
    # Predict the cleaned dataset and epoch it for comparison later
    orig_sr_epoched_cleaned_dataset, orig_sr_cleaned_dataset, epoched_cleaned_dataset, cleaned_dataset = \
        predict_time_series(model, callbacks_, normalized_raw_dataset, raw_dataset, orig_sr_raw_dataset, ecg_stats,
                            eeg_stats, opt_user.epoch_duration, good_idx)

    return orig_sr_epoched_cleaned_dataset, orig_sr_cleaned_dataset, epoched_cleaned_dataset, cleaned_dataset


def clean():
    """

    :return:
    """
    return


def _test_train_(opt=test_opt(None), str_arch='gru_arch_general4'):
    # Generate the train, validation and test sets and also obtain the index of epochs used in the validation
    # and test set
    xs, ys, vec_ix_slice = _test_generate_train_valid_test_()
    # Obtain the training and validation generators
    training_generator = Defaultgenerator(xs[0], ys[0], batch_size=opt.batch_size, shuffle=True)
    validation_generator = Defaultgenerator(xs[1], ys[1], batch_size=opt.batch_size, shuffle=True)
    model, callbacks_, m, epochs = train(training_generator, validation_generator, opt, str_arch)
    return model, callbacks_, m, epochs


def _test_predict_(str_arch='gru_arch_general4', str_sub='sub11', run_id=1, opt=test_opt(None)):
    # Generate the train, validation and test sets and also obtain the index of epochs used in the validation
    # and test set
    normalized_epoched_raw_dataset, normalized_raw_dataset, epoched_raw_dataset, \
    raw_dataset, orig_sr_epoched_raw_dataset, orig_sr_raw_dataset, \
    ecg_stats, eeg_stats, good_idx = preprocess_subject(str_sub, run_id, opt)
    xs, ys, vec_ix_slice = generate_train_valid_test(normalized_epoched_raw_dataset, opt=opt)
    # Obtain the training and validation generators
    training_generator = Defaultgenerator(xs[0], ys[0], batch_size=opt.batch_size, shuffle=True)
    validation_generator = Defaultgenerator(xs[1], ys[1], batch_size=opt.batch_size, shuffle=True)
    model, callbacks_, m, epochs = train(training_generator, validation_generator, opt, str_arch)

    """
    Prediction
    """
    predict(model, callbacks_, normalized_raw_dataset, raw_dataset, orig_sr_raw_dataset, ecg_stats, eeg_stats, opt,
            good_idx)

    return


if __name__ == '__main__':
    """ used for debugging """
    from pathlib import Path

    settings.init(Path.home(), Path.home())  # Call only once
    _test_train_()
    _test_predict_()
