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
from dataset_splitter import _test_generate_train_valid_test_, generate_train_valid_test_mr

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


def predict(model, callbacks_, vec_normalized_raw_dataset, vec_raw_dataset, vec_orig_sr_raw_dataset, vec_ecg_stats,
            vec_eeg_stats, opt, vec_good_idx):
    """
    TODO: Write doc

    :return:
    """
    # Predict the cleaned dataset and epoch it for comparison later
    vec_orig_sr_epoched_cleaned_dataset, vec_orig_sr_cleaned_dataset, vec_epoched_cleaned_dataset, \
    vec_cleaned_dataset = predict_time_series_mr(model, callbacks_, vec_normalized_raw_dataset, vec_raw_dataset,
                                                 vec_orig_sr_raw_dataset, vec_ecg_stats, vec_eeg_stats,
                                                 opt.epoch_duration, vec_good_idx)

    return vec_orig_sr_epoched_cleaned_dataset, vec_orig_sr_cleaned_dataset, vec_epoched_cleaned_dataset, \
           vec_cleaned_dataset


def clean():
    """

    :return:
    """
    return


def _test_train_(str_sub, vec_run_id, str_arch='gru_arch_general4', opt=test_opt(None)):
    if not isinstance(vec_run_id, list):
        if isinstance(vec_run_id, int):
            vec_run_id = [vec_run_id]
        else:
            raise Exception("Unsupported type; vec_run_id must be a list or an int.")

    # Generate the train, validation and test sets and also obtain the index of epochs used in the validation
    # and test set
    mr_combined_xs, mr_combined_ys, mr_vec_ix_slice, mr_ten_ix_slice = _test_generate_train_valid_test_(str_sub,
                                                                                                        vec_run_id)
    # Obtain the training and validation generators
    training_generator = Defaultgenerator(mr_combined_xs[0], mr_combined_ys[0], batch_size=opt.batch_size, shuffle=True)
    validation_generator = Defaultgenerator(mr_combined_xs[1], mr_combined_ys[1], batch_size=opt.batch_size,
                                            shuffle=True)
    model, callbacks_, m, epochs = train(training_generator, validation_generator, opt, str_arch)
    return model, callbacks_, m, epochs


def _test_predict_(str_sub, vec_run_id, str_arch='gru_arch_general4', opt=test_opt(None)):
    if not isinstance(vec_run_id, list):
        if isinstance(vec_run_id, int):
            vec_run_id = [vec_run_id]
        else:
            raise Exception("Unsupported type; vec_run_id must be a list or an int.")

    # Preprocess
    # Load, normalize and epoch the raw dataset from all runs
    vec_normalized_epoched_raw_dataset, vec_normalized_raw_dataset, vec_epoched_raw_dataset, vec_raw_dataset, \
    vec_orig_sr_epoched_raw_dataset, vec_orig_sr_raw_dataset, vec_ecg_stats, vec_eeg_stats, vec_good_idx \
        = preprocess_subject(str_sub, vec_run_id)
    mr_combined_xs, mr_combined_ys, mr_vec_ix_slice, mr_ten_ix_slice \
        = generate_train_valid_test_mr(vec_normalized_epoched_raw_dataset, vec_run_id, opt=opt)
    # Obtain the training and validation generators
    training_generator = Defaultgenerator(mr_combined_xs[0], mr_combined_ys[0], batch_size=opt.batch_size, shuffle=True)
    validation_generator = Defaultgenerator(mr_combined_xs[1], mr_combined_ys[1], batch_size=opt.batch_size,
                                            shuffle=True)
    model, callbacks_, m, epochs = train(training_generator, validation_generator, opt, str_arch)

    """
    Prediction
    """
    vec_orig_sr_epoched_cleaned_dataset, vec_orig_sr_cleaned_dataset, vec_epoched_cleaned_dataset, vec_cleaned_dataset \
        = predict(model, callbacks_, vec_normalized_raw_dataset, vec_raw_dataset, vec_orig_sr_raw_dataset,
                  vec_ecg_stats, vec_eeg_stats, opt, vec_good_idx)

    return vec_orig_sr_epoched_cleaned_dataset, vec_orig_sr_cleaned_dataset, vec_epoched_cleaned_dataset, \
           vec_cleaned_dataset


if __name__ == '__main__':
    """ used for debugging """
    from pathlib import Path

    settings.init(Path.home(), Path.home())  # Call only once
    _test_train_(str_sub='sub11', vec_run_id=1)
    _test_predict_(str_sub='sub11', vec_run_id=1)
