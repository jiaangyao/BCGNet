"""
demo2.py: A non-python interface for running the toolbox using argparse.
"""
import argparse
from pathlib import Path
import settings
import options
import preprocessor
import dataset_splitter
import training
import ttv
from utils.compute_rms import compute_rms


def run_training(str_sub, vec_run_id, str_arch):
    settings.init(Path.home(), Path.home())  # Call only once
    opt = options.test_opt(None)

    # Preprocess
    # Load, normalize and epoch the raw dataset from all runs
    vec_normalized_epoched_raw_dataset, vec_normalized_raw_dataset, vec_epoched_raw_dataset, vec_raw_dataset, \
    vec_orig_sr_epoched_raw_dataset, vec_orig_sr_raw_dataset, vec_ecg_stats, vec_eeg_stats, vec_good_idx \
        = preprocessor.preprocess_subject(str_sub, vec_run_id, opt)

    # Split the epoched dataset into training, validation and test sets
    mr_combined_xs, mr_combined_ys, mr_vec_ix_slice, mr_ten_ix_slice = \
        dataset_splitter.generate_train_valid_test_mr(vec_normalized_epoched_raw_dataset, vec_run_id, opt=opt)

    # Obtain the training and validation generators
    training_generator = training.Defaultgenerator(mr_combined_xs[0], mr_combined_ys[0], batch_size=opt.batch_size,
                                                   shuffle=True)
    validation_generator = training.Defaultgenerator(mr_combined_xs[1], mr_combined_ys[1], batch_size=opt.batch_size,
                                                     shuffle=True)

    # Train and fit
    model, callbacks_, m, epochs = ttv.train(training_generator, validation_generator, opt=opt, str_arch=str_arch)
    vec_orig_sr_epoched_cleaned_dataset, vec_orig_sr_cleaned_dataset, vec_epoched_cleaned_dataset, vec_cleaned_dataset \
        = ttv.predict(model, callbacks_, vec_normalized_raw_dataset, vec_raw_dataset, vec_orig_sr_raw_dataset,
                      vec_ecg_stats, vec_eeg_stats, opt, vec_good_idx)

    # Results
    mat_rms_test = compute_rms(vec_run_id, vec_orig_sr_epoched_raw_dataset, vec_orig_sr_epoched_cleaned_dataset,
                               mr_ten_ix_slice)

    return mat_rms_test


if __name__ == '__main__':
    FUNCTION_MAP = {'run_training': run_training}
    parser = argparse.ArgumentParser(description='something')
    parser.add_argument('--function', choices=FUNCTION_MAP.keys(), default='run_training')
    # TODO: I currently changed str_sub to a single string, but I saw in py_bcg_linbi it's a list. Need to discuss how to handle this.
    parser.add_argument('--str_sub', nargs='*', type=str, default='sub11')
    parser.add_argument('--vec_run_id', nargs='*', type=int, default=[1], help='')
    parser.add_argument('--str_arch', type=str, default='gru_arch_general4')
    args = parser.parse_args()
    method = FUNCTION_MAP[args.function]

    method(str_sub=args.str_sub,
           vec_run_id=args.vec_run_id,
           str_arch=args.str_arch)
