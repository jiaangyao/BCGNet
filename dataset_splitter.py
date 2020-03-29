import numpy as np
import mne
from utils.context_management import temp_seed


def generate_train_valid_test(epoched_dataset, opt=None):
    """
    Generate the training, validation and test epoch data from the MNE Epoch object based on the training set ratio,
    and validation set ratio provided in the opt object

    NOTE: vec_ix_slice_validation and vec_ix_slice_test are important when comparing the performance
    between the multiple models, calculating the RMSE value, and performing the PSD analysis since we want to use
    the same epochs from the ground_truth_eeg data for analysis purposes

    :param epoched_dataset: MNE Epoch object where epoched_dataset.get_data() is the normalized epoched data and is the
        form of (epoch, channel, data)
    :param opt: option object that was defined by the user and contains all the hyperparameters

    :return: xs: list containing all the ECG data in the form of [x_train, x_validation, x_test], where each element
        is of the form (epoch, channel, data)
    :return: ys: list containing all the corrupted EEG data in the form of [y_train, y_validation, y_test], where each
        element is of the form (epoch, channel, data)
    :return: vec_ix_slice: list in the form of [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test],
        where each element contains the indices of epochs in the original dataset belonging to the training, validation
        and test set respectively
    """

    # Obtain the data and the index of the ECG channel from the MNE Epoch object
    # Note that normalized_data is in the form (epoch, channel, data)
    normalized_data = epoched_dataset.get_data()
    ecg_ch = epoched_dataset.info['ch_names'].index('ECG')

    # Obtain the total number of epochs
    num_epochs = normalized_data.shape[0]

    # Temporarily set the random seed so that the data will be split in the same way
    with temp_seed(1997):
        # Generate the test set and (training + validation) set
        s_ev, s_test, vec_ix_slice_evaluation, vec_ix_slice_test = split_evaluation_test(normalized_data,
                                                                                         opt.training +
                                                                                         opt.validation)

        # Obtain the number of epochs in the (training + validation) set
        ev_epoch_num = int(np.round(num_epochs * (opt.training + opt.validation)))

        # Recalculate the percentage of validation epochs in the evaluation set so that the final outcome is
        # num_epochs * opt.validation
        per_validation = int(np.round(opt.validation * num_epochs))/ev_epoch_num

        # Generate the training and validation sets
        s_train, s_val, vec_ix_slice_train, vec_ix_slice_val = split_train_validation(s_ev, per_validation)

        # Obtain the ECG data in each set
        x_train = s_train[:, ecg_ch, :]
        x_validation = s_val[:, ecg_ch, :]
        x_test = s_test[:, ecg_ch, :]

        # Obtain the EEG data in each set
        y_train = np.delete(s_train, ecg_ch, axis=1)
        y_validation = np.delete(s_val, ecg_ch, axis=1)
        y_test = np.delete(s_test, ecg_ch, axis=1)

        # Obtain the indices for the validation set epochs in the original epoched_dataset
        vec_ix_slice_validation = vec_ix_slice_evaluation[vec_ix_slice_val]

        # Obtain the indices for the training set epochs in the original epoched_dataset
        vec_ix_slice_training = vec_ix_slice_evaluation[vec_ix_slice_train]

    # Package everything together into a list
    xs = [x_train, x_validation, x_test]
    ys = [y_train, y_validation, y_test]
    vec_ix_slice = [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test]

    return xs, ys, vec_ix_slice


def generate_train_valid_test_mr(vec_epoched_dataset, vec_run_id, opt=None):
    """
    Wrapper function using the generate_train_valid_test function to combine multiple epoched datasets first and then
    split the combined epoched dataset into training, validation and test sets

    :param vec_epoched_dataset: list containing mne.EpochArray objects where each object contains epoched data
        from a single run
    :param vec_run_id: list containing the indices for al the runs to be analyzed
    :param opt: option object that was defined by the user and contains all the hyperparameters

    :return: mr_combined_xs: list containing all the ECG data in the form of [x_train, x_validation, x_test], where each
        element is of the form (epoch, channel, data)
    :return: mr_combined_ys: list containing all the corrupted EEG data in the form of [y_train, y_validation, y_test],
        where each element is of the form (epoch, channel, data)
    :return: mr_vec_ix_slice: list in the form of [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test],
        where each element contains the indices of epochs in the original dataset belonging to the training, validation
        and test set respectively
    :return: mr_ten_ix_slice: list of list of list goes like [run1, run2, ...], where each list has the structure
        [training, validation, test], and each of the run list contains indices of epochs from a run within the
        particular set
    """

    mr_combined_epoched_dataset = None
    for i in range(len(vec_epoched_dataset)):
        if i == 0:
            mr_combined_epoched_dataset = vec_epoched_dataset[i]
        else:
            mr_combined_epoched_dataset = mne.concatenate_epochs([mr_combined_epoched_dataset, vec_epoched_dataset[i]])

    vec_n_events = [0]
    for i in range(len(vec_epoched_dataset)):
        n_events = vec_epoched_dataset[i].get_data().shape[0]
        vec_n_events.append(n_events)
    vec_n_events = np.cumsum(vec_n_events)

    mr_combined_xs, mr_combined_ys, mr_vec_ix_slice = generate_train_valid_test(mr_combined_epoched_dataset, opt)

    # rearrange the indices in terms of runs
    mr_ten_ix_slice = []
    for i in range(len(vec_run_id)):
        mr_mat_ix_slice_run = []
        for j in range(len(mr_vec_ix_slice)):
            vec_ix_slice_set = mr_vec_ix_slice[j]
            mr_vec_ix_slice_set = vec_ix_slice_set[np.where(np.logical_and(vec_n_events[i] <= vec_ix_slice_set,
                                                                           vec_ix_slice_set < vec_n_events[i + 1]))]

            mr_vec_ix_slice_set = mr_vec_ix_slice_set - vec_n_events[i]
            mr_mat_ix_slice_run.append(mr_vec_ix_slice_set)
        mr_ten_ix_slice.append(mr_mat_ix_slice_run)

    return mr_combined_xs, mr_combined_ys, mr_vec_ix_slice, mr_ten_ix_slice


def generate_train_valid_test_cv(epoched_dataset, opt=None):
    """
    Generate the training, validation and test epoch data from the MNE Epoch object based on the training set ratio,
    and validation set ratio provided in the opt object in a cross validation manner

    NOTE: vec_ix_slice_validation and vec_ix_slice_test are important when comparing the performance
    between the multiple models, calculating the RMSE value, and performing the PSD analysis since we want to use
    the same epochs from the ground_truth_eeg data for analysis purposes

    :param epoched_dataset: MNE Epoch object where epoched_dataset.get_data() is the normalized epoched data and is the
        form of (epoch, channel, data)
    :param opt: option object that was defined by the user and contains all the hyperparameters

    :return: vec_xs: list containing all the ECG data in the form of [fold1, fold2, ...], where each fold is in the form
        of [x_train, x_validation, x_test] and each element is of the form (epoch, data)
    :return: vec_ys: list containing all the corrupted EEG data in the form of [fold1, fold2, ...] where each fold is
        in the form [y_train, y_validation, y_test] and each element is of the form (epoch, channel, data)
    :return: mat_ix_slice: list in the form of [fold1, fold2, ...], where each fold is of the form of
        [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test], where each element contains the indices of
        epochs in the original dataset belonging to the training, validation and test set respectively
    """

    # Obtain the data and the index of the ECG channel from the MNE Epoch object
    # Note that normalized_data is in the form (epoch, channel, data)
    normalized_epoched_data = epoched_dataset.get_data()
    ecg_ch = epoched_dataset.info['ch_names'].index('ECG')

    # Obtain the total number of epochs
    num_epochs = normalized_epoched_data.shape[0]

    # Temporarily set the random seed so that the data will be split in the same way
    with temp_seed(1997):
        # Split everything into int(ceil(1/per_fold)) number of folds of roughly equal sizes
        permuted_vec_ix_epoch = np.random.permutation(num_epochs)
        num_fold = int(np.ceil(1/opt.per_fold))
        mat_ix_slice_test = np.array_split(permuted_vec_ix_epoch, num_fold)

        # Define the empty arrays to hold the variables
        mat_ix_slice = []

        vec_xs = []
        vec_ys = []

        # Loop through each fold and determine the validation set and training set for each fold according to defined
        # percentages
        for i in range(len(mat_ix_slice_test)):
            vec_ix_slice_test = mat_ix_slice_test[i]
            if not np.all(np.isin(vec_ix_slice_test, permuted_vec_ix_epoch)):
                raise Exception('Erroneous CV fold splitting')
            s_test = normalized_epoched_data[vec_ix_slice_test, :, :]

            # Obtain the indices that correspond to training + validation set and permute it
            vec_ix_evaluation = np.setdiff1d(permuted_vec_ix_epoch, vec_ix_slice_test)
            permuted_vec_ix_evaluation = vec_ix_evaluation[np.random.permutation(len(vec_ix_evaluation))]

            # Obtain the validation epochs
            num_epochs_validation = int(np.round(num_epochs * opt.validation))
            vec_ix_validation = permuted_vec_ix_evaluation[:num_epochs_validation]
            s_validation = normalized_epoched_data[vec_ix_validation, :, :]

            # Obtain the training epochs
            vec_ix_training = permuted_vec_ix_evaluation[num_epochs_validation:]
            s_training = normalized_epoched_data[vec_ix_training]

            # Obtain the xs and the ys
            x_training = s_training[:, ecg_ch, :]
            x_validation = s_validation[:, ecg_ch, :]
            x_test = s_test[:, ecg_ch, :]

            y_training = np.delete(s_training, ecg_ch, axis=1)
            y_validation = np.delete(s_validation, ecg_ch, axis=1)
            y_test = np.delete(s_test, ecg_ch, axis=1)

            # Package everything into a single list
            xs = [x_training, x_validation, x_test]
            ys = [y_training, y_validation, y_test]
            vec_ix_slice = [vec_ix_training, vec_ix_validation, vec_ix_slice_test]

            # Append those to the outer lists holding everything
            vec_xs.append(xs)
            vec_ys.append(ys)
            mat_ix_slice.append(vec_ix_slice)

    return vec_xs, vec_ys, mat_ix_slice


def generate_train_valid_test_cv_mr(vec_epoched_dataset, vec_run_id, opt=None):
    """
    Wrapper function using the generate_train_valid_test function to combine multiple epoched datasets first and then
    split the combined epoched dataset into training, validation and test sets in a CV manner

    :param vec_epoched_dataset: list containing mne.EpochArray objects where each object contains epoched data
        from a single run
    :param vec_run_id: list containing the indices for al the runs to be analyzed
    :param opt: option object that was defined by the user and contains all the hyperparameters

    :return:
    """

    mat_xs = []
    mat_ys = []
    mr_cv_ten_ix_slice = []
    for i in range(len(vec_epoched_dataset)):
        epoched_dataset = vec_epoched_dataset[i]
        vec_xs, vec_ys, mat_ix_slice = generate_train_valid_test_cv(epoched_dataset, opt)

        mat_xs.append(vec_xs)
        mat_ys.append(vec_ys)
        mr_cv_ten_ix_slice.append(mat_ix_slice)

    # mat_xs originally in the form of [run1, run2, ...], where each run goes [fold1, fold2, ...] and each fold goes
    # like [training, validation, test]. In the end, want mat_xs and mat_ys to be in the form of [fold1, fold2, ...],
    # where each fold is the form of [training, validation, test]

    # Loop through the folds
    mr_cv_combined_xs = []
    mr_cv_combined_ys = []
    for i in range(len(mat_xs[0])):
        xs_fold = []
        ys_fold = []

        # Loop through the sets
        for j in range(len(mat_xs[0][0])):
            xs_set = []
            ys_set = []

            # Loop through the runs
            for k in range(len(mat_xs)):
                xs_fold_set_run = mat_xs[k][i][j]
                ys_fold_set_run = mat_ys[k][i][j]

                xs_set.append(xs_fold_set_run)
                ys_set.append(ys_fold_set_run)

            xs_set = np.concatenate(xs_set, axis=0)
            ys_set = np.concatenate(ys_set, axis=0)

            xs_fold.append(xs_set)
            ys_fold.append(ys_set)

        mr_cv_combined_xs.append(xs_fold)
        mr_cv_combined_ys.append(ys_fold)

    return mr_cv_combined_xs, mr_cv_combined_ys, mr_cv_ten_ix_slice


# Obtain the test set from the rest of the data
def split_evaluation_test(epoched_data, per_evaluation):
    """
    Split the epoched data into test set and (training + validation) set

    :param epoched_data: numpy array containing the normalized epoched data and is in the form (orig_epoch, channel, data)
    :param per_evaluation: desired percentage of (training + validation) epochs, i.e. (1 - per_test)

    :return: epochs_evaluation: (training + validation) set, and has shape (orig_epoch * per_evaluation,
        channel, data)
    :return: epochs_test: test set, and has shape (orig_epoch * (1 - per_evaluation), channel, data)
    :return: vec_ix_slice_evaluation: index of epochs in the original epoched_data that goes to epochs_evaluation
    :return: vec_ix_slice_test: index of epochs in the original epoched_data that goes to epochs_test
    """

    # Perform permutation on indices of the epochs and then compute a cutoff based on per_evaluation
    vec_ix = np.random.permutation(len(epoched_data))
    vec_ix_cutoff = int(np.round(len(epoched_data) * per_evaluation))

    # Split the original set into (training + validation) set
    vec_ix_slice_evaluation = vec_ix[:vec_ix_cutoff]
    epochs_evaluation = epoched_data[vec_ix_slice_evaluation, :, :]

    # Put all the remaining epochs into the test set
    vec_ix_slice_test = vec_ix[vec_ix_cutoff:]
    epochs_test = epoched_data[vec_ix_slice_test, :, :]

    return epochs_evaluation, epochs_test, vec_ix_slice_evaluation, vec_ix_slice_test


# Splits evaluation data into training set and validation set
def split_train_validation(epochs_evaluation, per_validation):
    """
    Split the (training + validation) set into training set and validation set

    :param epochs_evaluation: numpy array from the previous function and is of the form (eva_epochs, channel, data)
    :param per_validation: desired percentage of validation epochs

    :return: epochs_train: training set, and has shape (eva_epochs * (1 - per_validation), channel, data)
    :return: epochs_validation: test set, and has shape (eva_epochs * per_validation, channel, data)
    :return: vec_ix_slice_train: index of epochs in the epochs_evaluation that goes into the training set
    :return: vec_ix_slice_val: index of epochs in the epochs_evaluation that goes into the validation set
    """

    # Perform permutation on indices of the epochs and then compute a cutoff based on per_validation
    vec_ix = np.random.permutation(len(epochs_evaluation))
    vec_ix_cutoff = int(np.round(len(epochs_evaluation) * per_validation))

    # Split the evaluation set into training set
    vec_ix_slice_train = vec_ix[vec_ix_cutoff:]
    epochs_train = epochs_evaluation[vec_ix_slice_train, :, :]

    # Split the evaluation set into validation set
    vec_ix_slice_val = vec_ix[:vec_ix_cutoff]
    epochs_validation = epochs_evaluation[vec_ix_slice_val, :, :]
    return epochs_train, epochs_validation, vec_ix_slice_train, vec_ix_slice_val


# Split the mne.io_ops.RawArray object holding the raw data in the form of time series
def split_epoched_dataset(epoched_dataset, vec_ix_slice):
    """
    Split the mne.io_ops.Mne object holding the cleaned EEG data into the same set of training, validation and test
    epochs as during the training

    :param epoched_dataset: mne.io_ops.Mne object that holds the epoched data, note that the data held has the form
        (epoch, channel, data)
    :param vec_ix_slice: list in the form of [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test],
        where each element contains the indices of epochs in the original dataset belonging to the training, validation
        and test set respectively

    :return: vec_epoched_dataset_set: list in the form of [epoched_dataset_training, epoched_dataset_validation,
        epoched_dataset_test], where each is an mne.io_ops.Mne object that holds epoched data belonging to the specified set
        during training
    """

    # Get the data and info object first
    epoched_data = epoched_dataset.get_data()
    info = epoched_dataset.info

    # Get the training, validation and test data
    epoched_data_training = epoched_data[vec_ix_slice[0], :, :]
    epoched_data_validation = epoched_data[vec_ix_slice[1], :, :]
    epoched_data_test = epoched_data[vec_ix_slice[2], :, :]

    epoched_dataset_training = mne.EpochsArray(epoched_data_training, info)
    epoched_dataset_validation = mne.EpochsArray(epoched_data_validation, info)
    epoched_dataset_test = mne.EpochsArray(epoched_data_test, info)

    vec_epoched_dataset_set = [epoched_dataset_training, epoched_dataset_validation, epoched_dataset_test]

    return vec_epoched_dataset_set


def split_epoched_dataset_mr(vec_epoched_dataset, mr_ten_ix_slice):
    """
    Wrapper function that uses split_epoched_dataset to split the data for all runs for a single subject

    :param vec_epoched_dataset: list containing the epoched dataset from all runs of a single subject
    :param mr_ten_ix_slice: list of list of list goes like [run1, run2, ...], where each list has the structure
        [training, validation, test], and each of the run list contains indices of epochs from a run within the
        particular set

    :return: mat_epoched_dataset_set: list of list where each list is in the form of [epoched_dataset_training,
        epoched_dataset_validation, epoched_dataset_test], where each is an mne.io_ops.Mne object that holds epoched
        data belonging to the specified set during training
    """

    mat_epoched_dataset_set = []
    for i in range(len(vec_epoched_dataset)):
        epoched_dataset = vec_epoched_dataset[i]
        vec_ix_slice = mr_ten_ix_slice[i]

        vec_epoched_dataset_set = split_epoched_dataset(epoched_dataset, vec_ix_slice)
        mat_epoched_dataset_set.append(vec_epoched_dataset_set)

    return mat_epoched_dataset_set


def gen_batches_rnn(data_list, flag_input):
    """
    Takes in continuous time windows and applies reshape and transpose so that they work with the shape of the Keras
    models

    :param data_list: list of the form [training_set, validation_set, test_set]; each set has the shape
        (epochs, m_time_stamps)
    :param flag_input: boolean for whether it's the data (x) or the labels (y)

    :return: batch_list = [training_set (one sample apart), validation_set (one sample apart), test_set (one sample apart)]
    """
    # In the case of RNN data, no padding was performed and all that's required is to transpose the data

    if flag_input:
        # For ECG data

        # create the list that will hold the batches from training, validation and test set
        batch_list = []

        # Loop through the sets
        for i in range(len(data_list)):
            # Keras wants data in the form of (epochs, data, 1)

            # Our input has data in the form of (epochs, data)
            data_i = data_list[i]

            # A simple reshape is enough
            batches = data_i.reshape(data_i.shape[0], data_i.shape[1], 1)

            # Append to the grand list
            batch_list.append(batches)

    else:
        # For corrupted EEG data

        # create the list that will hold the batches from training, validation and test set
        batch_list = []

        # Loop through the sets
        for i in range(len(data_list)):
            # Keras wants data in the form of (epochs, data, 63)

            # Our input has data in the form of (epochs, 63, data)
            data_i = data_list[i]

            # Then a simple transpose is enough
            batches = np.transpose(data_i, axes=(0, 2, 1))

            # Append to the grand list
            batch_list.append(batches)

    return batch_list
