import numpy as np
import mne


def combined_test_epochs(vec_epoched_cleaned_dataset_test, mat_ix_slice, epoched_raw_dataset):
    """
    Combine the test epoch from each fold into a single mne.EpochArray object that's sorted in time

    :param vec_epoched_cleaned_dataset_test: list in the form of [epoched_cleaned_dataset_test1,
        epoched_cleaned_dataset_test2, ...] where each comes from a single fold in the cross validation with
        corresponding indices stored in mat_ix_slice
    :param mat_ix_slice: list in the form of [fold1, fold2, ...], where each fold is of the form of
        [vec_ix_slice_training, vec_ix_slice_validation, vec_ix_slice_test], where each element contains the indices of
        epochs in the original dataset belonging to the training, validation and test set respectively
    :param epoched_raw_dataset: mne.EpochArray object containing the raw epoched data

    :return: combined_epoched_cleaned_dataset: mne.EpochArray object holding the cleaned data from all test epochs in
        a CV manner
    """

    # Create empty arrays that will hold data
    vec_epoched_cleaned_data_test = []
    mat_ix_slice_test = []

    # Loop through all the folds
    for i in range(len(vec_epoched_cleaned_dataset_test)):
        # Obtain the test set data and the corresponding indices for each fold
        epoched_cleaned_data_test = vec_epoched_cleaned_dataset_test[i].get_data()
        vec_ix_slice_test = mat_ix_slice[i][-1]

        # Append those to the list
        vec_epoched_cleaned_data_test.append(epoched_cleaned_data_test)
        mat_ix_slice_test.append(vec_ix_slice_test)

    # Combined all the test epochs from the individual folds, should have shape (num_epoch, channel, channel, data)
    combined_epoched_cleaned_data_test = np.concatenate(vec_epoched_cleaned_data_test, axis=0)

    # Combine all the indices from individual folds
    combined_vec_ix_slice_test = np.concatenate(mat_ix_slice_test, axis=0)

    # Test if the total number of test epochs is equal to the total number of epochs, if not then throw an exception
    if not (len(combined_vec_ix_slice_test) == epoched_raw_dataset.get_data().shape[0]):
        raise Exception('Total number of test epochs not equal to total number of test epochs')

    # Sort the test epochs and check if anything is missing
    vec_idx_sorting = np.argsort(combined_vec_ix_slice_test)
    sorted_combined_vec_ix_slice_test = combined_vec_ix_slice_test[vec_idx_sorting]

    if not np.all(sorted_combined_vec_ix_slice_test == np.arange(epoched_raw_dataset.get_data().shape[0])):
        raise Exception('Some epochs missing')

    # Sort the data accordingly
    sorted_combined_epoched_cleaned_data_test = combined_epoched_cleaned_data_test[vec_idx_sorting, :, :]

    # Construct an mne.EpochArray object that holds the data
    info = epoched_raw_dataset.info
    combined_epoched_cleaned_dataset = mne.EpochsArray(sorted_combined_epoched_cleaned_data_test, info)

    return combined_epoched_cleaned_dataset
