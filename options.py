from utils.options_default import *


# TODO: reorganize the configuration into dictionary
def test_opt(opt):
    """
    Class used for initializing hyperparameters of the training

    Processing parameters:
    _epoch_duration: duration in seconds of each time epoch to split the data into: float
    _mad_threshold: multiples of the median absolute deviation for outlier tests to reject the time epochs: float
    __int: new_fs: new sampling rate

    Training parameters:
    _epochs: number of epochs: int
    _training: percentage to be used as the training set: float
    _validation: percentage to be used as the validation set: float
    _batch_size: batch size for the training: int
    _lr: learning rate for the optimizer: float
    _multi_sub: flag for whether or not you are training on multiple subjects: boolean
    _multi_sub: flag for whether or not you are training on multiple runs: boolean
    _use_motion_data: flag for whether or not you are using motion data in addition to the ECG data: boolean
    _training_dataset_gen: whether to generate the dataset during training: boolean
    _training_figure_gen: whether to generate the figures during training: boolean
    _training_figure_num: number of intermediate figures to generate during training: int

    Early stopping parameters:
    _early_stopping: flag for turning on the early stopping feature: boolean
    _es_min_delta:  minimal change to be considered as an improvement by Keras: float
    _es_patience: patience factor for early stopping in Keras: int

    Evaluation mode parameters:
    _evaluate_model: flag for turning on the evaluation feature: boolean
    _evaluation_dataset_gen: whether to generate the dataset during evaluation: boolean
    _evaluation_figure_gen: whether to generate figures while evaluating the model: boolean
    _evaluation_figure_num: number of intermediate figures to generate while evaluating the model: int

    Cross validation mode parameters
    _per_fold: percentage of total data for each fold used in cross validation: float
    _training_cv_dataset_gen: whether to generate the dataset during training: boolean
    _training_cv_figure_gen: whether to generate the figures during training: boolean
    _training_cv_figure_num: number of intermediate figures to generate during training: int
    """

    if opt is None:
        # Initialize object
        opt = TrainDefault()

        # Processing parameters
        opt.epoch_duration = 3
        opt.mad_threshold = 5
        opt.new_fs = 100

        # Training parameters
        opt.epochs = 3
        opt.training = 0.70
        opt.validation = 0.15
        opt.batch_size = 1
        opt.lr = 1e-3
        opt.multi_sub = False
        opt.multi_run = True
        opt.use_motion_data = False
        opt.training_dataset_gen = False
        opt.training_figure_gen = True
        opt.training_figure_num = 63

        # Early stopping parameters
        opt.early_stopping = True
        opt.es_min_delta = 1e-5
        opt.es_patience = 25

        # Evaluation mode parameters
        # opt.evaluate_model = False
        opt.evaluation_dataset_gen = False
        opt.evaluation_figure_gen = False
        opt.evaluation_figure_num = 0

        # Cross validation parameters
        opt.per_fold = 0.15
        opt.training_cv_dataset_gen = False
        opt.training_cv_figure_gen = True
        opt.training_cv_figure_num = 63

    elif isinstance(opt, dict):
        assert False, 'need to add this option?'

    return opt
