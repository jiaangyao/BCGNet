#  Class used for setting hyperparameters of the training
class TrainDefault:
    """
    Class used for initializing hyperparameters of the training

    Processing parameters:
    _epoch_duration: duration in seconds of each time epoch to split the data into: float
    _mad_threshold: multiples of the median absolute deviation for outlier tests to reject the time epochs: float
    __n_downsampling: factor of downsampling during the preprocessing: int

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

    def __init__(self):
        # Processing parameters
        self._epoch_duration = 3
        self._mad_threshold = 5
        self._n_downsampling = 5

        # Training parameters
        self._epochs = 2500
        self._training = 0.70
        self._validation = 0.15
        self._batch_size = 1
        self._lr = 1e-2
        self._multi_sub = False
        self._multi_run = False
        self._use_motion_data = False
        self._training_dataset_gen = True
        self._training_figure_gen = True
        self._training_figure_num = 63

        # Early stopping parameters
        self._early_stopping = True
        self._es_min_delta = 1e-5
        self._es_patience = 25

        # Evaluation mode parameters
        self._evaluate_model = False
        self._evaluation_dataset_gen = False
        self._evaluation_figure_gen = False
        self._evaluation_figure_num = 0

        # Cross validation mode
        self._per_fold = 0.15
        self._training_cv_dataset_gen = True
        self._training_cv_figure_gen = True
        self._training_cv_figure_num = 63

    # Defining all the properties
    # Processing parameters
    @property
    def epoch_duration(self):
        return self._epoch_duration

    @property
    def mad_threshold(self):
        return self._mad_threshold

    @property
    def n_downsampling(self):
        return self._n_downsampling


    # Training parameters

    @property
    def epochs(self):
        return self._epochs

    @property
    def training(self):
        training = self._training
        if training is not None:
            if training >= 1:
                raise Exception('percentage of training cannot be greater than 1')
            if training == 0:
                training = None

        return training

    @property
    def validation(self):
        validation = self._validation
        if validation is not None:
            if validation >= 1:
                raise Exception('validation cannot be greater than 1')
            if validation == 0:
                validation = None

        return validation

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def lr(self):
        return self._lr

    @property
    def multi_sub(self):
        return self._multi_sub

    @property
    def multi_run(self):
        return self._multi_run

    @property
    def use_motion_data(self):
        return self._use_motion_data

    @property
    def training_dataset_gen(self):
        return self._training_dataset_gen

    @property
    def training_figure_gen(self):
        return self._training_figure_gen

    @property
    def training_figure_num(self):
        return self._training_figure_num


    # Early stopping parameters

    @property
    def early_stopping(self):
        return self._early_stopping

    @property
    def es_min_delta(self):
        return self._es_min_delta

    @property
    def es_patience(self):
        return self._es_patience


    # Evaluation mode parameters

    @property
    def evaluate_model(self):
        return self._evaluate_model

    @property
    def evaluation_dataset_gen(self):
        return self._evaluation_dataset_gen

    @property
    def evaluation_figure_gen(self):
        return self._evaluation_figure_gen

    @property
    def evaluation_figure_num(self):
        return self._evaluation_figure_num


    # Cross validation parameters

    @property
    def per_fold(self):
        return self._per_fold

    @property
    def training_cv_dataset_gen(self):
        return self._training_cv_dataset_gen

    @property
    def training_cv_figure_gen(self):
        return self._training_figure_gen

    @property
    def training_cv_figure_num(self):
        return self._training_figure_num



    # Defining all the mutator methods
    # Processing parameters

    @epoch_duration.setter
    def epoch_duration(self, value):
        self._epoch_duration = value

    @mad_threshold.setter
    def mad_threshold(self, value):
        self._mad_threshold = value

    @n_downsampling.setter
    def n_downsampling(self, value):
        self._n_downsampling = value


    # Training parameters

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @training.setter
    def training(self, value):
        self._training = value

    @validation.setter
    def validation(self, value):
        self._validation = value

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @lr.setter
    def lr(self, value):
        self._lr = value

    @multi_sub.setter
    def multi_sub(self, value):
        self._multi_sub = value

    @multi_run.setter
    def multi_run(self, value):
        self._multi_run = value

    @use_motion_data.setter
    def use_motion_data(self, value):
        self._use_motion_data = value

    @training_dataset_gen.setter
    def training_dataset_gen(self, value):
        self._training_dataset_gen = value

    @training_figure_gen.setter
    def training_figure_gen(self, value):
        self._training_figure_gen = value

    @training_figure_num.setter
    def training_figure_num(self, value):
        self._training_figure_num = value


    # Early stopping parameters

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value

    @es_min_delta.setter
    def es_min_delta(self, value):
        self._es_min_delta = value

    @es_patience.setter
    def es_patience(self, value):
        self._es_patience = value


    # Evaluation mode parameter

    @evaluate_model.setter
    def evaluate_model(self, value):
        self._evaluate_model = value

    @evaluation_dataset_gen.setter
    def evaluation_dataset_gen(self, value):
        self._evaluation_dataset_gen = value

    @evaluation_figure_gen.setter
    def evaluation_figure_gen(self, value):
        self._evaluation_figure_gen = value

    @evaluation_figure_num.setter
    def evaluation_figure_num(self, value):
        self._evaluation_figure_num = value


    # Cross validation parameters

    @per_fold.setter
    def per_fold(self, value):
        self._per_fold = value

    @training_cv_dataset_gen.setter
    def training_cv_dataset_gen(self, value):
        self._training_cv_dataset_gen = value

    @training_cv_figure_gen.setter
    def training_cv_figure_gen(self, value):
        self._training_cv_figure_gen = value

    @training_cv_figure_num.setter
    def training_cv_figure_num(self, value):
        self._training_cv_figure_num = value
