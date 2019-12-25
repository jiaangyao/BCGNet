#  Class used for setting hyperparameters of the training
class TrainDefault:

    def __init__(self):
        self._dataset_gen = False  # Whether to generate the dataset during the training and evaluation
        self._early_stopping = True  # Flag for turning on the early stopping feature
        self._epochs = 75
        self._es_min_delta = 0  # Minimal change to be considered as an improvement by Keras
        self._es_patience = 10  # Patience factor for early stopping in Keras
        self._evaluate_model = False  # Evaluate a particular model: save the dataset and optionally generate all the figures
        self._evaluation = None  # The percentage to be used as the evaluation (validation plus training) set
        self._f_arch = None  # Name of the architecture, have to be named in the case of resuming or evaluating
        self._fig_num = 1  # Number of figures to generate
        self._figgen = False  # Whether or not to generate figures while evaluating the model
        self._ga_mc = None  # Using the gradient or the motion dataset
        self._multi_ch = None  # Flag for whether or not you are training on multiple channels
        self._multi_run = None  # Flag for whether or not you are training on multiple runs
        self._multi_sub = None  # Flag for whether or not you are training on multiple subjects
        self._p_arch = None  # path of the architecture (mostly as a place holder)
        self._resume = False  # Whether or not you are resuming the training of a particular model
        self._save_bcg_dataset = False # Whether to save the predicted bcg into a .mat file during evaluation
        self._save_QRS_series = False  # Flag for whether or not to save the output from the middle dense layer
        self._target_ch = None
        self._use_bcg_input = False  # Flag for using bcg in input
        self._use_rs_data = False  # Flag for whether or not you are using the motion data in the training
        self._use_time_encoding = False  # Flag for whether or not you are using the time_encoding in the training
        self._validation = None  # The percentage to be used as the validation set

    @property
    def target_ch(self):
        return self._target_ch

    @property
    def epochs(self):
        return self._epochs

    @property
    def ga_mc(self):
        return self._ga_mc

    @property
    def p_arch(self):
        return self._p_arch

    @property
    def f_arch(self):
        return self._f_arch

    @property
    def es_min_delta(self):
        return self._es_min_delta

    @property
    def es_patience(self):
        return self._es_patience

    @property
    def early_stopping(self):
        return self._early_stopping

    @property
    def resume(self):
        return self._resume

    @property
    def evaluate_model(self):
        return self._evaluate_model

    @property
    def dataset_gen(self):
        return self._dataset_gen

    @property
    def save_bcg_dataset(self):
        return self._save_bcg_dataset

    @property
    def validation(self):
        validation = self._validation
        if validation is not None:
            if validation >= 1: raise Exception('es_validation cannot be greater than 1')
            if validation == 0: validation = None

        return validation

    @property
    def evaluation(self):
        evaluation = self._evaluation
        if evaluation is not None:
            if evaluation >= 1: raise Exception('percentage of evaluation cannot be greater than 1')
            if evaluation == 0: evaluation = None

        return evaluation

    @property
    def figgen(self):
        return self._figgen

    @property
    def fig_num(self):
        return self._fig_num

    @property
    def multi_ch(self):
        return self._multi_ch

    @property
    def multi_run(self):
        return self._multi_run

    @property
    def multi_sub(self):
        return self._multi_sub

    @property
    def use_rs_data(self):
        return self._use_rs_data

    @property
    def use_time_encoding(self):
        return self._use_time_encoding

    @property
    def save_QRS_series(self):
        return self._save_QRS_series

    @property
    def use_bcg_input(self):
        return self._use_bcg_input

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @ga_mc.setter
    def ga_mc(self, value):
        self._ga_mc = value

    @p_arch.setter
    def p_arch(self, value):
        self._p_arch = value

    @f_arch.setter
    def f_arch(self, value):
        self._f_arch = value

    @target_ch.setter
    def target_ch(self, value):
        self._target_ch = value

    @es_min_delta.setter
    def es_min_delta(self, value):
        self._es_min_delta = value

    @es_patience.setter
    def es_patience(self, value):
        self._es_patience = value

    @resume.setter
    def resume(self, value):
        self._resume = value

    @evaluate_model.setter
    def evaluate_model(self, value):
        self._evaluate_model = value

    @dataset_gen.setter
    def dataset_gen(self, value):
        self._dataset_gen = value

    @save_bcg_dataset.setter
    def save_bcg_dataset(self, value):
        self._save_bcg_dataset = value

    @evaluation.setter
    def evaluation(self, value):
        self._evaluation = value

    @validation.setter
    def validation(self, value):
        self._validation = value

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value

    @figgen.setter
    def figgen(self, value):
        self._figgen = value

    @fig_num.setter
    def fig_num(self, value):
        self._fig_num = value

    @multi_ch.setter
    def multi_ch(self, value):
        self._multi_ch = value

    @multi_run.setter
    def multi_run(self, value):
        self._multi_run = value

    @multi_sub.setter
    def multi_sub(self, value):
        self._multi_sub = value

    @use_rs_data.setter
    def use_rs_data(self, value):
        self._use_rs_data = value

    @use_time_encoding.setter
    def use_time_encoding(self, value):
        self._use_time_encoding = value

    @save_QRS_series.setter
    def save_QRS_series(self, value):
        self._save_QRS_series = value

    @use_bcg_input.setter
    def use_bcg_input(self, value):
        self._use_bcg_input = value
