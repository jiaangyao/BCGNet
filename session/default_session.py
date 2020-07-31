import time
import tensorflow as tf
import numpy as np

from dataset import Dataset
from models import update_init
from utils import temp_seed
from session import DefaultGenerator
from tensorflow.keras import callbacks


class Session:
    def __init__(self, str_sub, vec_idx_run, str_arch, random_seed=1997, verbose=2, overwrite=False,
                 cv_mode=False, num_fold=None, cfg=None):
        """
        Each session consists a single block of training, where a model (in normal mode) or multiple models of the
        same type (in cross validation mode) are trained on data generated from various runs from the same subject

        :param str str_sub: name of the subject, e.g. sub11
        :param list vec_idx_run: list containing the indices of all run on which the model is to be trained,
            e.g. [1, 4, 5]
        :param str str_arch: type of the model the user wishes to use. If not provided then default_rnn_model is used
            by default. Else if user wishes to use custom model, this string should be the same as the name of
            the corresponding .py file in 'models' directory
        :param int random_seed: numpy random_seed used during the random splitting of entire dataset into
            training, validation and test sets is always the same, useful for model selection and evaluation
        :param int verbose: verbose sets the verbosity of Keras during model training,
            0=silent, 1=progress bar, 2=one line per epoch
        :param bool overwrite: whether or not to overwrite existing cleaned data
        :param bool cv_mode: whether or not to use the cross validation mode. If num_fold argument is provided when
            cv_mode is enabled, percentage of test set and validation set data will be set to 1/num_fold and
            remaining data will be the training set. If num_fold argument is not provided
            when cross validation mode is enabled, then the number of folds defaults to 1/per_test where
            per_test is the percentage of test set defined in the yaml file
        :param int num_fold: (Optional) the number of cross validation folds to be used, only relevant
            if cross validation mode is enabled.
        :param cfg: configuration struct that is loaded from the yaml file
        """

        self.str_sub = str_sub
        self.vec_idx_run = vec_idx_run
        self.str_arch = str_arch
        self.cv_mode = cv_mode
        self.random_seed = random_seed
        self.overwrite = overwrite
        self.verbose = verbose
        self.cfg = cfg

        if self.cv_mode and num_fold is None:
            self.num_fold = int(np.round(1 / cfg.per_test))
        elif self.cv_mode and num_fold is not None:
            self.num_fold = num_fold

            self.cfg.per_test = 1 / num_fold
            self.cfg.per_valid = 1 / num_fold
            self.cfg.per_training = 1 - self.cfg.per_test - self.cfg.per_valid
        else:
            self.num_fold = num_fold

        self.d_root = self.cfg.d_root
        self.d_data = self.cfg.d_data
        self.d_model = self.cfg.d_model
        self.d_output = self.cfg.d_output
        self.d_eval = self.cfg.d_eval
        self.str_eval = self.cfg.str_eval
        self.input_file_naming_format = self.cfg.input_file_naming_format
        self.eval_file_naming_format = self.cfg.eval_file_naming_format
        self.output_file_naming_format = self.cfg.output_file_naming_format
        self.cv_output_file_naming_format = self.cfg.cv_output_file_naming_format

        self.batch_size = self.cfg.batch_size
        self.lr = self.cfg.lr
        self.num_epochs = self.cfg.num_epochs
        self.es_patience = self.cfg.es_patience
        self.es_min_delta = self.cfg.es_min_delta

        self.vec_dataset = []
        if cv_mode:
            self.vec_session_xs = None
            self.vec_session_ys = None
            self.mat_idx_permute = None
            self.vec_session_model = None

            self.vec_training_generator = None
            self.vec_valid_generator = None
            self.vec_test_generator = None

            self.vec_m = None
            self.vec_end_epoch = None
        else:
            self.session_xs = None
            self.session_ys = None
            self.vec_idx_permute = None
            self.session_model = None

            self.training_generator = None
            self.valid_generator = None
            self.test_generator = None

            self.m = None
            self.end_epoch = None
        self.vec_callbacks = None

    def load_all_dataset(self):
        """
        Load all requested runs of data from the given subject
        """

        # Obtain the absolute path to all dataset for a given subject
        vec_abs_data_path = Session._absolute_path_to_data(self.d_data, self.str_sub, self.vec_idx_run,
                                                           self.input_file_naming_format)
        if self.d_eval is not None:
            vec_abs_eval_path = Session._absolute_path_to_eval(self.d_eval, self.str_sub, self.vec_idx_run,
                                                               self.eval_file_naming_format)

        # initialize the dataset objects each corresponding to a single run of data
        for i in range(len(vec_abs_data_path)):
            idx_run = self.vec_idx_run[i]
            abs_data_path = vec_abs_data_path[i]
            if self.d_eval is not None:
                abs_eval_path = vec_abs_eval_path[i]

                curr_dataset = Dataset(abs_data_path, self.str_sub, idx_run, d_eval=abs_eval_path,
                                       str_eval=self.str_eval, random_seed=self.random_seed,
                                       cv_mode=self.cv_mode, num_fold=self.num_fold, cfg=self.cfg)
            else:
                curr_dataset = Dataset(abs_data_path, self.str_sub, idx_run, random_seed=self.random_seed,
                                       cv_mode=self.cv_mode, num_fold=self.num_fold, cfg=self.cfg)

            curr_dataset.prepare_dataset()
            curr_dataset.split_dataset()

            self.vec_dataset.append(curr_dataset)

    @staticmethod
    def _absolute_path_to_data(d_data, str_sub, vec_idx_run, input_file_naming_format):
        """
        Obtain the absolute path to the input dataset

        :param pathlib.Path d_data: absolute path to the dataset with filename and extension
        :param str str_sub: naming convention of the subject, e.g. 'sub11'
        :param list vec_idx_run: list containing the indices of runs to be used for training the model, e.g. [1, 2, 3]
        :param str input_file_naming_format: naming convention of the input files defined in the config yaml files

        :return: a list of pathlib.Path objects holding the absolute path to all the individual runs of data
        """

        vec_abs_path = []

        for idx_run in vec_idx_run:
            vec_abs_path.append(d_data / str_sub / (input_file_naming_format.format(str_sub, idx_run) + '.set'))

        return vec_abs_path

    @staticmethod
    def _absolute_path_to_eval(d_eval, str_sub, vec_idx_run, eval_file_naming_format):
        """
        Obtain the absolute path to the evaluation dataset


        :param pathlib.Path d_eval: absolute path to the dataset with filename and extension
        :param str str_sub: naming convention of the subject, e.g. 'sub11'
        :param list vec_idx_run: list containing the indices of runs to be used for training the model, e.g. [1, 2, 3]
        :param str eval_file_naming_format: naming convention of the evaluation files defined in the config yaml files

        :return: a list of pathlib.Path objects holding the absolute path to all the individual runs of evaluation data
        :rtype: list
        """

        vec_abs_path = []

        for idx_run in vec_idx_run:
            vec_abs_path.append(d_eval / str_sub / (eval_file_naming_format.format(str_sub, idx_run) + '.set'))

        return vec_abs_path

    def prepare_training(self):
        """
        Initializes the tensorflow model and obtains the training, validation and test sets of data
        """

        # check if all datasets have the same number of input and EEG channels
        vec_n_input = []
        vec_n_output = []
        for dataset in self.vec_dataset:
            vec_n_input.append(dataset.n_input)
            vec_n_output.append(dataset.n_output)

        if not all(x == vec_n_input[0] for x in vec_n_input):
            raise RuntimeError("Input datasets don't have same number of input channels")

        if not all(x == vec_n_output[0] for x in vec_n_output):
            raise RuntimeError("Input datasets don't have same number of EEG channels")

        session_n_input = vec_n_input[0]
        session_n_output = vec_n_output[0]

        if not self.cv_mode:

            # initialize the model
            self.session_model = self._init_model(n_input=session_n_input, n_output=session_n_output,
                                                  d_root=self.d_root, str_arch=self.str_arch, lr=self.lr)
            self.session_model.init_model()
            self.session_model.compile_model()

            # obtain the training, validation and test sets and initialize the generators
            self.session_xs, self.session_ys, self.vec_idx_permute = self._combine_from_runs(self.vec_dataset,
                                                                                             self.random_seed)
            self.training_generator = DefaultGenerator(self.session_xs[0], self.session_ys[0],
                                                       batch_size=self.batch_size, shuffle=True)

            self.valid_generator = DefaultGenerator(self.session_xs[1], self.session_ys[1],
                                                    batch_size=1, shuffle=False)

            self.test_generator = DefaultGenerator(self.session_xs[2], self.session_ys[2],
                                                   batch_size=1, shuffle=False)

            # initialize the list of callbacks during training
            self.vec_callbacks = Session._get_callback(self.es_patience, self.es_min_delta)

        else:
            # initialize the list of models
            vec_session_model = []
            for i in range(self.num_fold):
                fold_model = self._init_model(n_input=session_n_input, n_output=session_n_output,
                                              d_root=self.d_root, str_arch=self.str_arch, lr=self.lr)
                fold_model.init_model()
                fold_model.compile_model()

                vec_session_model.append(fold_model)
            self.vec_session_model = vec_session_model

            # initialize the list of training, validation and test sets and initialize the generators
            self.vec_session_xs, self.vec_session_ys, self.mat_idx_permute = \
                Session._combine_from_runs_cv(self.vec_dataset, self.num_fold, self.random_seed)

            vec_training_generator = []
            vec_valid_generator = []
            vec_test_generator = []
            for i in range(self.num_fold):
                training_generator = DefaultGenerator(self.vec_session_xs[i][0], self.vec_session_ys[i][0],
                                                      batch_size=self.batch_size, shuffle=True)

                valid_generator = DefaultGenerator(self.vec_session_xs[i][1], self.vec_session_ys[i][1],
                                                   batch_size=1, shuffle=False)

                test_generator = DefaultGenerator(self.vec_session_xs[i][2], self.vec_session_ys[i][2],
                                                  batch_size=1, shuffle=False)

                vec_training_generator.append(training_generator)
                vec_valid_generator.append(valid_generator)
                vec_test_generator.append(test_generator)
            self.vec_training_generator = vec_training_generator
            self.vec_valid_generator = vec_valid_generator
            self.vec_test_generator = vec_test_generator

            # initialize the list of callbacks during training
            self.vec_callbacks = Session._get_callback(self.es_patience, self.es_min_delta)

    @staticmethod
    def _init_model(n_input, n_output, d_root, str_arch=None, lr=1e-3):
        """
        Initializes the neural network model that will be trained

        :param int n_input: input dimension to the neural network, should be 1 (number of ECG channel)
        :param int n_output: output dimension to the neural network, should be the number (number of EEG channel)
        :param pathlib.Path d_root: absolute path to the root directory of the package
        :param str str_arch: the type of the model that's to be trained, if not provided then defaults to the
            default_rnn_model, else should be a string that's of the same name as the corresponding .py file under
            models directory
        :param float lr: learning rate for training the model
        """
        # update the init model in the models directory
        d_models = d_root / 'models'
        update_init(d_models)

        if str_arch is None:
            from models import RNNModel

            model = RNNModel(n_input=n_input, n_output=n_output, lr=lr)

            return model

        elif str_arch == 'default_rnn_model':
            from models import RNNModel

            model = RNNModel(n_input=n_input, n_output=n_output, lr=lr)

            return model

        elif (d_models / "{}.py".format(str_arch)).exists() is False:
            from models import RNNModel
            import warnings

            warnings.warn("Specified model not found, initialize default model instead", RuntimeWarning)

            model = RNNModel(n_input=n_input, n_output=n_output, lr=lr)

            return model

        elif (d_models / "{}.py".format(str_arch)).exists() is True:
            try:
                import importlib

                module = __import__("models")
                class_ = getattr(module, str_arch)

                model = class_(n_input=n_input, n_output=n_output, lr=lr)

                return model

            except ImportError:
                raise Exception("Issue with importing the given model")
        else:
            raise Exception("Unexpected error")

    @staticmethod
    def _combine_from_runs(vec_dataset, random_seed):
        """
        combine the training, validation and test sets from all dataset each holding a single run of data

        :param list vec_dataset: list of dataset objects where each object holds raw and precessed data from
            a single run
        :param int random_seed: the random seed used in the experiment for replicable splitting of the dataset

        :return: a tuple (session_xs, session_ys, vec_idx_permute), where session_xs is a list containing all the
            ECG data from all runs in the form of [x_train, x_validation, x_test] and each has
            shape (epoch, channel, data), where session_ys is a list containing all the
            corrupted EEG data in the form of [y_train, y_validation, y_test], and each has shape
            (epoch, channel, data) and vec_idx_permute containing the order of the random permutation when
            combining the epochs from each individual dataset
        """

        vec_xs = []
        vec_ys = []

        for curr_dataset in vec_dataset:
            vec_xs.append(curr_dataset.xs)
            vec_ys.append(curr_dataset.ys)

        session_xs, session_ys, vec_idx_permute = Session._concatenate_all_sets(vec_xs, vec_ys, random_seed)

        return session_xs, session_ys, vec_idx_permute

    @staticmethod
    def _concatenate_all_sets(vec_xs, vec_ys, random_seed):
        """
        Combines the training, validation and test sets from all individual runs

        :param list vec_xs: a list of individual xs from each run of data, where each xs is the list containing all
            the ECG data in the form of [x_train, x_validation, x_test] and each has shape (epoch, channel, data)
        :param list vec_ys: a list of individual ys from each run of data, where each ys is a list containing all the
            corrupted EEG data in the form of [y_train, y_validation, y_test], and each has shape
            (epoch, channel, data)
        :param int random_seed: the random seed used in the experiment for replicable splitting of the dataset

        :return: a tuple (session_xs, session_ys, vec_idx_permute), where session_xs is a list containing all the
            ECG data from all runs in the form of [x_train, x_validation, x_test] and each has
            shape (epoch, channel, data), where session_ys is a list containing all the
            corrupted EEG data in the form of [y_train, y_validation, y_test], and each has shape
            (epoch, channel, data) and vec_idx_permute containing the order of the random permutation when
            combining the epochs from each individual dataset
        """

        vec_session_x_training = []
        vec_session_x_valid = []
        vec_session_x_test = []

        vec_session_y_training = []
        vec_session_y_valid = []
        vec_session_y_test = []

        for i in range(len(vec_xs)):
            curr_xs = vec_xs[i]
            curr_ys = vec_ys[i]

            vec_session_x_training.append(curr_xs[0])
            vec_session_x_valid.append(curr_xs[1])
            vec_session_x_test.append(curr_xs[2])

            vec_session_y_training.append(curr_ys[0])
            vec_session_y_valid.append(curr_ys[1])
            vec_session_y_test.append(curr_ys[2])

        # concatenate the numpy arrays
        session_x_training = np.concatenate(vec_session_x_training, axis=0)
        session_x_valid = np.concatenate(vec_session_x_valid, axis=0)
        session_x_test = np.concatenate(vec_session_x_test, axis=0)

        session_y_training = np.concatenate(vec_session_y_training, axis=0)
        session_y_valid = np.concatenate(vec_session_y_valid, axis=0)
        session_y_test = np.concatenate(vec_session_y_test, axis=0)

        # permute the arrays again
        with temp_seed(random_seed):
            vec_idx_permute_training = np.random.permutation(session_x_training.shape[0])
            vec_idx_permute_valid = np.random.permutation(session_x_valid.shape[0])
            vec_idx_permute_test = np.random.permutation(session_x_test.shape[0])

        session_xs = [session_x_training[vec_idx_permute_training, :],
                      session_x_valid[vec_idx_permute_valid, :],
                      session_x_test[vec_idx_permute_test, :]]

        session_ys = [session_y_training[vec_idx_permute_training, :, :],
                      session_y_valid[vec_idx_permute_valid, :, :],
                      session_y_test[vec_idx_permute_test, :, :]]

        vec_idx_permute = [vec_idx_permute_training, vec_idx_permute_valid, vec_idx_permute_test]

        return session_xs, session_ys, vec_idx_permute

    @staticmethod
    def _combine_from_runs_cv(vec_dataset, num_fold, random_seed):
        """
        combine the training, validation and test sets from all dataset each holding a single run of data
        for the cross validation mode

        :param list vec_dataset: list of dataset objects where each object holds raw and precessed data from
            a single run
        :param int num_fold: the number of cross validation folds used
        :param int random_seed: the random seed used in the experiment for replicable splitting of the dataset

        :return: a tuple (vec_session_xs, vec_session_ys, mat_idx_permute), where vec_session_xs is a list
            containing all the ECG data from all folds [fold0, fold1, ...], and each fold of data is in the form
            [x_train, x_validation, x_test] and each has shape (epoch, channel, data), where vec_session_ys is a
            list containing all the corrupted EEG data from all folds [fold0, fold1, ...] and each fold is
            in the form of [y_train, y_validation, y_test], and each has shape (epoch, channel, data) and
            mat_idx_permute containing the order of the random permutation in each fold in the form
            [fold0, fold1, ...] and each fold contains the indices for the random permutation of epochs when
            combining the epochs from each individual dataset
        """

        vec_session_xs = []
        vec_session_ys = []
        mat_idx_permute = []

        # loop through all the folds
        for i in range(num_fold):
            vec_xs = []
            vec_ys = []

            # loop through all runs in each fold
            for curr_dataset in vec_dataset:
                vec_xs.append(curr_dataset.vec_xs[i])
                vec_ys.append(curr_dataset.vec_ys[i])

            session_xs, session_ys, vec_idx_permute = Session._concatenate_all_sets(vec_xs, vec_ys, random_seed)

            vec_session_xs.append(session_xs)
            vec_session_ys.append(session_ys)
            mat_idx_permute.append(vec_idx_permute)

        return vec_session_xs, vec_session_ys, mat_idx_permute

    @staticmethod
    def _get_callback(es_patience=25, es_min_delta=1e-5, verbose=0, **kwargs):
        """
        Setup the keras callbacks that will be used in training


        :param int es_patience: patience parameter for the early stopping
        :param int/float es_min_delta: lower bound for epoch to be considered improvement
        :param int verbose: verbosity
        :param kwargs: additional keyword arguments for keras.callbacks.EarlyStopping

        :return: a list containing the early stopping object defined with input parameters
        """

        # Generate the early stopping objects based on specfications from the options file
        vec_callbacks = [callbacks.EarlyStopping(monitor='val_loss', min_delta=es_min_delta,
                                                 patience=es_patience, verbose=verbose, mode='min',
                                                 restore_best_weights=True, **kwargs)]

        return vec_callbacks

    def train(self):
        """
        initiate training
        """
        if not self.cv_mode:
            if int(tf.__version__[0]) > 1:
                self.m = self.session_model.model.fit(x=self.training_generator, epochs=self.num_epochs,
                                                      verbose=self.verbose, callbacks=self.vec_callbacks,
                                                      validation_data=self.valid_generator)

            else:
                self.m = self.session_model.model.fit_generator(generator=self.training_generator,
                                                                epochs=self.num_epochs,
                                                                verbose=self.verbose, callbacks=self.vec_callbacks,
                                                                validation_data=self.valid_generator)

            self.end_epoch = len(self.m.epoch)
        else:
            vec_m = []
            vec_end_epoch = []
            for i in range(self.num_fold):
                print('\n\nTraining the {}-th fold'.format(i + 1))

                if int(tf.__version__[0]) > 1:
                    m = self.vec_session_model[i].model.fit(x=self.vec_training_generator[i], epochs=self.num_epochs,
                                                            verbose=self.verbose, callbacks=self.vec_callbacks,
                                                            validation_data=self.vec_valid_generator[i])

                else:
                    m = self.vec_session_model[i].model.fit_generator(generator=self.vec_training_generator[i],
                                                                      epochs=self.num_epochs, verbose=self.verbose,
                                                                      callbacks=self.vec_callbacks,
                                                                      validation_data=self.vec_valid_generator[i])
                vec_m.append(m)
                vec_end_epoch.append(len(m.epoch))
            self.vec_m = vec_m
            self.vec_end_epoch = vec_end_epoch

    def clean(self):
        """
        Clean all the dataset using the trained model
        """
        if not self.cv_mode:
            for i in range(len(self.vec_dataset)):
                curr_dataset = self.vec_dataset[i]

                curr_dataset.clean_dataset(self.session_model.model, self.vec_callbacks)
        else:
            vec_models = []
            for i in range(self.num_fold):
                vec_models.append(self.vec_session_model[i].model)

            for i in range(len(self.vec_dataset)):
                curr_dataset = self.vec_dataset[i]

                curr_dataset.clean_dataset_cv(vec_models, self.vec_callbacks)

    def evaluate(self, mode='test'):
        """
        Evaluate the performance of the model on all dataset

        :param str mode: either 'train', 'valid' or 'test', indicating which set to extract RMS value and
        power ratio from
        """

        print('\n'.ljust(51, '#'))
        print("#" + 'RESULT'.center(48) + "#")
        print('\n'.rjust(51, '#'))

        if not self.cv_mode:
            for i in range(len(self.vec_dataset)):
                curr_dataset = self.vec_dataset[i]

                curr_dataset.evaluate_dataset(mode=mode)

        else:
            for idx_fold in range(self.num_fold):
                print("Cross Validation Fold {}/{}\n\n".format(idx_fold + 1, self.num_fold))
                for idx_run in range(len(self.vec_dataset)):
                    curr_dataset = self.vec_dataset[idx_run]

                    curr_dataset.evaluate_dataset_cv(idx_fold, mode=mode)

                print('\n'.rjust(51, '='))

    def save_model(self):
        """
        Save the trained model
        """

        f_model = "{}_{}".format(self.str_arch, time.strftime("%Y%m%d_%H%M%S"))
        p_model = self.d_model / self.str_arch / self.str_sub / f_model
        p_model.mkdir(parents=True, exist_ok=True)

        if not self.cv_mode:
            self.session_model.save_model_weights(p_model, f_model, overwrite=self.overwrite)
        else:
            for idx_fold in range(self.num_fold):
                f_model_fold = "{}_fold{}".format(f_model, idx_fold)
                self.vec_session_model[idx_fold].save_model_weights(p_model, f_model_fold, overwrite=self.overwrite)

    def save_data(self):
        """
        Save the cleaned time series in MATLAB .mat files
        """

        p_output = self.d_output / self.str_sub
        if not self.cv_mode:
            for i in range(len(self.vec_dataset)):
                curr_dataset = self.vec_dataset[i]
                f_output = self.output_file_naming_format.format(self.str_sub, self.vec_idx_run[i]) + '.mat'

                curr_dataset.save_data(p_output, f_output, self.overwrite)
        else:
            for idx_fold in range(self.num_fold):
                for idx_run in range(len(self.vec_dataset)):
                    curr_dataset = self.vec_dataset[idx_run]
                    f_output = self.cv_output_file_naming_format.format(self.str_sub,
                                                                        self.vec_idx_run[idx_run], idx_fold) + ".mat"

                    curr_dataset.save_data(p_output, f_output, self.overwrite, idx_fold=idx_fold)

    def save_dataset(self):
        """
        Save the cleaned dataset in Neuromag .fif files
        """

        p_output = self.d_output / self.str_sub
        if not self.cv_mode:
            for i in range(len(self.vec_dataset)):
                curr_dataset = self.vec_dataset[i]
                f_output = self.output_file_naming_format.format(self.str_sub, self.vec_idx_run[i]) + '.mat'

                curr_dataset.save_dataset(p_output, f_output, self.overwrite)
        else:
            for idx_fold in range(self.num_fold):
                for idx_run in range(len(self.vec_dataset)):
                    curr_dataset = self.vec_dataset[idx_run]
                    f_output = self.cv_output_file_naming_format.format(self.str_sub,
                                                                        self.vec_idx_run[idx_run], idx_fold) + ".mat"

                    curr_dataset.save_dataset(p_output, f_output, self.overwrite, idx_fold=idx_fold)


if __name__ == '__main__':
    """ used for debugging """
