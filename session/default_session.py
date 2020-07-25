import os
import time
import tensorflow as tf
import numpy as np

from pathlib import Path
from config import get_config
from dataset import DefaultDataset
from models import update_init
from utils import temp_seed
from session import DefaultGenerator
from tensorflow.keras import callbacks

# TODO: finish all documentation


class DefaultSession:
    # TODO: change all str_arch to str_model?
    def __init__(self, str_sub, vec_idx_run, str_arch, random_seed=1997, verbose=2, overwrite=False,
                 cv_mode=False, cfg=None):

        self.str_sub = str_sub
        self.vec_idx_run = vec_idx_run
        self.str_arch = str_arch
        self.cv_mode = cv_mode
        self.random_seed = random_seed
        self.overwrite = overwrite
        self.verbose = verbose
        self.cfg = cfg

        self.d_root = cfg.d_root
        self.d_data = cfg.d_data
        self.d_model = cfg.d_model
        self.d_output = cfg.d_output
        self.d_eval = cfg.d_eval
        self.str_eval = cfg.str_eval

        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.num_epochs = cfg.num_epochs
        self.es_patience = cfg.es_patience
        self.es_min_delta = cfg.es_min_delta

        self.training_generator = None
        self.valid_generator = None
        self.test_generator = None
        self.vec_callbacks = None

        self.vec_dataset = []
        self.session_xs = None
        self.session_ys = None
        self.vec_idx_permute = None
        self.session_model = None

        self.m = None
        self.end_epoch = None

    def load_all_dataset(self):
        # Obtain the absolute path to all dataset for a given subject
        vec_abs_data_path = DefaultSession._absolute_path_to_data(self.d_data, self.str_sub, self.vec_idx_run)
        if self.d_eval is not None:
            vec_abs_eval_path = DefaultSession._absolute_path_to_eval(self.d_eval, self.str_sub, self.vec_idx_run)

        # initialize the dataset objects each corresponding to a single run of data
        for i in range(len(vec_abs_data_path)):
            # TODO: check the init statement here once dataset implementation is done
            idx_run = self.vec_idx_run[i]
            abs_data_path = vec_abs_data_path[i]
            if self.d_eval is not None:
                abs_eval_path = vec_abs_eval_path[i]

                curr_dataset = DefaultDataset(abs_data_path, self.str_sub, idx_run,
                                              d_eval=abs_eval_path, str_eval=self.str_eval,
                                              random_seed=self.random_seed, cfg=self.cfg)
            else:
                curr_dataset = DefaultDataset(abs_data_path, self.str_sub, idx_run,
                                              random_seed=self.random_seed, cfg=self.cfg)

            curr_dataset.prepare_dataset()
            curr_dataset.split_dataset()

            self.vec_dataset.append(curr_dataset)

    # TODO: get rid of the hardcoding of the dataset path convention
    @staticmethod
    def _absolute_path_to_data(d_data, str_sub, vec_idx_run):
        """
        Obtain the absolute path to the input dataset

        :param pathlib.Path d_data: absolute path to the dataset with filename and extension
        :param str str_sub: naming convention of the subject, e.g. 'sub11'
        :param list vec_idx_run: list containing the indices of runs to be used for training the model, e.g. [1, 2, 3]

        :return: a list of pathlib.Path objects holding the absolute path to all the individual runs of data
        """

        vec_abs_path = []

        for idx_run in vec_idx_run:
            vec_abs_path.append(d_data / str_sub / '{}_r0{}_rs.set'.format(str_sub, idx_run))

        return vec_abs_path

    # TODO: get rid of the hardcoding of the dataset path convention
    @staticmethod
    def _absolute_path_to_eval(d_eval, str_sub, vec_idx_run):
        """
        Obtain the absolute path to the evaluation dataset


        :param pathlib.Path d_eval: absolute path to the dataset with filename and extension
        :param str str_sub: naming convention of the subject, e.g. 'sub11'
        :param list vec_idx_run: list containing the indices of runs to be used for training the model, e.g. [1, 2, 3]

        :return: a list of pathlib.Path objects holding the absolute path to all the individual runs of evaluation data
        :rtype: list
        """

        vec_abs_path = []

        for idx_run in vec_idx_run:
            vec_abs_path.append(d_eval / str_sub / '{}_r0{}_rmbcg.set'.format(str_sub, idx_run))

        return vec_abs_path

    # TODO: clean this code up a little bit...
    # TODO: figure out the dimension of input and output more intelligently
    def prepare_training(self):
        if not self.cv_mode:

            self.session_model = self._init_model(d_root=self.d_root, str_arch=self.str_arch, lr=self.lr)
            self.session_model.init_model()
            self.session_model.compile_model()

            self.session_xs, self.session_ys, self.vec_idx_permute = self._combine_from_runs(self.vec_dataset,
                                                                                             self.random_seed)
            self.training_generator = DefaultGenerator(self.session_xs[0], self.session_ys[0],
                                                       batch_size=self.batch_size, shuffle=True)

            self.valid_generator = DefaultGenerator(self.session_xs[1], self.session_ys[1],
                                                    batch_size=1, shuffle=False)

            self.test_generator = DefaultGenerator(self.session_xs[2], self.session_ys[2],
                                                   batch_size=1, shuffle=False)

            self.vec_callbacks = DefaultSession._get_callback(self.es_patience, self.es_min_delta)

        else:
            pass

    @staticmethod
    def _init_model(d_root, str_arch=None, lr=1e-3):
        # update the init model in the models directory
        d_models = d_root / 'models'
        update_init(d_models)

        if str_arch is None:
            from models import RNNModel

            model = RNNModel(lr=lr)

            return model

        elif (d_models / "{}.py".format(str_arch)).exists() is False:
            from models import RNNModel
            import warnings

            warnings.warn("Specified model not found, initialize default model instead", RuntimeWarning)

            model = RNNModel(lr=lr)

            return model

        elif (d_models / "{}.py".format(str_arch)).exists() is True:
            try:
                import importlib

                module = __import__("models")
                class_ = getattr(module, str_arch)

                model = class_(lr=lr)

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
        :param random_seed: the random seed used in the experiment for replicable splitting of the dataset

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

        for curr_dataset in vec_dataset:
            curr_xs = curr_dataset.xs
            curr_ys = curr_dataset.ys

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

    # TODO: finish documentation
    # TODO: finish implementation using the non-CV version of the script
    @staticmethod
    def _combine_from_runs_cv(vec_dataset):
        """


        :param vec_dataset:
        :return:
        """

        raise NotImplementedError

    @staticmethod
    def _get_callback(es_patience=25, es_min_delta=1e-5, verbose=0, **kwargs):
        """
        Setup the keras callbacks that will be used in training


        :param int es_patience: patience parameter for the early stopping
        :param int/float es_min_delta: lower bound for epoch to be considered improvement
        :param int verbose: verbosity
        :param kwargs: additional keyword arguments for keras.callbacks.EarlyStopping
\
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

        if int(tf.__version__[0]) > 1:
            self.m = self.session_model.model.fit(x=self.training_generator, epochs=self.num_epochs,
                                                  verbose=self.verbose, callbacks=self.vec_callbacks,
                                                  validation_data=self.valid_generator)

        else:
            self.m = self.session_model.model.fit_generator(generator=self.training_generator, epochs=self.num_epochs,
                                                            verbose=self.verbose, callbacks=self.vec_callbacks,
                                                            validation_data=self.valid_generator)

        self.end_epoch = len(self.m.epoch)

    def clean(self):
        """
        Clean all the dataset using the trained model
        """

        for i in range(len(self.vec_dataset)):
            curr_dataset = self.vec_dataset[i]

            curr_dataset.clean_dataset(self.session_model.model, self.vec_callbacks)

    # TODO: fix the print command later...
    def evaluate(self, mode='test'):
        """
        Evaluate the performance of the model on all dataset

        :param str mode: either 'train', 'valid' or 'test', indicating which set to extract RMS value and
        power ratio from
        """

        print("\n#############################################")
        print("#                  Results                  #")
        print("#############################################\n")

        for i in range(len(self.vec_dataset)):
            curr_dataset = self.vec_dataset[i]

            curr_dataset.evaluate_dataset(mode=mode)

    # TODO: inherit the naming pattern from cfg
    def save_model(self):
        """
        Save the trained model
        """

        f_model = "{}_{}".format(self.str_arch, time.strftime("%Y%m%d_%H%M%S"))
        p_model = self.d_model / self.str_arch / self.str_sub / f_model
        p_model.mkdir(parents=True, exist_ok=True)

        self.session_model.save_model_weights(p_model, f_model, overwrite=self.overwrite)

    # TODO: inherit the naming pattern from cfg
    def save_data(self):
        """
        Save the cleaned time series
        """

        p_output = self.d_output / self.str_sub
        for i in range(len(self.vec_dataset)):
            curr_dataset = self.vec_dataset[i]
            f_output = "{}_r0{}_bcgnet.mat".format(self.str_sub, self.vec_idx_run[i])

            curr_dataset.save_data(p_output, f_output, self.overwrite)

    # TODO: think about whether this is needed at all
    def save_log(self):
        """
        Save the output

        :return:
        """


if __name__ == '__main__':
    """ used for debugging """
