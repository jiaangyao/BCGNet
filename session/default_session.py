import os
import tensorflow as tf
import numpy as np

from pathlib import Path
from dataset import DefaultDataset
from models import update_init
from utils import temp_seed
from session import DefaultGenerator
from tensorflow.keras import callbacks

# TODO: finish all documentation


class DefaultSession:
    # TODO: figure out d_data from cfg
    def __init__(self, d_root, d_data, d_model, d_output, str_sub, vec_idx_run, str_arch,
                 lr=1e-3, batch_size=1, num_epochs=2500, es_patience=25, es_min_delta=1e-5,
                 cv_mode=False, random_seed=1997, verbose=1, cfg=None):
        self.cfg = cfg

        self.d_root = d_root
        self.d_data = d_data
        self.d_model = d_model
        self.d_output = d_output

        self.str_sub = str_sub
        self.vec_idx_run = vec_idx_run
        self.str_arch = str_arch

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta

        self.cv_mode = cv_mode

        self.random_seed = random_seed
        self.verbose = verbose

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
        vec_abs_path = DefaultSession._absolute_path_to_data(self.d_data, self.str_sub, self.vec_idx_run)

        # initialize the dataset objects each corresponding to a single run of data
        for abs_path in vec_abs_path:
            # TODO: check the init statement here once dataset implementation is done
            curr_dataset = DefaultDataset(abs_path, new_fs=100, random_seed=self.random_seed)

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

    # TODO: clean this code up a little bit...
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

    def predict(self):
        raise NotImplementedError



if __name__ == '__main__':
    """ used for debugging """

    import os
    import settings

    settings.init(Path.home(), Path.home())  # Call only once
    d_root = Path(os.getcwd()).parent
    d_data = Path('/home/jyao/Local/working_eegbcg/proc_full/proc_rs/')

    d_model = None
    d_output = None

    str_sub = 'sub12'
    vec_idx_run = [1, 2, 3, 4, 5]

    s1 = DefaultSession(d_root, d_data, d_model, d_output, str_sub, vec_idx_run, str_arch='gru_arch_001',
                        lr=1e-3, batch_size=1, num_epochs=5, es_patience=25, es_min_delta=1e-5)
    s1.load_all_dataset()
    s1.prepare_training()
    s1.train()

    print('nothing')