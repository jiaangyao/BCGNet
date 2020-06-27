import os
from pathlib import Path
from dataset import DefaultDataset
from models import update_init
from session import data_generator

# TODO: finish all documentation


class DefaultSession:
    # TODO: figure out d_data from cfg
    def __init__(self, d_data, d_model, d_output, str_sub, vec_idx_run, cv_mode=False, cfg=None):
        self.cfg = cfg

        self.d_data = d_data
        self.d_model = d_model
        self.d_output = d_output
        self.str_sub = str_sub
        self.vec_idx_run = vec_idx_run

        self.cv_mode = cv_mode

        self.training_generator = None
        self.validation_generator = None
        self.test_generator = None

        self.vec_dataset = []

    def load_all_dataset(self):
        # Obtain the absolute path to all dataset for a given subject
        vec_abs_path = DefaultSession._absolute_path_to_data(self.d_data, self.str_sub, self.vec_idx_run)

        # initialize the dataset objects each corresponding to a single run of data
        for abs_path in vec_abs_path:
            # TODO: check the init statement here once dataset implementation is done
            curr_dataset = DefaultDataset(abs_path, new_fs=100)

            curr_dataset.prepare_dataset()
            curr_dataset.split_dataset()

            self.vec_dataset.append(curr_dataset)

    # TODO: finish documentation
    @staticmethod
    def _absolute_path_to_data(d_data, str_sub, vec_idx_run):
        """

        :param d_data:
        :param str_sub:
        :param vec_idx_run:
        :return:
        """

        vec_abs_path = []

        for idx_run in vec_idx_run:
            vec_abs_path.append(d_data / '{}_r0{}.set'.format(str_sub, idx_run))

        return vec_abs_path

    def prepare_training(self):
        if not self.cv_mode:
            pass
        else:
            pass

    @staticmethod
    def _init_model(d_root, str_arch=None):
        # update the init model in the models directory
        d_models = d_root / 'models'
        update_init(d_models)

        if str_arch is None:
            from models import RNNModel

            model = RNNModel()

            return model

        elif (d_models / "{}.py".format(str_arch)).exists() is False:
            from models import RNNModel
            import warnings

            warnings.warn("Specified model not found, initialize default model instead", RuntimeWarning)

            model = RNNModel()

            return model

        elif (d_models / "{}.py".format(str_arch)).exists() is True:
            module = __import__("models")
            class_ = getattr(module, str_arch)

            model = class_()

            return model

            # try:
            #
            #
            # except ImportError:
            #     raise Exception("Issue with importing the given model")
        else:
            raise Exception("Unexpected error")


    # TODO: finish documentation
    @staticmethod
    def _combine_from_runs(vec_dataset):
        """

        :param vec_dataset:
        :return:
        """
        pass

    # TODO: finish documentation
    @staticmethod
    def _combine_from_runs_cv(vec_dataset):
        """


        :param vec_dataset:
        :return:
        """

        pass


if __name__ == '__main__':
    """ used for debugging """

    t1 = DefaultSession._init_model(Path(os.getcwd()).parent, 'gru_arch_000')

    print('nothing')