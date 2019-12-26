from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers, callbacks, regularizers, optimizers
from tensorflow.python.keras import backend as K
from collections import namedtuple
import settings
import bcg_net_architecture


def opt_default():
    # this is a function in ttv.py with default settings
    Opt = namedtuple('Opt', ['epochs', 'es_min_delta', 'es_patience',
                             'early_stopping', 'resume', 'overwrite',
                             'ttv_split', 'debug_mode', 'arch',
                             'extra_string'])

    return Opt(
        epochs = 2500,
        es_min_delta = 1e-5,
        es_patience = 25,  # How many times does the validation not increase
        early_stopping = True,
        resume = True,
        overwrite = False,
        ttv_split = [0.7, 0.2, 0.1],  # train/test/validate split
        debug_mode = False,  # more output, also plots
        arch = bcg_net_architecture.arch0001,
        extra_string = ''
    )


def train(d_features, opt):
    # See bcg_net.py for inspiration.  The extra thing to consider is that
    # it would be nice to be able to switch between CNN and RNN which require
    # fundamentally different data restructuring {- maybe - this could go in
    # the architecture but probably there is some reason I have not thought
    # about it, for why this doesn’t work }. I have example code for this,
    # but may not be in bcg_net.

    # for each package of X, y, opt in d_features, let’s train and test!
        # load train, test and validation features, also opt_feature_extract
        # do stuff

         # arch = opt.arch.create_arch(n_output, opt_feature_extract)
         # arch_name = opt.arch.get_name()
         #
         # f_arch = ?  # again some name and hash that depends on the opt. It should probably be in a folder that has the same name as the d_features folder (i.e. what I called something_unique_based_on_opt in generate_ws_features) but with a file that is unique based on the arch.

        # if f_arch exists, we load it, if it is finished training that’s it
        # if it hasn’t finished training, we finish training
        # if it doesn’t exist we start training

    return


def predict():
    return


def clean():
    return


if __name__ == '__main__':
    """ used for debugging """


