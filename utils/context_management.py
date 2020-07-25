import contextlib
import numpy as np
import os
import sys


@ contextlib.contextmanager
def temp_seed(seed):
    """
    This function allows temporary creating a numpy random number generator state and is used to ensure that
    splitting the data can be performed with the same random seed 20194040 while the rest of the script is not affected
    by that random state

    :param seed: Desired random seed to be used
    """

    # Obtain the old random seed
    state = np.random.get_state()

    # Set the np random seed in the current environment to the desired seed number
    np.random.seed(seed)

    try:
        yield
    finally:
        # Reset the seed when the function is not called
        np.random.set_state(state)


@contextlib.contextmanager
def suppress_stdout():
    """
    This function allows temporary suppression of standard out and standard error in the terminal to avoid having too
    many output from MNE package when doing the median absolute deviation based epoch rejection
    """

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
