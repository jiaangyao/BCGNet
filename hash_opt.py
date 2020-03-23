import hashlib
import random
from collections import namedtuple
import numpy as np


def set_seed(ps):
    """
    Sets the seed based on a string (e.g. path)

    :type ps: str
    :param ps: any string to be hashed
    """
    hash_ = gen_hash(ps)
    random.seed(hash_)
    np.random.seed(hash_)  # not tested


def gen_hash(ps):
    """
    Gets the hash of a string (e.g. path)

    :type ps: str
    :param ps: any string to be hashed
    """
    str_hash = str(ps)
    hash_object = hashlib.md5(str_hash.encode('utf-8'))
    hash = int(hash_object.hexdigest(), 16) % 2 ** 32
    return hash


def namedtuple_to_dict(opt, exclusions=None, **kwargs):
    """
    Generate a dictionary of items to be hashed.

    :param opt:
    :param exclusions:
    :param kwargs:
    :return:
    """
    default_exclusions = ['overwrite', 'plot']
    if exclusions:
        exclusions = default_exclusions.append(exclusions)
    else:
        exclusions = default_exclusions

    hashable_dict = opt._asdict()
    hashable_dict.update(kwargs)  # in case you want to add something extra

    # remove overwrite strings and opt_met stuff
    for e in exclusions:
        hashable_dict = {k: v for k, v in hashable_dict.items() if e not in k}
    hashable_dict = {k: v for k, v in hashable_dict.items() if v is not None}

    # force the ordering
    hashable_dict = {k: v for k, v in sorted(hashable_dict.items())}

    return hashable_dict


def dict_to_hash(dict, exclusions=None):
    """

    :param dict:
    :param exclusions:
    :return:
    """
    ip_string = dict_to_string(dict)
    return gen_hash(ip_string)


def dict_to_string(hashable_dict):
    """

    :param hashable_dict:
    :return:
    """
    ip_string = ''
    for k, v in hashable_dict.items():
        ip_string = ip_string + k
        if not isinstance(v, list):
            v = [v]
        for i in v:
            ip_string = ip_string + str(i) + '_'
    return ip_string


if __name__ == "__main__":
    Person = namedtuple('Person', 'name age gender')
    opt = Person(name='John', age=45, gender='male')

    dh = namedtuple_to_dict(opt)
    h = dict_to_hash(dh, exclusions=None)
    # h can then be used in the string specification of saved files (so that if you change something in opt
    # you know that the saved file doesn't match). You can add exclusions (as there are somethings that you want to be
    # able to change in opt, without triggering new file generation).

    # this can also be used to set the seed when processing individual subjects so that we get reliable seeds for each subject:
    subject_hash = gen_hash('subject_name')
    set_seed(subject_hash)
