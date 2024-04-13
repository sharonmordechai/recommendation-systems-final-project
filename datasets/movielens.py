"""
Utilities for fetching the Movielens datasets [1]_.

References
----------

.. [1] https://grouplens.org/datasets/movielens/
"""

import os
import h5py

from utils.transport import get_data
from utils.interactions import Interactions

VARIANTS = ('100K', '1M', '10M')
URL_PREFIX = 'https://github.com/sharon12312/recommender-datasets/releases/download'
VERSION = '0.1.0'  # the latest tag


def _get_movielens(dataset, extension='.hdf5'):
    path = get_data('/'.join((URL_PREFIX, VERSION, dataset + extension)), os.path.join('movielens', VERSION), f'movielens_{dataset}{extension}')

    with h5py.File(path, 'r') as data:
        return data['/user_id'][:], data['/item_id'][:], data['/rating'][:], data['/timestamp'][:]


def get_movielens_dataset(variant='100K'):
    """
    Download and return one of the Movielens datasets ('100K', '1M', '10M').
    """

    if variant not in VARIANTS:
        raise ValueError(f'Variant must be one of {VARIANTS}, and got {variant}.')

    url = f'movielens_{variant}'
    return Interactions(*_get_movielens(url))
