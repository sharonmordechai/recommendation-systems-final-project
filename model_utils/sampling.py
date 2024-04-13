"""
Module containing functions for negative item sampling.
"""

import numpy as np


def sample_items(num_items, shape, random_state=None):
    """
    Randomly sample a number of items.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, shape, dtype=np.int64)

    return items
