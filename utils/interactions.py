"""
Classes describing datasets of user-item interactions.
"""

import numpy as np
import scipy.sparse as sp


class Interactions(object):
    """
    Interactions object.
    Contains (at a minimum) pair of user-item interactions.
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max() + 1)
        self.num_items = num_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        self._check()

    def __repr__(self):
        return f'<Interactions dataset ({self.num_users} users x {self.num_items} items x {len(self)} interactions)>'

    def __len__(self):
        return len(self.user_ids)

    def _check(self):
        if self.user_ids.max() >= self.num_users:
            raise ValueError('Maximum user id greater than declared number of users.')
        if self.item_ids.max() >= self.num_items:
            raise ValueError('Maximum item id greater than declared number of items.')

        num_interactions = len(self.user_ids)

        for name, value in (('item IDs', self.item_ids), ('ratings', self.ratings),
                            ('timestamps', self.timestamps), ('weights', self.weights)):
            if value is None:
                continue

            if len(value) != num_interactions:
                raise ValueError(f'Invalid {name} dimensions: length must be equal to number of interactions')

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings if self.ratings is not None else np.ones(len(self))

        return sp.coo_matrix((data, (row, col)), shape=(self.num_users, self.num_items))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()