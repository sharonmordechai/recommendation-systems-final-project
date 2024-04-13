"""
Embedding layers useful for recommender models.
"""

import numpy as np
import torch
import torch.nn as nn

from hashlib import md5, sha1, sha256
from sklearn.utils import murmurhash3_32
from xxhash import xxh32
import metrohash
from farmhash import FarmHash32WithSeed

SEEDS = [
    179424941,
    179425457,
    179425907,
    179426369,
    179424977,
    179425517,
    179425943,
    179426407,
    179424989,
    179425529,
    179425993,
    179426447,
    179425003,
    179425537,
    179426003,
    179426453,
    179425019,
    179425559,
    179426029,
    179426491,
    179425027,
    179425579,
    179426081,
    179426549,
]
HASH_FUNCTIONS = [
    "MurmurHash",
    "xxHash",
    "MD5",
    "SHA1",
    "SHA256",
    "MetroHash",
    "FarmHash",
]


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values to using a normal variable scaled
    by the inverse of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values to using a normal variable scaled
    by the inverse of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class BloomEmbedding(nn.Module):
    """
    An embedding layer that compresses the number of embedding
    parameters required by using bloom filter-like hashing.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        compression_ratio=0.2,
        num_hash_functions=4,
        padding_idx=0,
        hash_function="MurmurHash",
    ):

        super(BloomEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression_ratio = compression_ratio
        self.compressed_num_embeddings = int(compression_ratio * num_embeddings)
        self.num_hash_functions = num_hash_functions
        self.padding_idx = padding_idx

        if num_hash_functions > len(SEEDS):
            raise ValueError(
                f"Can use at most {len(SEEDS)} hash functions ({num_hash_functions} requested)"
            )
        if hash_function not in HASH_FUNCTIONS:
            raise ValueError(f"Cannot use Hash function: {hash_function}")

        self._masks = SEEDS[: self.num_hash_functions]

        self.embeddings = ScaledEmbedding(
            self.compressed_num_embeddings,
            self.embedding_dim,
            padding_idx=self.padding_idx,
        )

        # Hash cache. We pre-hash all the indices, and then just map the
        # indices to their pre-hashed values as we go through the minibatches.
        self._hashes = None
        self._offsets = None

        # Initialize hash function name
        self._hash_function = hash_function

    def __repr__(self):
        return f"<BloomEmbedding (compression_ratio: {self.compression_ratio}): {self.embeddings}>"

    def _get_hashed_indices(self, original_indices):
        def _hash(x, seed):
            # The vector can be hashed using different hash functions
            result = self._get_hashed_values(x, seed, self._hash_function)
            result[self.padding_idx] = 0

            return result % self.compressed_num_embeddings

        if self._hashes is None:
            indices = np.arange(self.num_embeddings, dtype=np.int32)
            hashes = np.stack(
                [_hash(indices, seed) for seed in self._masks], axis=1
            ).astype(np.int64)
            assert hashes[self.padding_idx].sum() == 0

            self._hashes = torch.from_numpy(hashes)

            if original_indices.is_cuda:
                self._hashes = self._hashes.cuda()

        hashed_indices = torch.index_select(self._hashes, 0, original_indices.squeeze())

        return hashed_indices

    def forward(self, indices):
        """
        Retrieve embeddings corresponding to indices.

        See documentation on PyTorch ``nn.Embedding`` for details.
        """

        if indices.dim() == 2:
            batch_size, seq_size = indices.size()
        else:
            batch_size, seq_size = indices.size(0), 1

        if not indices.is_contiguous():
            indices = indices.contiguous()

        indices = indices.data.view(batch_size * seq_size, 1)
        hashed_indices = self._get_hashed_indices(indices)
        embedding = self.embeddings(hashed_indices)
        embedding = embedding.sum(1)
        embedding = embedding.view(batch_size, seq_size, -1)

        return embedding

    @staticmethod
    def _get_hashed_values(x, seed, hash_function):
        if hash_function == "MurmurHash":
            return murmurhash3_32(x, seed=seed)
        elif hash_function == "xxHash":
            return np.array([xxh32(i, seed=seed).intdigest() for i in x])
        elif hash_function == "MD5":
            return np.array(
                [
                    int.from_bytes(md5(i + np.int32(seed)).digest()[:4], "little")
                    for i in x
                ]
            )
        elif hash_function == "SHA1":
            return np.array(
                [
                    int.from_bytes(sha1(i + np.int32(seed)).digest()[:4], "little")
                    for i in x
                ]
            )
        elif hash_function == "SHA256":
            return np.array(
                [
                    int.from_bytes(sha256(i + np.int32(seed)).digest()[:4], "little")
                    for i in x
                ]
            )
        elif hash_function == "MetroHash":
            return np.array([metrohash.hash64_int(i, seed=seed) for i in x])
        elif hash_function == "FarmHash":
            return np.array([FarmHash32WithSeed(i, seed=seed) for i in x])
        return None
