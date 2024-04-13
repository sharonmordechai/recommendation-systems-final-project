import numpy as np
import torch


def gpu(tensor, gpu=False):
    return tensor.cuda() if gpu else tensor


def cpu(tensor):
    return tensor.cpu() if tensor.is_cuda else tensor


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def set_seed(seed, cuda=False):
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)