import numpy as np

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def regularization_matrix(layer_sizes):
    return [np.r_[np.zeros((1, layer_sizes[i + 1])), np.ones((layer_sizes[i], layer_sizes[i + 1]))] for i
            in range(len(layer_sizes) - 1)]
