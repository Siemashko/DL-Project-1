import numpy as np
from enum import Enum


class WeightInitialization(Enum):
    BASIC = 0
    XAVIER = 1
    HE = 2


def basic_random(layer_sizes, random_seed=34):
    return [np.random.rand(1 + layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]


def xavier(layer_sizes, random_seed=34):
    return [np.r_[
                np.zeros((1, layer_sizes[i + 1])),
                np.random.rand(layer_sizes[i], layer_sizes[i + 1]) - 1 / 2
            ] * 2 * np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            for i in range(len(layer_sizes) - 1)]


def he(layer_sizes, random_seed=34):
    return [np.r_[
                np.zeros((1, layer_sizes[i + 1])),
                np.random.normal(size=(layer_sizes[i], layer_sizes[i + 1])) * np.sqrt(2 / (layer_sizes[i]))
            ]
            for i in range(len(layer_sizes) - 1)]


weight_initialization_methods = {
    WeightInitialization.BASIC: basic_random,
    WeightInitialization.XAVIER: xavier,
    WeightInitialization.HE: he
}
