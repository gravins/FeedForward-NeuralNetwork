import numpy as np


class ActivationFunction:
    def __init__(self, f, dxf):
        # callable is generic: works with built in functions too
        if not callable(f) or not callable(dxf):
            raise ValueError(type(f), " or ", type(dxf), " is not a function")
        self.f = f
        self.dxf = dxf

    def __repr__(self):
        return str(self.f.__name__)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_dx(x):
    return x * (1 - x)


def linear(x):
    return x


def linear_dx(x):
    return np.ones_like(x)


def tanh(x):
    return np.tanh(x)


def tanh_dx(x):
    return 1 - (np.tanh(x)**2)


def relu(x):
    return np.asarray(list(map(lambda y: max(0., y), x)))


def relu_dx(x):
    return np.asarray(list(map(lambda y: 0. if y <= 0 else 1., x)))


ACTIVATION_DICT = {"linear": ActivationFunction(linear, linear_dx),
                   "sigmoid": ActivationFunction(sigmoid, sigmoid_dx),
                   "tanh": ActivationFunction(tanh, tanh_dx),
                   "relu": ActivationFunction(relu, relu_dx)}
