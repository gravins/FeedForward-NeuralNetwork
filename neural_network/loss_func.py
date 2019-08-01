import numpy as np
from copy import deepcopy


class LossFunction:
    def __init__(self, f, dxf):
        # callable is generic: works with built in functions too
        if not callable(f) or not callable(dxf):
            raise ValueError(type(f), " or ", type(dxf), " is not a function")
        self.f = f
        self.dxf = dxf

    def __repr__(self):
        return str(self.f.__name__)


def least_square_error(y_pred, y_true):
    return sum((y_pred - y_true) ** 2)


def least_square_error_dx(y_pred, y_true):
    return y_true - y_pred


def mean_square_error(y_pred, y_true):
    # Mean square error loss function
    return ((y_true - y_pred) ** 2).mean()


def mean_square_error_dx(y_pred, y_true):
    # Mean square error loss function's derivative
    return (-2 / len(y_pred)) * (y_true - y_pred)


def mean_euclidean_error(y_pred, y_true):
    # Mean Euclidean Error loss function
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)).mean()


def mean_euclidean_error_dx(y_pred, y_true):
    # Mean Euclidean Error loss function's derivative
    euc = np.expand_dims((np.sqrt(np.sum((y_true - y_pred) ** 2, axis=0))), axis=1)
    return -((y_true - y_pred) / (euc + 1e-8))


def accuracy(y_pred, y_true):
    return sum(convert_classification(y_pred, 0.5) == y_true) / len(y_true)


def convert_classification(y_pred, threshold):
    class_output = deepcopy(y_pred)
    class_output[class_output > threshold] = 1
    class_output[class_output < threshold] = 0
    return class_output


LOSS_DICT = {"lse": LossFunction(least_square_error, least_square_error_dx),
             "mse": LossFunction(mean_square_error, mean_square_error_dx),
             "mee": LossFunction(mean_euclidean_error, mean_euclidean_error_dx)}
