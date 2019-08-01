import numpy as np
from neural_network.regularization import Regularization, REGULARIZATION_DICT


class SGD:
    """
    Least Square SGD
    """

    def __init__(self, lr=0.1, reg_lambda=0.0, momentum=0.0, decay=0.0, reg_func=REGULARIZATION_DICT["L1"]):
        if lr <= 0:
            raise ValueError("Learning rate value need to be greater than 0")
        if reg_lambda < 0:
            raise ValueError("Regularization lambda value need to be greater or equal than 0")
        if momentum < 0:
            raise ValueError("Momentum value need to be greater or equal than 0")
        if decay < 0:
            raise ValueError("Learning rate Decay value need to be greater than 0")
        if not isinstance(reg_func, Regularization):
            raise ValueError(type(reg_func), " is not Regularization type")
        self.learning_rate = lr
        self.starting_learning_rate = lr
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.prev_grad = None
        self.reg_func = reg_func
        self.lr_decay = decay

    def __repr__(self):
        return str(type(self).__name__)

    def update(self, weights, x_train, y_train, y_pred, batch_size, epoch, loss_func):
        gradient = np.dot(x_train.T, loss_func(y_pred, y_train))
        gradient /= batch_size
        gradient *= self.learning_rate

        # Regularization
        gradient -= self.reg_func.dxf(weights, self.reg_lambda)

        # Momentum
        if self.prev_grad is not None:
            gradient += self.momentum * self.prev_grad

        self.prev_grad = gradient
        weights += gradient

        # Apply learning rate decay
        if self.lr_decay > 0:
            self.learning_rate = self.starting_learning_rate * (1 / float(1 + self.lr_decay * epoch))

        return weights


class Adam:
    """
    Adam
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=0.000000001, reg_lambda=0.0, decay=0.0, reg_func=REGULARIZATION_DICT["L1"]):
        if (beta1 < 0 or beta1 >= 1) or (beta2 < 0 or beta2 >= 1):
            raise ValueError("beta value need to be in range [0,1)")
        if epsilon <= 0:
            raise ValueError("epsilon value need to be greater than 0")
        if lr <= 0:
            raise ValueError("Learning rate value need to be greater than 0")
        if reg_lambda < 0:
            raise ValueError("Regularization lambda value need to be greater or equal than 0")
        if decay < 0:
            raise ValueError("Learning rate Decay value need to be greater than 0")
        if not isinstance(reg_func, Regularization):
            raise ValueError(type(reg_func), " is not Regularization type")

        self.learning_rate = lr
        self.starting_learning_rate = lr
        self.beta_1 = beta1
        self.beta_2 = beta2
        self.e = epsilon
        self.first_moment_vect = None
        self.second_moment_vect = None
        self.reg_lambda = reg_lambda
        self.reg_func = reg_func
        self.lr_decay = decay

    def __repr__(self):
        return str(type(self).__name__)

    def update(self, weights, x_train, y_train, y_pred, batch_size, epoch, loss_func):
        if self.first_moment_vect is None:
            self.first_moment_vect = np.zeros(weights.shape)
            self.second_moment_vect = np.zeros(weights.shape)

        epoch += 1
        gradient = np.dot(-x_train.T, loss_func(y_pred, y_train))
        gradient /= batch_size

        self.first_moment_vect = self.beta_1 * self.first_moment_vect + (1 - self.beta_1) * gradient
        self.second_moment_vect = self.beta_2 * self.second_moment_vect + (1 - self.beta_2) * (np.power(gradient, 2))

        """
        # Non optimized version
        corrected_first_moment = self.first_moment_vect / (1 - (self.beta_1 ** epoch))
        corrected_second_moment = self.second_moment_vect / (1 - (self.beta_2 ** epoch))

        weights -= (self.a * corrected_first_moment) / ((corrected_second_moment ** (1/2)) + self.e)

        """
        # Optimized version
        lr = self.learning_rate * np.sqrt(1 - (np.power(self.beta_2, epoch)))
        weights -= lr * self.first_moment_vect / ((np.power(self.second_moment_vect, (1 / 2))) + self.e)

        # Regularization
        if self.reg_lambda > 0:
            weights += self.reg_func.dxf(weights, self.reg_lambda)

        # Apply learning rate decay
        if self.lr_decay > 0:
            self.learning_rate = self.starting_learning_rate * (1 / float(1 + self.lr_decay * epoch))

        return weights


class AdaMax:
    """
    AdaMax
    """

    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, epsilon=0.000000001, reg_lambda=0.0, decay=0.0, reg_func=REGULARIZATION_DICT["L1"]):
        if (beta1 < 0 or beta1 >= 1) or (beta2 < 0 or beta2 >= 1):
            raise ValueError("beta value need to be in range [0,1)")
        if epsilon <= 0:
            raise ValueError("epsilon value need to be greater than 0")
        if lr <= 0:
            raise ValueError("Learning rate value need to be greater than 0")
        if reg_lambda < 0:
            raise ValueError("Regularization lambda value need to be greater or equal than 0")
        if decay < 0:
            raise ValueError("Learning rate Decay value need to be greater than 0")
        if not isinstance(reg_func, Regularization):
            raise ValueError(type(reg_func), " is not Regularization type")
        self.learning_rate = lr
        self.starting_learning_rate = lr
        self.beta_1 = beta1
        self.beta_2 = beta2
        self.e = epsilon
        self.first_moment_vect = None
        self.exp_weighted_infinity_norm = None
        self.reg_lambda = reg_lambda
        self.reg_func = reg_func
        self.lr_decay = decay

    def __repr__(self):
        return str(type(self).__name__)

    def update(self, weights, x_train, y_train, y_pred, batch_size, epoch, loss_func):
        if self.first_moment_vect is None:
            self.first_moment_vect = np.zeros(weights.shape)
            self.exp_weighted_infinity_norm = np.zeros(weights.shape)
        epoch += 1
        gradient = np.dot(-x_train.T, loss_func(y_pred, y_train))
        gradient /= batch_size

        self.first_moment_vect = self.beta_1 * self.first_moment_vect + (1 - self.beta_1) * gradient
        self.exp_weighted_infinity_norm = np.maximum(self.beta_2 * self.exp_weighted_infinity_norm, np.abs(gradient))
        weights -= (self.learning_rate / (1 - (np.power(self.beta_1, epoch)))) * self.first_moment_vect / self.exp_weighted_infinity_norm

        # Regularization
        if self.reg_lambda > 0:
            weights += self.reg_func.dxf(weights, self.reg_lambda)

        # Apply learning rate decay
        if self.lr_decay > 0:
            self.learning_rate = self.starting_learning_rate * (1 / float(1 + self.lr_decay * epoch))

        return weights
