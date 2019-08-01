import numpy as np
from neural_network.regularization import Regularization, REGULARIZATION_DICT
from copy import deepcopy


class OptimizationAlgorithm:
    """
    General Optimization Algorithm Super-Class
    """

    def __init__(self, lr=0.001, reg_lambda=0.0, decay=0.0, reg_func=REGULARIZATION_DICT["L1"]):
        if lr <= 0:
            raise ValueError("Learning rate value need to be greater than 0")
        if reg_lambda < 0:
            raise ValueError("Regularization lambda value need to be greater or equal than 0")
        if not isinstance(reg_func, Regularization):
            raise ValueError(type(reg_func), " is not Regularization type")
        if decay < 0:
            raise ValueError("Learning rate Decay value need to be greater than 0")

        self.learning_rate = lr
        self.starting_learning_rate = lr
        self.reg_lambda = reg_lambda
        self.reg_func = reg_func
        self.lr_decay = decay

    def update(self, weights, bias, grad, bias_grad, epoch):
        # Apply learning rate decay
        self.update_learning_rate(epoch)

    def update_learning_rate(self, epoch):
        if self.lr_decay:
            self.learning_rate = self.starting_learning_rate * (1 / float(1 + self.lr_decay * epoch))

    def __repr__(self):
        return str(type(self).__name__)


class SGD(OptimizationAlgorithm):
    """
    SGD
    """

    def __init__(self, momentum=0.0, nesterov=False, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

        if momentum < 0:
            raise ValueError("Momentum value need to be greater or equal than 0")

        self.momentum = momentum
        self.nesterov = nesterov
        self.previous_grad = None
        self.previous_grad_bias = None

    def update(self, weights, bias, grad, bias_grad, epoch):

        grad *= -self.learning_rate
        bias_grad *= -self.learning_rate

        # Apply momentum and update previous gradient if necessary
        if self.momentum and self.previous_grad is not None:
            grad += self.momentum * self.previous_grad
            bias_grad += self.momentum * self.previous_grad_bias

        if self.momentum:
            self.previous_grad = deepcopy(grad)
            self.previous_grad_bias = deepcopy(bias_grad)

        weights += grad
        bias += bias_grad

        # Generic update from Super-Class
        super().update(weights, bias, grad, bias_grad, epoch)

        return weights, bias

    def nesterov_grad(self):
        return (self.momentum * self.previous_grad, self.momentum * self.previous_grad_bias) if (self.nesterov and self.previous_grad is not None) else (None, None)


class Adam(OptimizationAlgorithm):
    """
    Adam
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)

        if (beta1 < 0 or beta1 >= 1) or (beta2 < 0 or beta2 >= 1):
            raise ValueError("beta value need to be in range [0,1)")
        if epsilon <= 0:
            raise ValueError("epsilon value need to be greater than 0")

        self.beta_1 = beta1
        self.beta_2 = beta2
        self.e = epsilon
        self.first_moment_vect = None
        self.second_moment_vect = None
        self.bias_first_moment_vect = None
        self.bias_second_moment_vect = None

    def update(self, weights, bias, grad, bias_grad, epoch):
        if self.first_moment_vect is None:
            self.first_moment_vect = deepcopy(np.asarray(weights)) * 0.
            self.second_moment_vect = deepcopy(self.first_moment_vect)

        if self.bias_first_moment_vect is None:
            self.bias_first_moment_vect = deepcopy(np.asarray(bias)) * 0.
            self.bias_second_moment_vect = deepcopy(self.bias_first_moment_vect)

        epoch += 1
        lr = self.learning_rate * np.sqrt(1 - (self.beta_2 ** epoch))

        for layer_index, _ in enumerate(weights):

            self.first_moment_vect[layer_index] = self.beta_1 * self.first_moment_vect[layer_index] + (1 - self.beta_1) * grad[layer_index]
            self.second_moment_vect[layer_index] = self.beta_2 * self.second_moment_vect[layer_index] + (1 - self.beta_2) * (grad[layer_index] ** 2)

            self.bias_first_moment_vect[layer_index] = self.beta_1 * self.bias_first_moment_vect[layer_index] + (1 - self.beta_1) * bias_grad[layer_index]
            self.bias_second_moment_vect[layer_index] = self.beta_2 * self.bias_second_moment_vect[layer_index] + (1 - self.beta_2) * (bias_grad[layer_index] ** 2)

            weights[layer_index] -= lr * self.first_moment_vect[layer_index] / ((self.second_moment_vect[layer_index] ** (1 / 2)) + self.e)
            bias[layer_index] -= lr * self.bias_first_moment_vect[layer_index] / ((self.bias_second_moment_vect[layer_index] ** (1 / 2)) + self.e)

        # Generic update from Super-Class
        super().update(weights, bias, grad, bias_grad, epoch)

        return weights, bias


class AdaMax(OptimizationAlgorithm):
    """
    AdaMax
    """

    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        super(AdaMax, self).__init__(lr=lr, *args, **kwargs)

        if (beta1 < 0 or beta1 >= 1) or (beta2 < 0 or beta2 >= 1):
            raise ValueError("beta value need to be in range [0,1)")

        self.beta_1 = beta1
        self.beta_2 = beta2
        self.e = epsilon
        self.first_moment_vect = None
        self.exp_weighted_infinity_norm = None
        self.bias_first_moment_vect = None
        self.bias_exp_weighted_infinity_norm = None

    def update(self, weights, bias, grad, bias_grad, epoch):
        if self.first_moment_vect is None:
            self.first_moment_vect = deepcopy(np.asarray(weights)) * 0.
            self.exp_weighted_infinity_norm = deepcopy(self.first_moment_vect)

        if self.bias_first_moment_vect is None:
            self.bias_first_moment_vect = deepcopy(np.asarray(bias)) * 0.
            self.bias_exp_weighted_infinity_norm = deepcopy(self.bias_first_moment_vect)

        epoch += 1
        lr = self.learning_rate / (1 - (self.beta_1 ** epoch))

        for layer_index, _ in enumerate(weights):

            self.first_moment_vect[layer_index] = self.beta_1 * self.first_moment_vect[layer_index] + (1 - self.beta_1) * grad[layer_index]
            self.exp_weighted_infinity_norm[layer_index] = np.maximum(self.beta_2 * self.exp_weighted_infinity_norm[layer_index], np.abs(grad[layer_index]))
            self.bias_first_moment_vect[layer_index] = self.beta_1 * self.bias_first_moment_vect[layer_index] + (1 - self.beta_1) * bias_grad[layer_index]
            self.bias_exp_weighted_infinity_norm[layer_index] = np.maximum(self.beta_2 * self.bias_exp_weighted_infinity_norm[layer_index], np.abs(bias_grad[layer_index]))

            weights[layer_index] -= lr * self.first_moment_vect[layer_index] / (self.exp_weighted_infinity_norm[layer_index] + self.e)
            bias[layer_index] -= lr * self.bias_first_moment_vect[layer_index] / (self.bias_exp_weighted_infinity_norm[layer_index] + self.e)

        # Generic update from Super-Class
        super().update(weights, bias, grad, bias_grad, epoch)

        return weights, bias


OPTIMIZER_DICT = {"SGD": SGD,
                  "ADAM": Adam,
                  "ADAMAX": AdaMax}
