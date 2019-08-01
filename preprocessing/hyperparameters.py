from neural_network.activation_func import ACTIVATION_DICT
from neural_network.loss_func import LOSS_DICT
from neural_network.optimizer import OPTIMIZER_DICT
from neural_network.regularization import REGULARIZATION_DICT
from preprocessing.enums import TaskType
import numpy as np
import itertools


class HyperParameters():

    def __init__(
            self,
            inner_dimension=None,
            init_weights=None,
            init_bias=None,
            use_bias=None,
            epochs=None,
            batch_size=None,
            activation_function=None,
            first_activation=None,
            task_type=None,
            loss=None,
            optimizer=None,
            gradient_rule=None,
            verbose=None):

        self.params = dict(
            epochs=epochs if epochs is not None else [25],
            batch_size=batch_size if batch_size is not None else [1],
            inner_dimension=inner_dimension if inner_dimension is not None else [[1]],
            init_weights=init_weights if init_weights is not None else [np.asarray([])],
            init_bias=init_bias if init_bias is not None else [np.asarray([])],
            use_bias=use_bias if use_bias is not None else [True],
            activation_function=activation_function if activation_function is not None else [[ACTIVATION_DICT["linear"]]],
            first_activation=first_activation if first_activation is not None else [ACTIVATION_DICT["linear"]],
            task_type=task_type if task_type is not None else [TaskType.regression],
            loss=loss if loss is not None else [LOSS_DICT["mse"]],
            optimizer=optimizer if optimizer is not None else [OPTIMIZER_DICT['SGD']()],
            gradient_rule=gradient_rule if gradient_rule is not None else [None],
            verbose=verbose if verbose is not None else [0])

    @staticmethod
    def extraploate_hyperparameters(hyp, exp):
        # Extrapolates all hyper-parameters name and values
        labels, terms = zip(*hyp.params.items())

        params_list = [dict(zip(labels, term)) for term in itertools.product(*terms)]
        params = [{**(exp), **p} for p in params_list]

        return params
