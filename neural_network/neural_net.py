import numpy as np
import pickle
import itertools
from copy import deepcopy
from neural_network.loss_func import LOSS_DICT, LossFunction, convert_classification
from neural_network.activation_func import ACTIVATION_DICT, ActivationFunction
from plot.plot_graph import plot_graph, plot_contour
from plot.net_plot import DrawNN
from preprocessing.enums import TaskType
from preprocessing.hyperparameters import HyperParameters
from linear_model.error_functions import DataErrorFunction
from math import sqrt

np.set_printoptions(precision=5)
np.random.seed(seed=42)


class NeuralNet:

    def __init__(self, input_dim, optimizer, loss=LOSS_DICT["mse"], task_type=TaskType.regression, gradient_rule=None, name="net"):

        self.input_dim = input_dim
        self.output_dim = None
        self.task_type = task_type
        self.gradient_rule = gradient_rule

        if not isinstance(loss, LossFunction):
            raise TypeError(type(loss), " is not an instance of LossFunction")
        self.loss = loss

        self.states = []
        self.net = []
        self.delta = []
        self.weights = []
        self.bias_states = []
        self.bias = []
        self.activation_function = []
        self.optimizer = optimizer
        self.name = name
        self.saved_weights = None
        self.saved_bias = None

    def add_layer(self, neurons, activation_fun=ACTIVATION_DICT["tanh"], use_bias=False, init_weights=None, bias_weights=None):
        if neurons <= 0:
            raise ValueError("neurons <= 0")

        if not isinstance(activation_fun, ActivationFunction):
            raise TypeError(type(activation_fun), " is not an instance of ActivationFunction")

        self.activation_function.append(activation_fun)

        prev_neurons = self.input_dim
        if self.states:
            prev_neurons = len(self.states[-1])

        self.states.append(np.zeros(shape=(neurons)))
        self.delta.append(np.zeros(shape=(neurons)))

        if init_weights is not None:
            self.weights.append(init_weights)
        else:
            self.weights.append(np.random.randn(prev_neurons, neurons) * sqrt(2 / prev_neurons))

        if bias_weights is not None:
            self.bias.append(bias_weights)
        else:
            self.bias.append(np.random.randn(neurons) * sqrt(2 / prev_neurons))

        if use_bias:
            self.bias_states.append(1)
        else:
            self.bias_states.append(0)

    def initialize(self, first_layer_activation=ACTIVATION_DICT["linear"]):

        if not self.states:
            raise ValueError("You must append at least one inner layer")

        if not isinstance(first_layer_activation, ActivationFunction):
            raise TypeError(type(first_layer_activation), " is not an instance of ActivationFunction")

        # Initialize matrices that will be used by the network
        self.output_dim = len(self.states[-1])

        self.activation_function.insert(0, first_layer_activation)
        self.states.insert(0, np.zeros(shape=(self.input_dim)))
        self.delta.insert(0, np.zeros(shape=(self.input_dim)))

        self.weights = np.asarray(self.weights)
        self.bias = np.asarray(self.bias)
        self.bias_states = np.asarray(self.bias_states)
        self.states = np.asarray(self.states)
        self.delta = np.asarray(self.delta)
        self.net = [[] for _ in self.states]

    def forward_propagate(self, data):

        self.net[0] = np.asarray(data)
        self.states[0] = self.activation_function[0].f(self.net[0])

        for index, layer_weight in enumerate(self.weights):
            net_bias = np.multiply(self.bias_states[index], self.bias[index].transpose())
            net_weights = np.asarray(np.dot(layer_weight.T, self.states[index]))
            self.net[index + 1] = np.sum([net_weights, net_bias], axis=0)
            self.states[index + 1] = self.activation_function[index + 1].f(self.net[index + 1])

        return self.states

    def back_propagate(self, data_target):

        if len(data_target) != len(self.states[-1]):
            raise ValueError("Target and NN's output dimension have different size")

        data_target = np.asarray(data_target)
        delta = self.delta * 0.

        for layer in range(len(self.delta) - 1, -1, -1):
            net_dx = self.activation_function[layer].dxf(self.states[layer])

            if layer == len(self.delta) - 1:
                # Compute error in last layer
                net_output = self.states[-1]
                delta[layer] = self.loss.dxf(net_output, data_target) * net_dx

            else:
                # Compute error on the hidden layer
                error = np.dot(self.weights[layer], delta[layer + 1])
                delta[layer] = net_dx * error

        return delta

    def compute_gradient(self, states):

        if self.gradient_rule:
            return self.gradient_rule(self.weights, self.bias)

        grad = self.weights * 0.
        bias_grad = self.bias * 0.
        reg_lambda = self.optimizer.reg_lambda
        reg_func = self.optimizer.reg_func

        for layer_index, _ in enumerate(self.weights):
            # Compute gradient
            grad[layer_index] = np.outer(states[layer_index], self.delta[layer_index + 1])
            bias_grad[layer_index] = self.bias_states[layer_index] * self.delta[layer_index + 1]

            # Regularization
            if reg_lambda > 0:
                grad[layer_index] += reg_func.dxf(self.weights[layer_index], reg_lambda)
                bias_grad[layer_index] += reg_func.dxf(self.bias[layer_index], reg_lambda)

        return grad, bias_grad

    def fit(self, x_train, y_train, params, x_test=None, y_test=None, verbose=1, print_plot=False, save_weights=False):

        if "epochs" in params.keys():
            if params["epochs"] <= 0:
                raise ValueError("At least 1 epoch")
            else:
                epochs = params["epochs"]

        if "batch_size" in params.keys() and params["batch_size"] > 0:
            batch_size = min(params["batch_size"], len(x_train))
        else:
            batch_size = len(x_train)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        scores = []
        scores_test = []
        loss_scores = []
        loss_scores_test = []
        bar_update = [int(i * epochs / 20) for i in range(1, 21)]
        width = bar_update.count(0)
        n_g = None

        if save_weights:
            self.saved_weights = []
            self.saved_bias = []
            self.saved_weights.append(deepcopy(self.weights))
            self.saved_bias.append(deepcopy(self.bias))

        for epoch in range(epochs):
            if verbose > 0:
                # Update the progress bar
                print("epoch: ", epoch, "\t[" + "#" * width + " " * (19 - width) + "]")
                if epoch in bar_update:
                    width += bar_update.count(epoch)

            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]

            # Batching of data
            for chunk in range(0, len(x_train), batch_size):
                upper_bound = min(len(x_train), chunk + batch_size)

                sum_states = self.states * 0.
                grad = self.weights * 0.
                bias_grad = self.bias * 0.
                self.delta *= 0.

                if "SGD" in self.optimizer.__repr__():
                    (n_g, n_g_bias) = self.optimizer.nesterov_grad()
                    if n_g is not None:
                        for l in range(len(self.weights)):
                            self.weights[l] += n_g[l]
                            self.bias[l] += n_g_bias[l]

                for x_tr, y_tr in zip(x_train[chunk:upper_bound].tolist(), y_train[chunk:upper_bound].tolist()):
                    sum_states = self.forward_propagate(x_tr)
                    self.delta = self.back_propagate(y_tr)
                    next_grad, next_bias_grad = self.compute_gradient(sum_states)
                    grad += next_grad
                    bias_grad += next_bias_grad

                if n_g is not None:
                    for l in range(len(self.weights)):
                        self.weights[l] -= n_g[l]
                        self.bias[l] -= n_g_bias[l]

                batch = upper_bound - chunk
                self.weights, self.bias = self.optimizer.update(
                    self.weights, self.bias, grad / batch, bias_grad / batch, epoch)

                if save_weights:
                    self.saved_weights.append(deepcopy(self.weights))
                    self.saved_bias.append(deepcopy(self.bias))

            if print_plot:
                # Estimate over each epoch the score
                prediction = self.predict(x_train)
                score = params["performance_function"](prediction, y_train)
                scores.append(score)
                loss_scores.append(self.loss.f(prediction, y_train))
                if x_test is not None and y_test is not None:
                    prediction_test = self.predict(x_test)
                    score_ts = params["performance_function"](prediction_test, y_test)
                    scores_test.append(score_ts)
                    loss_scores_test.append(self.loss.f(prediction_test, y_test))

        if print_plot:
            # Plot over each epoch of the score
            if scores_test:
                plot_graph(scores, scores_test, ylabel=params["performance_function"].__name__, name=self.name)
                plot_graph(loss_scores, loss_scores_test, ylabel=self.loss.__repr__(), name='loss-' + self.name)
            else:
                plot_graph(scores, ylabel=params["performance_function"].__name__, name=self.name)
                plot_graph(loss_scores, ylabel=self.loss.__repr__(), name='loss-' + self.name)

    def predict(self, x_test):
        """Make a prediction given a list of inputs.


        Keyword arguments:

        x_test -- the input values where the prediction is needed

        Output:

        net_output -- the output array containing the predicted network outputs
        """
        results = []
        for x_ts in x_test:
            self.forward_propagate(x_ts)
            net_output = self.states[-1]
            results.append(deepcopy(net_output))

        return np.asarray(results)

    @staticmethod
    def train_and_result(x_train, y_train, x_test, y_test, params, print_plot=False, save_weights=False):
        """Instantiate, train and returns a Feed Forward Neural Network with a specific set of hyperparameters.


        Keyword arguments:

        x_train    -- the input training data

        y_train    -- the target training data

        x_test     -- the input test data

        y_test     -- the target test data

        params     -- the network parameters

        print_plot -- flag for a terminal plot (default False)


        This function allows to instantiate a new neural network with the hyperparameters
        in params, and then assess the efficacy of the net by evaluation over
        training and test set.


        Output:

        score -- a dictionary containing the resulting trained network and all the performance results.
        """
        name = params["name"] if "name" in params.keys() else "net"

        net = NeuralNet(
            len(x_train[0]),
            params["optimizer"],
            params["loss"],
            params["task_type"],
            params["gradient_rule"],
            name)

        weights = params["init_weights"] if params["init_weights"].any() else [None] * len(params["inner_dimension"])
        bias = params["init_bias"] if params["init_bias"].any() else [None] * len(params["inner_dimension"])

        for neurons_number, act_fun, wei, bia in zip(params["inner_dimension"], params["activation_function"], weights, bias):
            net.add_layer(neurons_number, activation_fun=act_fun, use_bias=params['use_bias'], init_weights=wei, bias_weights=bia)

        if params["first_activation"]:
            net.initialize(params["first_activation"])
        else:
            net.initialize()

        # Saving weights for network comparison
        pickle.dump(net.weights, open("weights.p", "wb"))

        if "verbose" in params.keys():
            net.fit(x_train, y_train, params, x_test, y_test, verbose=params["verbose"], print_plot=print_plot, save_weights=save_weights)
            if params["verbose"] > 0:
                print("** Model: ")
                NeuralNet.print_dict(params, color_keys="red")
                print("end **")
        else:
            net.fit(x_train, y_train, params, x_test, y_test, print_plot=print_plot, save_weights=save_weights)

        if y_test is not None:
            score = {"tr_score": params["performance_function"](net.predict(x_train), y_train),
                     "ts_score": params["performance_function"](net.predict(x_test), y_test),
                     "trained_net": net}
            print("tr_score: ", score["tr_score"], " ts_score: ", score["ts_score"], "**")
        else:
            score = {"tr_score": params["performance_function"](net.predict(x_train), y_train),
                     "trained_net": net}
            print("tr_score: ", score["tr_score"], "**")

        # DrawNN.draw(net, params, "lol")
        # print(net.weights)
        return score

    @staticmethod
    def train_without_ms(x_tr, y_tr, x_ts=None, y_ts=None, hyp=None, exp=None, name='', save_weights=False):
        """Train the Neural Network without the use of model selection.


        Keyword arguments:

        x_tr -- the input training data

        y_tr -- the target training data

        x_ts -- the input test data

        y_ts -- the target test data

        hyp  -- the hyperparameters of the network

        exp  -- the experimental settings parameters

        Output:

        net  -- the trained Neural Network with test results
        """
        if hyp is None:
            raise ValueError('hyp cannot be None')

        if exp is None:
            raise ValueError('exp cannot be None')

        params = HyperParameters.extraploate_hyperparameters(hyp, exp.params)

        for i, _ in enumerate(params):
            params[i]["name"] = name + "_plot_" + str(i)

        return NeuralNet.train_and_result(x_tr, y_tr, x_ts, y_ts, params[0], print_plot=True, save_weights=save_weights)

    @staticmethod
    def generate_field_data(input_s, target_s, test_input_s, test_target_s, hyps, exp, legend):
        """Generate the Z-axis data to approximate a level error plot.


        Keyword arguments:

        input_data  -- the input data

        target_data -- the target data

        hyp         -- the hyperparameters of the network

        exp         -- the experimental settings parameters

        p1_max      -- maximum span value of first parameter in the level plot

        p2_max      -- maximum span value of second parameter in the level plot

        resolution  -- resolution of the level plot


        This function tries, From a data-set (made from input and target data) to recreate an
        approximation to the level plot of the error. Since the level plot can be plotted only
        with 2 parameters it only works with a Neural Net with 1 inner layer and 2 weights
        (one from input to inner and one from inner to output).
        The way the output is generated is by creating a field (a matrix) of plausible weights
        values and by evaluating every combination using the data-set, the choosen error function
        and the network properties.
        """

        params = HyperParameters.extraploate_hyperparameters(hyps[0], exp.params)[0]
        name = "Level Plot Hyp"

        net = NeuralNet(
            len(input_s[0]),
            params["optimizer"],
            params["loss"],
            params["task_type"],
            params["gradient_rule"],
            name)

        weights = [None] * len(params["inner_dimension"])
        bias = [None] * len(params["inner_dimension"])

        for neurons_number, act_fun, wei, bia in zip(params["inner_dimension"], params["activation_function"], weights, bias):
            net.add_layer(
                neurons_number,
                activation_fun=act_fun,
                use_bias=params['use_bias'],
                init_weights=wei, bias_weights=bia
            )

        if params["first_activation"]:
            net.initialize(params["first_activation"])
        else:
            net.initialize()

        trained_weights = []
        trained_biases = []
        for hyp in hyps:
            trained_net = NeuralNet.train_without_ms(input_s, target_s, input_s, target_s, hyp, exp, save_weights=True)
            trained_weights.append(trained_net["trained_net"].saved_weights)
            trained_biases.append(trained_net["trained_net"].saved_bias)

        def f(x, y):
            net.weights[0][0][0] = x
            net.bias[0][0] = y
            return params["performance_function"](net.predict(input_s), target_s)

        data_f = DataErrorFunction(f)

        plot_contour(data_f, weights=trained_weights, biases=trained_biases, legend=legend)

    def print_net(self):
        """
        This function print the internal state of the net
        """
        def pprint(matrix):
            print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))

        print("^^------------^^")
        print("Net:")
        print("weights")
        pprint(self.weights)
        print("bias")
        pprint(self.bias)
        print("states")
        pprint(self.states)
        print("delta")
        pprint(self.delta)
        print("net")
        pprint(self.net)
        print("^^^^^^^^^^^^^^^^")

    @staticmethod
    def print_dict(dictionary, color_keys="reset"):
        """
        This function is based on ANSI escape code to print colorful dicionary
        :param dictionary: dictionary to be print
        :param color_keys: color of keys Black: \u001b[30m
                                         Red: \u001b[31m
                                         Green: \u001b[32m
                                         Yellow: \u001b[33m
                                         Blue: \u001b[34m
                                         Magenta: \u001b[35m
                                         Cyan: \u001b[36m
                                         White: \u001b[37m
                                         Reset: \u001b[0m
        """
        if isinstance(dictionary, dict):
            color = dict(black="\u001b[30m",
                         red="\u001b[31m",
                         green="\u001b[32m",
                         yellow="\u001b[33m",
                         blue="\u001b[34m",
                         magenta="\u001b[35m",
                         cyan="\u001b[36m",
                         white="\u001b[37m",
                         reset="\u001b[0m")
            color_keys = color_keys.lower()
            if color_keys not in color.keys():
                raise ValueError("Wrong color, choose between ", color.keys())

            print("\t", color[color_keys.lower()] + "{")
            for key in dictionary.keys():
                print("\t ", color[color_keys.lower()] + key, ": ", color["reset"] + str(dictionary[key]))
            print("\t", color[color_keys.lower()] + "}" + color["reset"])
        else:
            raise TypeError(type(dictionary), " is not a dict type")
