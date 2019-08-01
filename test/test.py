from preprocessing.task import Task
from preprocessing.hyperparameters import HyperParameters
from preprocessing.experiment_settings import ExperimentSettings
from preprocessing.enums import TaskType
from model_selection.model_selection import ModelSelection
from model_selection.generate_folds import KFolds
from neural_network.activation_func import ACTIVATION_DICT
from neural_network.loss_func import LOSS_DICT, mean_square_error, accuracy
from neural_network.neural_net import NeuralNet
from neural_network.optimizer import OPTIMIZER_DICT
from linear_model.error_functions import DataErrorFunction, AckleyFunction, RosenbrockFunction, QUAD_FUNCTION_DICT
from plot.net_plot import DrawNN
from plot.plot_graph import plot_sorted, plot_contour
from random import randint, random
import matplotlib.pyplot as plt
import numpy as np
import itertools


class Test1:

    @staticmethod
    def init():
        input_s = [[randint(-10, 10) / 20 + 0.5,
                    randint(-10, 10) / 20 + 0.5,
                    randint(-10, 10) / 20 + 0.5] for x in range(10)]

        target_s = [[-2.95 + np.cos(inp[2] * inp[0]) + inp[0] * 3 - inp[1] ** 3 +
                     0.5 * inp[2] - inp[0] * inp[2] + np.sin(inp[1]),
                     np.cos(inp[0] + inp[1]) - np.sin(inp[2]) + inp[1] * inp[2]] for inp in input_s]

        # target_s = [[np.cos(inp[0]+inp[1]) - np.sin(inp[2]) + inp[1]*inp[2]] for inp in input_s]

        print(max(target_s))
        print(min(target_s))

        print("input", input_s)
        print("target", target_s)
        print("sum: ", sum(target_s[0]))

        inner_dim = [[4, 2, 2]]
        activ_fun = [[ACTIVATION_DICT["tanh"],
                      ACTIVATION_DICT["tanh"],
                      ACTIVATION_DICT["linear"]]]

        return np.asarray(input_s), np.asarray(target_s), inner_dim, activ_fun

    @staticmethod
    def run():

        input_s, target_s, inner_dim, activ_fun = Test1.init()

        exp = ExperimentSettings()
        hyp = HyperParameters(inner_dimension=inner_dim, activation_function=activ_fun)
        fold_s = KFolds.cross_validation_folds(data_indexes=list(range(len(target_s))), folds_number=2)
        # fold_s = KFolds.double_cross_validation_folds(data_indexes=list(range(len(target_s))), external_folds_dim=1, internal_folds_dim=2)

        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)

        res = mod_sel.run_grid_search(n_workers=1)
        # res = mod_sel.double_cross_validation(n_workers=2)

        print(res)
        plt.show()


class Test2():

    @staticmethod
    def run():

        input_s, target_s, inner_dim, activ_fun = Test1.init()

        params = dict(performance_function=mean_square_error,
                      select_function=min,
                      inner_dimension=inner_dim[0],
                      epochs=30,
                      batch_size=1,
                      activation_function=activ_fun[0],
                      first_activation=[],
                      learning_rate=0.01,
                      loss=LOSS_DICT["mse"],
                      lambda_regularization=0,
                      momentum=0,
                      task_type=TaskType.regression)

        res = NeuralNet.train_and_result(input_s, target_s, input_s, target_s, params)

        print(res)

        plt.show()


class Test3():

    @staticmethod
    def run():

        input_s, target_s, _, _ = Test1.init()

        params = dict(inner_dimension=[2, 1])
        net = NeuralNet(len(input_s[0]), 0.01, 0, 0, LOSS_DICT["mse"])
        net.add_layer(2, activation_fun=ACTIVATION_DICT["linear"], use_bias=False, init_weights=np.asarray([[0.11, 0.12], [0.21, 0.08]]))
        net.add_layer(1, activation_fun=ACTIVATION_DICT["linear"], use_bias=False, init_weights=np.asarray([[0.14], [0.15]]))
        net.initialize()
        input_s = [[2, 3]]
        target_s = [[17]]
        net.fit(input_s, target_s, 1)
        DrawNN.draw(net, params, "net_graph1")
        net.fit(input_s, target_s, 1)
        DrawNN.draw(net, params, "net_graph2")

        plt.show()


class Test4():

    # This test aim to evaluate the efficacy of regularization
    @staticmethod
    def run():
        tr_dim = 100
        ts_dim = 200

        input_s = [[(x / tr_dim)] for x in range(tr_dim)]
        target_s = [[np.sin(inp[0] * 15) + np.cos(inp[0] * 2)] for inp in input_s]

        test_input_s = [[x / ts_dim] for x in range(ts_dim)]
        test_target_s = [[np.sin(inp[0] * 15) + np.cos(inp[0] * 2)] for inp in test_input_s]

        plot_sorted(input_s, target_s, "train")
        plot_sorted(test_input_s, test_target_s, "test")

        inner_dim = [[30, 30, 1]]
        activ_fun = [[ACTIVATION_DICT["tanh"],
                      ACTIVATION_DICT["tanh"],
                      ACTIVATION_DICT["linear"]]]

        exp = ExperimentSettings()
        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[500],
            batch_size=[5])

        hyp2 = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[100],
            batch_size=[1])

        fold_s = KFolds.cross_validation_folds(data_indexes=list(range(len(target_s))), folds_number=2)

        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)
        mod_sel_with_regul = ModelSelection(task, exp, hyp2)

        res = mod_sel.run_grid_search(n_workers=1)
        res_with_regul = mod_sel_with_regul.run_grid_search(n_workers=1)

        net = (res["results_list"][res["best_score"][0]]["single_result"][0]["score"]["trained_net"])
        net_with_regul = (res_with_regul["results_list"][res_with_regul["best_score"][0]]["single_result"][0]["score"]["trained_net"])

        prediction = net.predict(test_input_s)
        prediction_with_regul = net_with_regul.predict(test_input_s)

        plot_sorted(test_input_s, prediction, "result")
        plot_sorted(test_input_s, prediction_with_regul, "result_with_regul")

        plt.show()


class Test5():

    # This test aim to evaluate the efficacy of regularization
    @staticmethod
    def run():
        tr_dim = 160
        ts_dim = 500

        input_s = [[(x / tr_dim)] for x in range(tr_dim)]
        target_s = [[np.sin(inp[0] * 15) + np.cos(inp[0] * 2)] for inp in input_s]

        test_input_s = [[x / ts_dim] for x in range(ts_dim)]
        test_target_s = [[np.sin(inp[0] * 15) + np.cos(inp[0] * 2)] for inp in test_input_s]

        plot_sorted(input_s, target_s, "train")
        plot_sorted(test_input_s, test_target_s, "test")

        inner_dim = [[30, 20, 30, 1]]
        activ_fun = [[ACTIVATION_DICT["tanh"],
                      ACTIVATION_DICT["tanh"],
                      ACTIVATION_DICT["tanh"],
                      ACTIVATION_DICT["linear"]]]

        exp = ExperimentSettings()

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[300],
            batch_size=[1])

        fold_s = KFolds.cross_validation_folds(data_indexes=list(range(len(target_s))), folds_number=2)

        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)

        res = mod_sel.run_grid_search(n_workers=2)

        # nets = [net["score"]["trained_net"] for net in res["results_list"][res["best_score"][0]]["single_result"]]
        nets = [net["single_result"][0]["score"]["trained_net"] for net in res["results_list"]]

        [plot_sorted(test_input_s, net.predict(test_input_s), net.name) for net in nets]

        plt.show()


class StepByStepExample():

    @staticmethod
    def run():

        x_tr = [[0.05, 0.1]]
        y_tr = [[0.01, 0.99]]

        inner_dim = [[2, 2, 2, 2]]
        init_weights = [np.asarray([np.asarray([[0.15, 0.25], [0.2, 0.3]]), np.asarray([[0.4, 0.5], [0.45, 0.55]])])]
        init_bias = [np.asarray([np.asarray([0.35, 0.35]), np.asarray([0.6, 0.6])])]

        activ_fun = [[ACTIVATION_DICT["sigmoid"],
                      ACTIVATION_DICT["sigmoid"],
                      ACTIVATION_DICT["sigmoid"],
                      ACTIVATION_DICT["sigmoid"]]]

        exp = ExperimentSettings()

        optimizer = OPTIMIZER_DICT['ADAMAX'](lr=0.5)

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            init_weights=init_weights,
            init_bias=init_bias,
            activation_function=activ_fun,
            epochs=[1000],
            batch_size=[1],
            optimizer=[optimizer])

        print(NeuralNet.train_without_ms(x_tr, y_tr, x_tr, y_tr, hyp, exp))


class LevelSetTest():

    # This test aim to evaluate the efficacy of regularization
    @staticmethod
    def run():
        tr_dim = 30
        ts_dim = 10

        input_s = [[(x / tr_dim)] for x in range(tr_dim)]
        # target_s = [[np.sin(inp[0] * 15) + np.cos(inp[0] * 2)] for inp in input_s]
        # target_s = [[np.sin(inp[0])] for inp in input_s]
        target_s = [[inp[0] * 2 + 3] for inp in input_s]

        test_input_s = [[x / ts_dim] for x in range(ts_dim)]
        # test_target_s = [[np.sin(inp[0] * 15) + np.cos(inp[0] * 2)] for inp in test_input_s]
        # test_target_s = [[np.sin(inp[0])] for inp in test_input_s]
        test_target_s = [[inp[0] * 2 + 3] for inp in test_input_s]

        plot_sorted(input_s, target_s, "plot/plots/train")
        plot_sorted(test_input_s, test_target_s, "plot/plots/test")

        inner_dim = [[1]]
        activ_fun = [[ACTIVATION_DICT["linear"]]]

        optimizer1 = OPTIMIZER_DICT['SGD'](lr=0.5)
        optimizer2 = OPTIMIZER_DICT['ADAM'](lr=1)
        optimizer3 = OPTIMIZER_DICT['ADAMAX'](lr=1)

        exp = ExperimentSettings()

        hyp1 = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            use_bias=[True],
            init_weights=[np.asarray([[[1.]]])],
            init_bias=[np.asarray([[1.]])],
            epochs=[200],
            optimizer=[optimizer1],
            task_type=[TaskType.regression],
            batch_size=[10])

        hyp2 = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            use_bias=[True],
            init_weights=[np.asarray([[[1.]]])],
            init_bias=[np.asarray([[1.]])],
            epochs=[200],
            optimizer=[optimizer2],
            task_type=[TaskType.regression],
            batch_size=[10])

        hyp3 = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            use_bias=[True],
            init_weights=[np.asarray([[[1.]]])],
            init_bias=[np.asarray([[1.]])],
            epochs=[200],
            optimizer=[optimizer3],
            task_type=[TaskType.regression],
            batch_size=[10])

        hyps = [hyp1, hyp2, hyp3]

        NeuralNet.generate_field_data(
            input_s,
            target_s,
            test_input_s,
            test_target_s,
            hyps,
            exp,
            legend=["GD", "GD - MOMENTUM", "GD - NESTEROV", "ADAM", "ADAMAX"]
        )


class TestErrorFunctions():

    @staticmethod
    def run():

        input_s = [[0]]
        target_s = [[0]]

        optimizers = [
            OPTIMIZER_DICT['SGD'](lr=5e-4),
            OPTIMIZER_DICT['SGD'](lr=5e-4, momentum=0.9),
            OPTIMIZER_DICT['SGD'](lr=5e-4, momentum=0.9, nesterov=True),
            OPTIMIZER_DICT['ADAM'](lr=5e-2),
            OPTIMIZER_DICT['ADAMAX'](lr=5e-2)
        ]

        ep = 200
        bs = 1

        for name, err_f in zip(QUAD_FUNCTION_DICT.keys(), QUAD_FUNCTION_DICT.values()):

            err_fun = err_f[0]
            gradient_rule = err_f[1]
            start_x, start_y = err_fun.starting_point
            exp = ExperimentSettings()

            def genhyp(optimizer):
                return HyperParameters(
                    init_weights=[np.asarray([[[start_x]]])],
                    init_bias=[np.asarray([[start_y]])],
                    epochs=[ep],
                    optimizer=[optimizer],
                    gradient_rule=[gradient_rule],
                    batch_size=[bs]
                )

            hyps = [genhyp(opt) for opt in optimizers]

            trained_weights = []
            trained_biases = []
            for hyp in hyps:
                trained_net = NeuralNet.train_without_ms(input_s, target_s, input_s, target_s, hyp, exp, save_weights=True)
                trained_weights.append(trained_net["trained_net"].saved_weights)
                trained_biases.append(trained_net["trained_net"].saved_bias)

            plot_contour(
                err_fun,
                weights=trained_weights,
                biases=trained_biases,
                resolution=150,
                save_name=name,
                legend=["GD", "GD - MOMENTUM", "GD - NESTEROV", "ADAM", "ADAMAX"]
            )
