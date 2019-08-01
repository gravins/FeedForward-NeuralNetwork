from preprocessing.enums import TaskType
from neural_network.optimizer import OPTIMIZER_DICT
from preprocessing.parserExcel import get_dataframe_from_excel, dataframe2list
from neural_network.activation_func import ACTIVATION_DICT
from neural_network.loss_func import LOSS_DICT
from neural_network.neural_net import NeuralNet
from preprocessing.task import Task
from model_selection.model_selection import ModelSelection
from model_selection.generate_folds import KFolds
from preprocessing.hyperparameters import HyperParameters
from preprocessing.experiment_settings import ExperimentSettings
from plot.net_plot import DrawNN
from plot.plot_graph import pca_plot, plot_mds, plot_mds_all
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np


class MicheliDataset:

    @staticmethod
    def init():

        data_frame = get_dataframe_from_excel("./data_set/ML-CUP18-TR.csv", out_col=[11, 12])

        target_s = [i for i in dataframe2list(data_frame[["y_0", "y_1"]])]
        input_s = dataframe2list(data_frame.drop(['y_0', 'y_1', 'x_0'], axis=1))

        print("input dimension: ", len(input_s), "x", len(input_s[0]))
        print("target dimension: ", len(target_s), "x", len(target_s[0]))

        test_data_frame = get_dataframe_from_excel("./data_set/ML-CUP18-TS.csv")
        test_input_s = test_data_frame.drop(['x_0'], axis=1)

        inner_dim = [[100, 2]]
        activ_fun = [[ACTIVATION_DICT["tanh"]] * (len(inner_dim[0]) - 1) + [ACTIVATION_DICT["linear"]]]

        return input_s, target_s, test_input_s, inner_dim, activ_fun, data_frame

    @staticmethod
    def run():

        input_s, target_s, test_input_s, inner_dim, activ_fun, data_frame = MicheliDataset.init()

        # plot_mds(target_s, "target")

        exp = ExperimentSettings(
            performance_function=LOSS_DICT["mee"].f,
            select_function=min
        )

        optimizer1 = OPTIMIZER_DICT['SGD'](lr=1e-3, momentum=0.5)
        optimizer2 = OPTIMIZER_DICT['ADAM']()
        optimizer3 = OPTIMIZER_DICT['ADAMAX']()

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[250],
            batch_size=[10],
            optimizer=[optimizer3],
            task_type=[TaskType.regression],
            loss=[LOSS_DICT["mee"]],
            verbose=[1]
        )

        res = NeuralNet.train_without_ms(input_s[:-216], target_s[:-216], input_s[-216:], target_s[-216:], hyp, exp, 'SGD')

        # TRAINING plots
        net = res['trained_net']

        prediction = net.predict(input_s[:-216])
        plot_mds_all(target_s[:-216], prediction, "prediction_all")

        # TEST plots
        prediction_test = net.predict(input_s[-216:])
        plot_mds_all(target_s[-216:], prediction_test, "prediction_all_test")

    @staticmethod
    def run_nested():
        input_s, target_s, test_input_s, inner_dim, activ_fun, data_frame = MicheliDataset.init()

        # plot_mds(target_s, "target")

        exp = ExperimentSettings(
            performance_function=LOSS_DICT["mse"].f,
            select_function=min
        )

        optimizer = [OPTIMIZER_DICT['SGD'](), OPTIMIZER_DICT['ADAMAX'](), OPTIMIZER_DICT['ADAM']()]

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[250],
            batch_size=[1],
            optimizer=optimizer,
            task_type=[TaskType.classification]
        )

        fold_s = KFolds.double_cross_validation_folds(data_indexes=list(range(len(target_s))), external_folds_dim=3, internal_folds_dim=4)
        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)

        res = mod_sel.double_cross_validation(n_workers=50)
        pickle.dump(res, open("cup_nested_res.p", "wb"))


class MicheliModelSelection:

    @staticmethod
    def init():

        data_frame = get_dataframe_from_excel("./data_set/ML_train_70%.csv", out_col=[10, 11])
        target_s = [i for i in dataframe2list(data_frame[["y_0", "y_1"]])]
        input_s = dataframe2list(data_frame.drop(['y_0', 'y_1'], axis=1))

        print("input dimension: ", len(input_s), "x", len(input_s[0]))
        print("target dimension: ", len(target_s), "x", len(target_s[0]))

        test_data_frame = get_dataframe_from_excel("./data_set/ML_test_30%.csv", out_col=[10, 11])
        test_target_s = [i for i in dataframe2list(test_data_frame[["y_0", "y_1"]])]
        test_input_s = dataframe2list(test_data_frame.drop(['y_0', 'y_1'], axis=1))

        inner_dim = [[100, 2]]
        activ_fun = [[ACTIVATION_DICT["tanh"]] * (len(inner_dim[0]) - 1) + [ACTIVATION_DICT["linear"]]]

        return input_s, target_s, test_input_s, test_target_s, inner_dim, activ_fun, data_frame, test_data_frame

    @staticmethod
    def run():

        input_s, target_s, test_input_s, test_target_s, inner_dim, activ_fun, \
            data_frame, test_data_frame = MicheliModelSelection.init()

        exp = ExperimentSettings(
            performance_function=LOSS_DICT["mee"].f,
            select_function=min
        )

        optimizer = [OPTIMIZER_DICT['SGD'](), OPTIMIZER_DICT['ADAMAX'](), OPTIMIZER_DICT['ADAM']()]

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[250],
            batch_size=[1],
            optimizer=optimizer,
            task_type=[TaskType.classification]
        )

        fold_s = KFolds.cross_validation_folds(data_indexes=list(range(len(target_s))), folds_number=2)

        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)

        res = mod_sel.run_grid_search(n_workers=1)
        pickle.dump(res, open("cup_validation_res.p", "wb"))

        best_nets = []
        for r in res["results_list"]:
            if r["avg_ts_score"] == res["best_score"][1]:
                best_nets.append(r["params"])

        evaluate_and_plot(input_s, target_s, test_input_s, test_target_s, best_nets)

        score = exp.params["performance_function"](prediction, target_s)
        print("score: ", score)

        plt.show()


def evaluate_and_plot(x_tr, y_tr, x_ts=None, y_ts=None, params=None):
    score = []
    for p in params:
        score.append((p, NeuralNet.train_and_result(x_tr, y_tr, x_ts, y_ts, p, print_plot=True)))

    pickle.dump(score, open("score_best_net.p", "wb"))
