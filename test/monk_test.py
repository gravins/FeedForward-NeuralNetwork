from preprocessing.parserExcel import get_dataframe_from_excel, dataframe2list, one_hot_encoding
from neural_network.activation_func import ACTIVATION_DICT
from neural_network.loss_func import accuracy
from neural_network.neural_net import NeuralNet
from preprocessing.enums import TaskType
from preprocessing.task import Task
from model_selection.model_selection import ModelSelection
from model_selection.generate_folds import KFolds
from preprocessing.hyperparameters import HyperParameters
from preprocessing.experiment_settings import ExperimentSettings
from neural_network.optimizer import SGD, OPTIMIZER_DICT
import matplotlib
import matplotlib.pyplot as plt
import pickle


class MonkTest:

    @staticmethod
    def init(num):

        data_frame = get_dataframe_from_excel("./data_set/monk" + str(num) + "_train.csv", out_col=[0])

        target_s = [[i] for i in dataframe2list(data_frame["y_0"])]
        inp = data_frame.drop(['y_0', 'x_7'], axis=1)
        one_hot = one_hot_encoding(inp, inp.columns)
        input_s = dataframe2list(one_hot)

        print("input dimension: ", len(input_s), "x", len(input_s[0]))
        print("target dimension: ", len(target_s), "x", len(target_s[0]))

        test_data_frame = get_dataframe_from_excel("./data_set/monk" + str(num) + "_test.csv", out_col=[0])

        test_target_s = [[i] for i in dataframe2list(test_data_frame["y_0"])]
        inp = test_data_frame.drop(['y_0', 'x_7'], axis=1)
        one_hot = one_hot_encoding(inp, inp.columns)
        test_input_s = dataframe2list(one_hot)

        inner_dim = [[7, 1]]

        activ_fun = [[ACTIVATION_DICT["tanh"]] * (len(inner_dim[0]) - 1) +
                     [ACTIVATION_DICT["tanh"]]]

        return input_s, target_s, test_input_s, test_target_s, inner_dim, activ_fun

    @staticmethod
    def run():

        input_s, target_s, test_input_s, test_target_s, inner_dim, activ_fun = MonkTest.init(1)

        exp = ExperimentSettings(
            performance_function=accuracy,
            select_function=max,
        )

        optimizer = OPTIMIZER_DICT['ADAMAX']()

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[250],
            batch_size=[1],
            optimizer=[optimizer],
            task_type=[TaskType.classification]
        )

        # print(NeuralNet.train_without_ms(input_s, target_s, test_input_s, test_target_s, hyp, exp))

        fold_s = KFolds.cross_validation_folds(data_indexes=list(range(len(target_s))), folds_number=2)

        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)

        res = mod_sel.run_grid_search(n_workers=80)
        pickle.dump(res, open("monk1_validation_res.p", "wb"))

        best_nets = []
        for r in res["results_list"]:
            if r["avg_ts_score"] == res["best_score"][1]:
                best_nets.append(r["params"])

        print("Evaluating best net:")
        evaluate_and_plot(input_s, target_s, test_input_s, test_target_s, best_nets)


class MonkTest2:

    @staticmethod
    def run():

        input_s, target_s, test_input_s, test_target_s, inner_dim, activ_fun = MonkTest.init(2)

        exp = ExperimentSettings(
            performance_function=accuracy,
            select_function=max,
        )

        optimizer1 = OPTIMIZER_DICT['SGD'](lr=0.001, momentum=0.1, nesterov=True)
        optimizer2 = OPTIMIZER_DICT['ADAM']()
        optimizer3 = OPTIMIZER_DICT['ADAMAX']()

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[500],
            batch_size=[1],
            optimizer=[optimizer1],
            task_type=[TaskType.classification]
        )

        #print(NeuralNet.train_without_ms(input_s, target_s, test_input_s, test_target_s, hyp, exp))

        fold_s = KFolds.cross_validation_folds(data_indexes=list(range(len(target_s))), folds_number=2)

        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)

        res = mod_sel.run_grid_search(n_workers=1)
        pickle.dump(res, open("monk2_validation_res.p", "wb"))

        best_nets = []
        for r in res["results_list"]:
            if r["avg_ts_score"] == res["best_score"][1]:
                best_nets.append(r["params"])

        print("Evaluating best net:")
        evaluate_and_plot(input_s, target_s, test_input_s, test_target_s, best_nets)


class MonkTest3:

    @staticmethod
    def run():

        input_s, target_s, test_input_s, test_target_s, inner_dim, activ_fun = MonkTest.init(3)

        exp = ExperimentSettings(
            # performance_function=accuracy,
            # select_function=max,
        )

        optimizer = OPTIMIZER_DICT['ADAMAX']()

        hyp = HyperParameters(
            inner_dimension=inner_dim,
            activation_function=activ_fun,
            epochs=[500],
            batch_size=[1],
            optimizer=[optimizer],
            task_type=[TaskType.regression]
        )

        # print(NeuralNet.train_without_ms(input_s, target_s, test_input_s, test_target_s, hyp, exp))

        fold_s = KFolds.cross_validation_folds(data_indexes=list(range(len(target_s))), folds_number=6)

        task = Task(input_s, target_s, fold_s)
        mod_sel = ModelSelection(task, exp, hyp)

        res = mod_sel.run_grid_search(n_workers=80)
        pickle.dump(res, open("monk3_validation_res.p", "wb"))

        best_nets = []
        for r in res["results_list"]:
            if r["avg_ts_score"] == res["best_score"][1]:
                best_nets.append(r["params"])


def evaluate_and_plot(x_tr, y_tr, x_ts, y_ts, params):
    score = []
    for p in params:
        score.append((params, NeuralNet.train_and_result(x_tr, y_tr, x_ts, y_ts, p, print_plot=True)))

    pickle.dump(score, open("score_best_net.p", "wb"))
