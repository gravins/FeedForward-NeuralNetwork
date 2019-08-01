from preprocessing.experiment_settings import ExperimentSettings
from preprocessing.hyperparameters import HyperParameters
from neural_network.neural_net import NeuralNet
from model_selection.generate_folds import KFolds
from pathos.multiprocessing import ProcessingPool
from preprocessing.task import Task
from preprocessing.hyperparameters import HyperParameters
import itertools
import numpy as np


class ModelSelection:

    def __init__(
            self, task,
            exp_settings=ExperimentSettings(),
            hyper_param=HyperParameters()):

        self.param = exp_settings.params
        self.hyper_parameters = hyper_param
        self.task = task

    def run_grid_search(self, n_workers=0, task=None):
        """
        This function perform the grid search, need a KFolds instance to be run
        """
        task = self.task if task is None else task

        if isinstance(task.folds.folds[0].train_indexes, KFolds):
            raise ValueError("You can't run gridsearch over double cross validation settings")

        params = HyperParameters.extraploate_hyperparameters(self.hyper_parameters, self.param)

        # Define name for plot with relative legend
        with open("plot_legend.txt", "w") as f:
            f.write("nome file\t->\tparametri\n")
            for i, p in enumerate(params):
                f.write("\nplot_" + str(i) + "\t->\t" + str(p) + "\n")
                params[i]["name"] = "plot_" + str(i)

        results = []
        if n_workers > 1:
            # nodes - number(and potentially description) of workers, if not given
            #         will autodetect processors
            # ncpus - number of worker processors
            pool = ProcessingPool(nodes=n_workers)

            # For all combinations of hyper-parameters (grid search)
            # run and return the results of a net with those parameters
            results = pool.map(self.run_validation, params, [task] * len(params))
        else:
            for _, par in enumerate(params):
                results.append(self.run_validation(par, task))

        score_list = [r["avg_ts_score"] for r in results]
        most_valuable_res = self.param["select_function"](score_list)
        return {"results_list": results,
                "best_score": (score_list.index(most_valuable_res), most_valuable_res),
                "task": task}

    def run_validation(self, params, task=None):
        task = self.task if task is None else task
        # Create a list of net results at each permutation of the kfold-cv
        results = []

        for folds in task.folds.folds:
            x_tr = task.inputs[folds.train_indexes]
            y_tr = task.targets[folds.train_indexes]
            x_val = task.inputs[folds.test_indexes]
            y_val = task.targets[folds.test_indexes]

            results.append({"score": NeuralNet.train_and_result(x_tr, y_tr, x_val, y_val, params),
                            "params": params})

        return {"avg_tr_score": sum([r["score"]["tr_score"] for r in results]) / len(results),
                "avg_ts_score": sum([r["score"]["ts_score"] for r in results]) / len(results),
                "single_result": results,
                "params": params}

    def double_cross_validation(self, n_workers, task=None):
        task = self.task if task is None else task
        if not isinstance(task.folds.folds[0].train_indexes, KFolds):
            raise ValueError("You can't run double cross validation without correct settings")

        nested_res = {"nested_scores": [], "params": []}

        for ext_f in task.folds.folds:
            # Run the model selection over the internal folds
            internal_task = Task(task.inputs, task.targets, ext_f.train_indexes)
            res = self.run_grid_search(n_workers, internal_task)

            # Train the selected model and test over the esternal fold
            training_indexes = ext_f.train_indexes.folds[0].train_indexes + ext_f.train_indexes.folds[0].test_indexes
            x_tr = [task.inputs[i] for i in training_indexes]
            y_tr = [task.targets[i] for i in training_indexes]
            x_val = [task.inputs[i] for i in ext_f.test_indexes]
            y_val = [task.targets[i] for i in ext_f.test_indexes]

            params = (res["results_list"][res["best_score"][0]])["params"]
            nested_res["params"].append(params)
            nested_res["nested_scores"].append(NeuralNet.train_and_result(x_tr, y_tr, x_val, y_val, params))

        # Store the results
        score_list = []
        for score in nested_res["nested_scores"]:
            score_list.append(score["ts_score"])
        score_list = np.array(score_list)
        nested_res["mean"] = score_list.mean()
        nested_res["std"] = score_list.std()

        return nested_res
