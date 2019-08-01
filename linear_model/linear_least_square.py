import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from linear_model.optimizer import SGD, Adam, AdaMax
from neural_network.loss_func import LOSS_DICT
matplotlib.use('Agg')


class LinearLeastSquare:

    def __init__(self, optimizer, bias=False):
        self.weights = None
        self.bias = bias
        if not isinstance(optimizer, SGD) and not isinstance(optimizer, Adam) and not isinstance(optimizer, AdaMax):
            raise ValueError(type(optimizer), " is not a good optimizer")
        self.optimizer = optimizer
        self.loss_func = LOSS_DICT["lse"]

    def predict(self, x_train):
        if not isinstance(x_train, np.ndarray):
            x_train = np.asarray(x_train)
        return np.dot(x_train, self.weights)

    def fit(self, x_train, y_train, epochs, batch_size=None, init_weights=None, x_test=None, y_test=None, print_plot=False):

        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must be of same dimension.")

        if not isinstance(epochs, int):
            raise ValueError("epochs must be an int.")
        epochs = 10 if epochs <= 0 else epochs

        # Looking for the input dimension
        input_dim = len(x_train[0])

        if init_weights is not None:
            self.weights = init_weights
        else:
            self.weights = np.random.uniform(-0.5, 0.5, input_dim)

        if self.bias:
            bias = np.ones(shape=(len(x_train), 1))
            x_train = np.append(bias, x_train, axis=1)

        batch_size = len(x_train) if batch_size is None else batch_size
        scores = []
        scores_test = []
        for e in range(epochs):
            for chunk in range(0, len(x_train), batch_size):
                upper_bound = min(len(x_train), chunk + batch_size)

                x = x_train[chunk:upper_bound]
                y = y_train[chunk:upper_bound]
                y_pred = self.predict(x)

                self.weights = self.optimizer.update(self.weights, x, y, y_pred, batch_size, e, self.loss_func.dxf)

            if print_plot:
                    # Estimate over each epoch the score
                    prediction = self.predict(x_train)
                    score = self.loss_func.f(prediction, y_train)
                    scores.append(score)
                    if x_test is not None and y_test is not None:
                        prediction_test = self.predict(x_test)
                        score_ts = self.loss_func.f(prediction_test, y_test)
                        scores_test.append(score_ts)

        if print_plot:
            # Plot over each epoch of the score
            if scores_test:
                plot_graph(scores, scores_test, ylabel="least square", name="lls_train")
            else:
                plot_graph(scores, ylabel="least square", name="lls_test")


def normalization(x_train):
    if isinstance(x_train, DataFrame):
        for c in x_train.columns:
            x_train[c] = (x_train[c] - x_train[c].mean()) / x_train[c].std()

    elif isinstance(x_train, np.ndarray):
        for feature in x_train.T:
            m = np.mean(feature)
            std = np.std(feature)

            feature -= m
            feature /= std
    else:
        raise ValueError(type(x_train), " is not a Dataframe or numpy array instance.")

    return x_train


def plot_graph(vals, vals_ts=None, ylabel="least square", name="lls"):
    _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(vals, linewidth=0.4, label="Training Set")
    if vals_ts is not None:
        ax.plot(vals_ts, linestyle="dashed", linewidth=0.4, label="Test Set")
    ax.set_ylabel(ylabel)
    ax.set_xlabel('epochs')
    ax.legend(loc='best')

    plt.tight_layout()

    plt.savefig(name + ".png", dpi=400)
