from preprocessing.parserExcel import get_dataframe_from_excel, dataframe2list, one_hot_encoding
from linear_model.linear_least_square import LinearLeastSquare, normalization
from linear_model.QRdecomposition import QR
from neural_network.loss_func import least_square_error, accuracy
from linear_model.optimizer import SGD, Adam, AdaMax
from random import randint
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plot.plot_graph import set_plot_proprieties, plot_iris, plot_iris3d, plot_contour
from linear_model.error_functions import QUAD_FUNCTION_DICT, DataErrorFunction
import matplotlib.pyplot as plt
import numpy as np


def iris_test():
    iris = datasets.load_iris()

    x = iris['data'][:, [0, 2, 3]]
    y = (iris['target'] > 0).astype(int)

    labels = (iris.target_names[0], iris.target_names[1] + " / " + iris.target_names[2])
    # labels = (iris.target_names[0], iris.target_names[1], iris.target_names[2])
    plot_iris3d(x, y, 'correct', labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

    return x_train, y_train, x_test, y_test, labels


def lls_monk3():
    # Load training set
    data_frame = get_dataframe_from_excel("./data_set/monk3_train.csv", out_col=[0])

    y_train = dataframe2list(data_frame["y_0"])
    inp = data_frame.drop(['y_0', 'x_7'], axis=1)
    one_hot = one_hot_encoding(inp, inp.columns)
    x_train = normalization(one_hot).values

    # Load test set
    test_data_frame = get_dataframe_from_excel("./data_set/monk3_test.csv", out_col=[0])

    y_test = dataframe2list(test_data_frame["y_0"])
    inp = test_data_frame.drop(['y_0', 'x_7'], axis=1)
    one_hot = one_hot_encoding(inp, inp.columns)
    x_test = normalization(one_hot).values

    return x_train, y_train, x_test, y_test


def lls_test1():
    # Create training set
    x_train = np.asarray([np.asarray([randint(-10, 10) / 20 + 0.5,
                         randint(-10, 10) / 20 + 0.5,
                         randint(-10, 10) / 20 + 0.5]) for x in range(10)])
    x_train = normalization(x_train)

    y_train = [-2.95 * inp[0] for inp in x_train]
    # y_train = [-2.95*inp[0] + inp[1] * 3 - inp[1] *0.01 + inp[2] * inp[1] + inp[2] * inp[2] for inp in x_train]

    return x_train, y_train, None, None


if __name__ == '__main__':

    qr_dec = False

    # x_train, y_train, x_test, y_test = lls_monk3()
    # x_train, y_train, x_test, y_test = lls_test1()
    x_train, y_train, x_test, y_test, labels = iris_test()

    print("input dimension: ", len(x_train), "x", len(x_train[0]))
    if x_test is not None:
        print("target dimension: ", len(x_test), "x", len(x_test[0]))

    if not qr_dec:
        sgd = SGD(lr=0.1, reg_lambda=0, momentum=0.)
        adam = Adam(reg_lambda=0.1)
        adamax = AdaMax()
        predictor = LinearLeastSquare(optimizer=adamax)
        predictor.fit(x_train=x_train, y_train=y_train, epochs=1000, x_test=x_test, y_test=y_test, print_plot=True)
        pred_train = least_square_error(predictor.predict(x_train), y_train)
    else:
        predictor = QR(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float))
        predictor.decomposition()
        predictor.solve()
        pred_train = least_square_error(predictor.predict(np.asarray(x_train, dtype=float)), y_train)
        pred_train_acc = accuracy(predictor.predict(np.asarray(x_train, dtype=float)), y_train)

    print("Prediction train error: ", pred_train)

    if qr_dec:
        print("Prediction train accuracy: ", pred_train_acc)

    if x_test is not None:
        pred_test = least_square_error(predictor.predict(x_test), y_test)
        print("Prediction test error: ", pred_test)

        if qr_dec:
            pred_test = accuracy(predictor.predict(x_test), y_test)
            print("Prediction test accuracy: ", pred_test)

