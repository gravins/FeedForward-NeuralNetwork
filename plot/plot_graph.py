import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def set_plot_proprieties():

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 13

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_graph(vals, vals_ts=None, score_lab="Training Set", score_lab_ts="Test Set", xlabel="epochs", ylabel="score", name="plot"):

    _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(vals, linewidth=0.4, label=score_lab)

    if vals_ts is not None:
        ax.plot(vals_ts, linestyle="dashed", linewidth=0.4, label=score_lab_ts)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(loc='best')

    plt.tight_layout()

    plt.savefig(name + ".png", dpi=400)
    # plt.show()


def plot_mds(plot_arg, name):
    _, axf = plt.subplots(figsize=(10, 10))

    first_arg = [arg[0] for arg in plot_arg]
    second_arg = [arg[1] for arg in plot_arg]

    colors = [x for x in range(len(first_arg))]
    axf.scatter(first_arg, second_arg, c=colors, cmap=plt.hot())

    axf.set_ylabel("target")
    axf.set_xlabel("input")

    plt.tight_layout()

    plt.savefig("plot/plots/" + name + ".png", dpi=400)


def plot_mds_all(plot_arg, plot_output_arg, name):
    _, axf = plt.subplots(figsize=(10, 10))

    first_arg = [arg[0] for arg in plot_arg]
    second_arg = [arg[1] for arg in plot_arg]

    output_first_arg = [arg[0] for arg in plot_output_arg]
    output_second_arg = [arg[1] for arg in plot_output_arg]

    sorted_target_first = []
    sorted_target_second = []
    sorted_output_first = []
    sorted_output_second = []
    for tar2, tar1, out1, out2 in sorted(zip(second_arg, first_arg, output_first_arg, output_second_arg)):
        sorted_target_first.append(tar1)
        sorted_target_second.append(tar2)
        sorted_output_first.append(out1)
        sorted_output_second.append(out2)

    colors = [x for x in range(len(second_arg))]
    axf.scatter(sorted_target_first, sorted_target_second, c=colors, cmap=plt.hot(), edgecolors='black', linewidth=2)
    axf.scatter(sorted_output_first, sorted_output_second, c=colors, cmap=plt.hot(), edgecolors='blue', linewidth=2)

    axf.set_ylabel("output var 1")
    axf.set_xlabel("output var 2")

    plt.tight_layout()

    plt.savefig("plot/plots/" + name + ".png", dpi=400)


def pca_plot(data_frame, net_output, name):

    df_input = data_frame.drop(['y_0', 'y_1', 'x_0'], axis=1)
    df_target = data_frame[["y_0", "y_1"]]
    df_output = pd.DataFrame(net_output, columns=['y_0', 'y_1'])

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_input)
    principal_df = pd.DataFrame(
        data=principal_components,
        columns=['pc1', 'pc2'])

    df_final = pd.concat([principal_df, df_target], axis=1)

    fig = plt.figure(figsize=(20, 8))
    axf = fig.add_subplot(121, projection='3d')
    axf.set_xlabel('Principal Component 1 (' + format(pca.explained_variance_ratio_[0], '02f') + ')', fontsize=15)
    axf.set_ylabel('Principal Component 2 (' + format(pca.explained_variance_ratio_[1], '02f') + ')', fontsize=15)
    axf.set_title('2 Component PCA: Target (' + format(pca.explained_variance_ratio_.cumsum()[1], '02f') + ')', fontsize=20)
    axf.scatter(df_final["pc1"], df_final["pc2"], df_final["y_0"], c=df_final["y_1"], cmap=plt.hot())

    axf2 = fig.add_subplot(122, projection='3d')
    axf2.set_xlabel('Principal Component 1 (' + format(pca.explained_variance_ratio_[0], '02f') + ')', fontsize=15)
    axf2.set_ylabel('Principal Component 2 (' + format(pca.explained_variance_ratio_[1], '02f') + ')', fontsize=15)
    axf2.set_title('2 Component PCA: Output (' + format(pca.explained_variance_ratio_.cumsum()[1], '02f') + ')', fontsize=20)
    axf2.scatter(df_final["pc1"], df_final["pc2"], df_output["y_0"], c=df_output["y_1"], cmap=plt.hot())

    plt.savefig("plot/plots/" + name + ".png", dpi=400)


def plot_sorted(first_arg, second_arg, name):
    _, ax = plt.subplots(figsize=(10, 10))

    sorted_input = []
    sorted_target = []
    for inp, tar in sorted(zip(first_arg, second_arg)):
        sorted_input.append(inp)
        sorted_target.append(tar)
    ax.plot(sorted_input, sorted_target, 'o')

    ax.set_ylabel("target")
    ax.set_xlabel("input")

    plt.tight_layout()

    plt.savefig(name + ".png", dpi=200)
    #plt.show()


def plot_iris3d(X, y, name, labels, weights=None, fontsize=12):

    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    fig.suptitle('Iris Dataset plot', fontweight='bold')
    plt.tight_layout(pad=4, w_pad=1.3, h_pad=2)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    z_min, z_max = X[:, 2].min() - .5, X[:, 2].max() + .5

    colors = ("red", "blue", "green")
    classes = (0, 1)
    type1 = 'petal'
    type2 = 'sepal'

    if weights is not None:
        w1, w2, w3 = 0, 1, 2
        # x_w = np.linspace(4,8)
        # y_w = np.linspace(2,7)
        x_w = [5, 8]
        y_w = [3, 7]
        grid_x, grid_y = np.meshgrid(x_w, y_w)
        z_w = -(grid_x * weights[w1] + grid_y * weights[w2] - 0.5) / weights[w3]

        mycmap = plt.get_cmap('gist_earth')
        surf1 = axs.plot_surface(x_w, y_w, z_w, alpha=0.5, cmap=mycmap)
        fig.colorbar(surf1, ax=axs, shrink=0.5, aspect=5)

    for color, label, target_class in zip(colors, labels, classes):
        target_index = y == target_class
        data = X[target_index, :]
        axs.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Set1,
                    edgecolor='k', s=[10], alpha=0.6, label=label)

    axs.set_title('Petal and Sepal comparison', fontsize=fontsize)
    axs.set(xlabel=type1 + ' length', ylabel=type2 + ' length', zlabel=type2 + ' width')
    axs.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(z_min, z_max))
    axs.legend(loc=2)

    plt.savefig(name + ".png", dpi=200)


def plot_iris(X, y, name, labels, weights=None):

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Iris Dataset plot', fontweight='bold')
    plt.tight_layout(pad=4, w_pad=1.3, h_pad=2)

    plot_iris_sepal_petal(axs[0], X[:, :2], y, 'sepal', labels, weights)
    plot_iris_sepal_petal(axs[1], X[:, 2:], y, 'petal', labels, weights)

    plt.savefig(name + ".png", dpi=200)


def plot_iris_sepal_petal(axs, X, y, type_name, labels, weights, fontsize=12):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    colors = ("red", "blue")
    classes = (0, 1)

    if weights is not None:
        w1, w2 = (0, 1) if type_name == 'sepal' else (2, 3)
        x_w = np.linspace(0, 10)
        y_w = x_w * (-weights[w1] / weights[w2]) + 0.5 / weights[w2]
        axs.plot(x_w, y_w)

    for color, label, target_class in zip(colors, labels, classes):
        target_index = y == target_class
        data = X[target_index, :]
        axs.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.Set1,
                    edgecolor='k', s=[10], alpha=0.6, label=label)

    axs.set_title(type_name + ' comparison', fontsize=fontsize)
    axs.set(xlabel=type_name + ' length', ylabel=type_name + ' width')
    axs.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    axs.legend(loc=2)


def plot_contour(error_function, resolution=150, weights=None, biases=None, save_name="level_plot", legend=[""]):

    graph_size = error_function.graph_size
    x_size = graph_size[0]
    y_size = graph_size[1]
    xlist = np.linspace(x_size[0], x_size[1], resolution)
    ylist = np.linspace(y_size[0], y_size[1], resolution)

    X, Y = np.meshgrid(xlist, ylist)
    Z = error_function.f(X, Y)

    f = plt.figure()
    ax = plt.gca()
    contour = plt.contour(xlist, ylist, Z, error_function.levels, colors='black', alpha=0.8)
    # plt.clabel(contour, cmap='RdGy', fmt='%2.1f', fontsize=12)
    # contour_filled = plt.contourf(X, Y, Z, levels, cmap='RdGy')
    plt.imshow(Z, extent=[x_size[0], x_size[1], y_size[0], y_size[1]], origin='lower', cmap='gist_yarg', alpha=0.95)
    plt.colorbar()
    plt.axis(aspect='image')

    plt.title("Plot level: " + save_name)
    plt.xlabel('First Parameter (Weight)')
    plt.ylabel('Second Parameter (Bias)')

    if weights:
        label_num = 0
        print("new")
        for weight, bias in zip(weights, biases):
            x = []
            y = []
            for w, b in zip(weight, bias):
                x.append(w[0][0][0])
                y.append(b[0][0])
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(x, y, linewidth=2, alpha=0.8, color=color, label=legend[label_num])
            plt.plot(x[-1], y[-1], marker='o', color=color)
            label_num += 1

    plt.legend()
    plt.savefig("plot/plots/level_plot/" + save_name + ".png", dpi=200)
    plt.close(f)
