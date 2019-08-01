from neural_network import loss_func


class ExperimentSettings:

    def __init__(
            self,
            performance_function=loss_func.mean_square_error,
            select_function=min,
            using_level_plot=False):

        self.params = dict(
            performance_function=performance_function,
            select_function=select_function,
            using_level_plot=using_level_plot)
