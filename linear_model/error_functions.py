import numpy as np
from numpy import linalg as LA


class ErrorFunction(object):

    def __init__(self, f, error_plot_level_id=0):
        self.f = f
        self.levels = ErrorFunction.error_plot_levels(error_plot_level_id)
        self.starting_point = ErrorFunction.get_starting_point(error_plot_level_id)
        self.graph_size = ErrorFunction.get_graph_size(error_plot_level_id)

    def __repr__(self):
        return str(self.f.__name__)

    @staticmethod
    def function_error(func):
        """Generic way of approaching a function error to create a level plot:

        Keyword arguments:
        func -- the function used to evaluate the error given a point (x, y)


        from the two network parameters

        param 1 = [x1, x2, ..., xn]
        param 2 = [y1, y2, ..., yn]

        is defined a meshgrid

        mesh_param 1 = [ [x1, ..., xn ], [x1, ..., xn], ..., [x1, ..., xn] ]
        mesh_param 2 = [ [y1, ..., y1 ], [y2, ..., y2], ..., [yn, ..., yn] ]

        so that a grid of points is created, where every point is took from every couple
        in the same position:

        row 1 = [(x1, y1), (x2, y1), (x3, y1), ..., (xn, y1)]
        row 2 = [(x1, y2), (x2, y2), (x3, y2), ..., (xn, y2)]
        ...
        row k = [(x1, yk), (x2, yk), (x3, yk), ..., (xn, yk)]
        ...
        row n = [(x1, yn), (x2, yn), (x3, yn), ..., (xn, yn)].

        This algorithm divides first every row using zip(mesh_param 1, mesh_param 2)
        and inside every row takes every couple of point by reiterating map and zip
        functions on every row. On every resulting point it evaluates the error function
        value (passed as parameter) which should return a 2d structure resembling
        the Z dimension in the level plot.

        Output:
        -- a lambda function used to evaluate Z values from X and Y vectors.
        """
        return lambda a, b: \
            list(map(lambda y:
                     list(map(lambda x: func(np.asarray([[x[0]], [x[1]]])),
                              zip(y[0], y[1]))),
                     zip(a, b)))

    # Error levels at which the level plot should draw curves.
    @staticmethod
    def error_plot_levels(n):
        levels = np.asarray(
            [
                [-15, -14, -10, 0, 10, 25, 50, 75, 100, 150, 200, 250, 300],
                [-28, -20, -10, 0, 10, 25, 50, 75, 100],
                [-150, -125, -75, -50, -35, -20, -10, 0, 10, 25, 50],
                [-4000, -2000, -1000, -750, -500, -250, 0, 250, 500, 1000, 2000],
                [-25, -20, -10, 0, 10, 25, 50],
                [0, 2, 4, 6, 8, 10, 12, 14],
                [0, 1, 2, 3, 4, 6, 15, 25, 50, 100, 200],
                [0, 1, 2.5, 3.5, 5, 7, 10, 20],
                [0, 2, 5, 10, 20, 50],
                [0.1, 0.5, 2, 5, 10, 20, 50, 100, 200, 300],
                [0.1, 0.5, 2, 5, 10, 20, 50, 75, 100, 125, 150, 200],
            ])
        return levels[n]

    @staticmethod
    def get_starting_point(n):
        points = np.asarray(
            [
                (0.7, 0.7),
                (0.7, 0.7),
                (-3., -5.),
                (1., -1.),
                (-3, -3.2),
                (0.7, 0.7),
                (0.1, 1.5),
                (0.5, 0.7),
                (0.7, 0.7),
                (0.7, 0.7),
            ]
        )
        return points[n]

    @staticmethod
    def get_graph_size(n):
        size = np.asarray(
            [
                ((-3, 1.5), (-3, 1.5)),
                ((-5, 2), (-5, 2)),
                ((-15, -2), (-15, -2)),
                ((-30, 5), (-30, 5)),
                ((-5, -2.5), (-5, -2.5)),
                ((0.5, 1), (0.4, 0.8)),
                ((-3, 3), (-3, 3)),
                ((-3, 3), (-3, 3)),
                ((-1, 2), (-1, 1)),
                ((-5, 10), (-5, 10))
            ]
        )
        return size[n]

    @staticmethod
    def evaluate_gradient(weights, bias):
        """This function is used to evaluate directly the gradient in a Neural Network
        with one weight and one bias.


        Keyword arguments:
        weights -- the network weights
        bias    -- the network bias


        Output:
        grad -- the computed 2-dimensional gradient with the corrisponding sub-class function
        """
        return 0, 0


class QuadraticFunction(ErrorFunction):

    def __init__(self, n):
        """Generic function to evaluate the error function in a quadratic problem.
        f = @(x, y) [x, y] * Q * [x; y] / 2 + q' * [x; y];
        """
        mat = QuadraticFunction.generic_quadratic_matrix(n)
        self.Q = mat[0]
        self.q = mat[1]
        super(QuadraticFunction, self).__init__(self.quadratic_function_error_n(self.Q, self.q), n)

    # Set of possible Q Matrix and q vector used in the quadratic problem
    # to test the network performance.
    @staticmethod
    def generic_quadratic_matrix(n):
        q = np.asarray([[10], [5]])
        Q = np.asarray(
            [
                [[6, -2], [-2, 6]],
                [[5, -3], [-3, 5]],
                [[4, -4], [-4, 4]],
                [[3, -5], [-5, 3]],
                [[101, -99], [-99, 101]]
            ])
        return (Q[n], q)

    def quadratic_function_error(self, x, Q, q):
        return (x.T.dot(Q).dot(x) / 2 + q.T.dot(x))[0][0]

    # Function used to easily choose from all test-functions on quadratic problem.
    def quadratic_function_error_n(self, Q, q):
        return self.function_error(lambda x: self.quadratic_function_error(x, Q, q))

    @staticmethod
    def evaluate_gradient(n, weights, bias):
        """Function derivative is
        dx(f_{Q, q}(x, y)) / d(x, y) = Q * [x, y] + q
        """
        x = weights[0][0][0]
        y = bias[0][0]

        mat = QuadraticFunction.generic_quadratic_matrix(n)
        Q = mat[0]
        q = mat[1]

        dx_y = Q.dot(np.asarray([x, y]).T) + q.T
        return dx_y[0][0], dx_y[0][1]


class RosenbrockFunction(ErrorFunction):

    QUADRATIC_LEVEL_ID = 5

    def __init__(self):
        """The Rosenbrock function error implementation
        f = @(x, y) 100 * ( y - x^2 )^2 + ( x - 1 )^2
        """
        super(RosenbrockFunction, self).__init__(self.get_rosenbrock_function_error(), self.QUADRATIC_LEVEL_ID)

    def rosenbrock_function_error(self, point):
        x = point[0][0]
        y = point[1][0]
        return 100 * np.power(y - (np.power(x, 2)), 2) + np.power(x - 1, 2)

    def get_rosenbrock_function_error(self):
        return self.function_error(self.rosenbrock_function_error)

    @staticmethod
    def evaluate_gradient(weights, bias):
        """Function derivative is
        dx(f(x, y)) / d(x) = 2 * x - 400 * x * ( - x^2 + y ) - 2
        dx(f(x, y)) / d(y) = - 200 * x^2 + 200 * y
        """
        x = weights[0][0][0]
        y = bias[0][0]

        dx = 2 * x - 400 * x * (-np.power(x, 2) + y) - 2
        dy = - 200 * np.power(x, 2) + 200 * y
        return dx, dy


class SixHumpCamelFunction(ErrorFunction):

    QUADRATIC_LEVEL_ID = 6

    def __init__(self):
        """The Six hump camel function error implementation
        f = @(x, y) ( 4 - 2.1 * x^2 + x^4 / 3 ) * x^2 + x * y + 4 * ( y^2 - 1 ) * y^2
        """
        super(SixHumpCamelFunction, self).__init__(self.get_sixhumpcamel_function_error(), self.QUADRATIC_LEVEL_ID)

    def sixhumpcamel_function_error(self, point):
        x = point[0][0]
        y = point[1][0]
        return (4 - 2.1 * np.power(x, 2) + np.power(x, 4) / 3) * \
            np.power(x, 2) + x * y + 4 * (np.power(y, 2) - 1) * np.power(y, 2)

    def get_sixhumpcamel_function_error(self):
        return self.function_error(self.sixhumpcamel_function_error)

    @staticmethod
    def evaluate_gradient(weights, bias):
        """Function derivative is
        dx(f(x, y)) / d(x) = 2 * x^5 - ( 42 * x^3 ) / 5 + 8 * x + y
        dx(f(x, y)) / d(y) = 16 * y^3 - 8 * y + x
        """
        x = weights[0][0][0]
        y = bias[0][0]

        dx = 2 * np.power(x, 5) - (42 * np.power(x, 3)) / 5 + 8 * x + y
        dy = 16 * np.power(y, 3) - 8 * y + x
        return dx, dy


class AckleyFunction(ErrorFunction):

    QUADRATIC_LEVEL_ID = 7

    def __init__(self):
        """The Ackley function error implementation
        f = @(x, y) - 20 * exp( - 0.2 * sqrt( ( x^2 + y^2 ) / 2 ) ) ...
                   - exp( ( cos( 2 * pi * x ) + cos( 2 * pi * y ) ) / 2 ) ...
                   + 20 + exp(1)
        """
        super(AckleyFunction, self).__init__(self.get_ackley_function_error(), self.QUADRATIC_LEVEL_ID)

    def ackley_function_error(self, point):
        x = point[0][0]
        y = point[1][0]
        return -20 * np.exp(-0.2 * np.sqrt((np.power(x, 2) + np.power(y, 2) / 2))) \
            - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2) \
            + 20 + np.exp(1)

    def get_ackley_function_error(self):
        return self.function_error(self.ackley_function_error)

    @staticmethod
    def evaluate_gradient(weights, bias):
        """Function derivative is
        dx(f(x, y)) / d(x) =
           pi*exp(cos(2*pi*x)/2 + cos(2*pi*y)/2)*sin(2*pi*x) +
           (2*x*exp(-(x^2/2 + y^2/2)^(1/2)/5))/(x^2/2 + y^2/2)^(1/2)
        dx(f(x, y)) / d(y) =
           pi*exp(cos(2*pi*x)/2 + cos(2*pi*y)/2)*sin(2*pi*y) +
           (2*y*exp(-(x^2/2 + y^2/2)^(1/2)/5))/(x^2/2 + y^2/2)^(1/2)
        """
        x = weights[0][0][0]
        y = bias[0][0]
        sqn2 = (np.power(x, 2) + np.power(y, 2)) / 2
        cosx = np.cos(2 * np.pi * x)
        cosy = np.cos(2 * np.pi * y)
        comp1 = np.exp(-np.sqrt(sqn2) / 5)
        comp2 = np.exp((cosx + cosy) / 2)
        sinx = np.sin(2 * np.pi * x)
        siny = np.sin(2 * np.pi * y)

        dx = np.pi * comp2 * sinx + 2 * x * comp1 / np.sqrt(sqn2)
        dy = np.pi * comp2 * siny + 2 * y * comp1 / np.sqrt(sqn2)
        return dx, dy


class LassoFunction(ErrorFunction):

    QUADRATIC_LEVEL_ID = 8

    def __init__(self):
        """The Lasso non differentiable function error implementation
        f( x , y ) = || 3 * x + 2 * y - 2 ||_2^2 + 10 ( | x | + | y | )
        """
        super(LassoFunction, self).__init__(self.get_lasso_function_error(), self.QUADRATIC_LEVEL_ID)

    def lasso_function_error(self, point):
        x = point[0][0]
        y = point[1][0]
        return np.power(np.abs(3 * x + 2 * y - 2), 2) + 10 * (np.abs(x) + np.abs(y))

    def get_lasso_function_error(self):
        return self.function_error(self.lasso_function_error)

    @staticmethod
    def evaluate_gradient(weights, bias):
        """Function derivative is
        dx(f(x, y)) / d(x) = 18 * x + 12 * y - 12 + 10 * sign(x)
        dx(f(x, y)) / d(y) = 12 * x + 8  * y - 8  + 10 * sign(y)
        """
        x = weights[0][0][0]
        y = bias[0][0]

        dx = 18 * x + 12 * y - 12 + 10 * np.sign(x)
        dy = 12 * x + 8 * y - 8 + 10 * np.sign(y)
        return dx, dy


class DataErrorFunction(ErrorFunction):

    QUADRATIC_LEVEL_ID = 9

    def __init__(self, eval_fun):
        """This function plot an approximation of the error space using
        directly the data-set
        """
        super(DataErrorFunction, self).__init__(self.get_data_function_error(), self.QUADRATIC_LEVEL_ID)
        self.eval_fun = eval_fun

    def data_function_error(self, point):
        x = point[0][0]
        y = point[1][0]
        return self.eval_fun(x, y)

    def get_data_function_error(self):
        return self.function_error(self.data_function_error)


QUAD_FUNCTION_DICT = {"quad1": (QuadraticFunction(0), lambda w, b: QuadraticFunction.evaluate_gradient(0, w, b)),
                      "quad2": (QuadraticFunction(1), lambda w, b: QuadraticFunction.evaluate_gradient(1, w, b)),
                      "quad3": (QuadraticFunction(2), lambda w, b: QuadraticFunction.evaluate_gradient(2, w, b)),
                      "quad4": (QuadraticFunction(3), lambda w, b: QuadraticFunction.evaluate_gradient(3, w, b)),
                      "quad5": (QuadraticFunction(4), lambda w, b: QuadraticFunction.evaluate_gradient(4, w, b)),
                      "rosenbrock": (RosenbrockFunction(), RosenbrockFunction.evaluate_gradient),
                      "sixhumpcamel": (SixHumpCamelFunction(), SixHumpCamelFunction.evaluate_gradient),
                      "ackley": (AckleyFunction(), AckleyFunction.evaluate_gradient),
                      "lasso": (LassoFunction(), LassoFunction.evaluate_gradient)}
