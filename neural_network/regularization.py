import numpy as np


class Regularization:
    def __init__(self, f, dxf):
        # callable is generic: works with built in functions too
        if not callable(f) or not callable(dxf):
            raise ValueError(type(f), " or ", type(dxf), " is not a function")
        self.f = f
        self.dxf = dxf

    def __repr__(self):
        return str(self.f.__name__)


def l2(weights, reg_lambda=0.0):
    return reg_lambda * np.sum((np.square(weights)))


def l1(weights, reg_lambda=0.0):
    return (np.abs(weights)).sum() * reg_lambda


def l2reguldx(weights, reg_lambda=0.0):
    return 2 * (weights * reg_lambda)


def l1reguldx(weights, reg_lambda=0.0):
    # Derivative of L1 regularization. Note that it is not differentiable in
    # 0, we therefore use a subgradient(0 in this case).
    return reg_lambda * (np.sign(weights))


REGULARIZATION_DICT = {"L2": Regularization(l2, l2reguldx),
                       "L1": Regularization(l1, l1reguldx)}
