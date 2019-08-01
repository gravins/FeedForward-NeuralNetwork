import numpy as np
import pandas as pd
from copy import deepcopy


class QR:
    """
    QR decomposition for Ax = b
    """

    def __init__(self, data, target):
        """
        :param data: corresponds to the matrix A in Ax = b
        :param target: corresponds to the vector b in Ax = b
        """
        if not (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)):
            raise ValueError("data cannot be instance of ", type(data))
        if not (isinstance(target, np.ndarray) or isinstance(target, pd.DataFrame)):
            raise ValueError("target cannot be instance of ", type(target))

        self.data = data if isinstance(data, np.ndarray) else np.asarray(data.values, dtype=float)
        self.target = target if isinstance(target, np.ndarray) else np.asarray(target.values, dtype=float)
        self.weights = None
        self.Q = None
        self.R = None

    def householder(self, v):
        s = -np.sign(v[0]) * np.linalg.norm(v)
        u = deepcopy(v)
        u[0] = u[0] - s
        return u / np.linalg.norm(u), s

    def decomposition(self):
        self.Q = np.eye(self.data.shape[0], dtype=float)
        self.R = deepcopy(self.data)
        for i in range(self.data.shape[1]):
            u, s = self.householder(np.asarray([self.R[i:, i]]).T)
            self.R[i, i] = s
            self.R[i + 1:, i] = 0
            self.R[i:, i + 1:] -= 2 * u * (u.T.dot(self.R[i:, i + 1:]))
            self.Q[:, i:] -= (self.Q[:, i:].dot(u * 2 * u.T))

        self.Q = self.Q[:, :self.R.shape[1]]
        self.R = self.R[:self.R.shape[1], :]

    def solve(self):
        if self.Q is None or self.R is None:
            raise ValueError("You must perform decomposition before solve")
        self.weights = np.linalg.inv(self.R).dot(self.Q.T.dot(self.target))

    def predict(self, inputs):
        if self.weights is None:
            raise ValueError("You must perform method solve before evaluate")
        if not (isinstance(inputs, np.ndarray) or isinstance(inputs, pd.DataFrame)):
            raise ValueError("inputs cannot be instance of ", type(inputs))

        inputs = inputs if isinstance(inputs, np.ndarray) else inputs.values

        return inputs.dot(self.weights)
