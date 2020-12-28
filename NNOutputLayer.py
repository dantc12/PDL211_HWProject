import numpy as np

from utils import softmax_vector, softmax_cross_entropy_loss


class SoftMax:
    n_labels: int
    m: int
    X: np.ndarray  # n x m
    Y: np.ndarray  # n_labels x m , each column looks like [0, ..., 1, ..., 0]^T

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.n_labels = Y.shape[0]
        self.m = self.Y.shape[1]
        self.X = X
        self.Y = Y

    def objective_loss(self, W: np.ndarray, b: np.ndarray) -> float:
        """
        :param W: n x n_labels
        :param b: n_labels x 1 (but 1-D array)
        """
        objective_loss = 0
        for i in range(self.m):
            label = self.Y[:, i].to_list().index(1)
            objective_loss += softmax_cross_entropy_loss(self.X[:, i].dot(W[:, label]) + b[label], label)
        return objective_loss / self.m

    def _exp_X_T_W_j_plus_bias(self, W: np.ndarray, b: np.ndarray, j: int) -> float:
        """
        :param W: n x n_labels
        :param b: n_labels x 1 (but 1-D array)
        :param j: int
        """
        #  X.T: m x n
        #  W[:, j]: n x 1 (but 1-D array)
        #  => X.T.dot(W[:, j]): m x 1 (but 1-D array)
        #  => X.T.dot(W[:, j]): m x 1 (but 1-D array)
        return np.exp(self.X.T.dot(W[:, j]) + np.ones(self.m) * b[j])  # m x 1 (but 1-D array)

    def grad_w_j(self, W: np.ndarray, b: np.ndarray, j: int) -> np.ndarray:
        """
        :param W: n x n_labels
        :param b: n_labels x 1 (but 1-D array)
        :param j: int
        """
        c_j = self.Y[j, :]  # m x 1 (but 1-D array)
        return (1 / self.m) * self.X.dot(
            (self._exp_X_T_W_j_plus_bias(W, b, j) /
             (np.sum([self._exp_X_T_W_j_plus_bias(W, b, k) for k in range(self.n_labels)]))) - c_j.T)  # n x 1
