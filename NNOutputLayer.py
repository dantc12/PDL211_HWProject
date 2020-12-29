from typing import Callable, Tuple

import numpy as np

from NNLayer import NNLayer


class NNOutputLayer(NNLayer):
    output_func: Callable[[np.ndarray], np.ndarray]  # output function
    loss_func_with_grad: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]  # the loss function (on single
    # sample) with it's gradient
    loss: float  # the loss for safe-keeping (on single sample)

    def __init__(self, l: int, k1: int, k2: int, output_func: Callable[[np.ndarray], np.ndarray],
                 loss_func_with_grad: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]):
        self.output_func = output_func
        self.loss_func_with_grad = loss_func_with_grad
        super().__init__(l, k1, k2)

    def output(self, x_input: np.ndarray) -> np.ndarray:
        self.layer_output = self.output_func(x_input.dot(self.W) + self.b)
        return self.layer_output

    def calc_loss_and_grad(self, true_y_output: np.ndarray) -> Tuple[float, np.ndarray]:
        if not self.layer_output:
            raise Exception('Tried to call loss function before calculated output')
        self.loss, self.grad = self.loss_func_with_grad(self.layer_output, true_y_output)
        return self.loss, self.grad

    # def _exp_X_T_W_j_plus_bias(self, W: np.ndarray, b: np.ndarray, j: int) -> float:
    #     """
    #     :param W: n x n_labels
    #     :param b: n_labels x 1 (but 1-D array)
    #     :param j: int
    #     """
    #     #  X.T: m x n
    #     #  W[:, j]: n x 1 (but 1-D array)
    #     #  => X.T.dot(W[:, j]): m x 1 (but 1-D array)
    #     #  => X.T.dot(W[:, j]): m x 1 (but 1-D array)
    #     return np.exp(self.X.T.dot(W[:, j]) + np.ones(self.m) * b[j])  # m x 1 (but 1-D array)
    #
    # def grad_w_j(self, W: np.ndarray, b: np.ndarray, j: int) -> np.ndarray:
    #     """
    #     :param W: n x n_labels
    #     :param b: n_labels x 1 (but 1-D array)
    #     :param j: int
    #     """
    #     c_j = self.Y[j, :]  # m x 1 (but 1-D array)
    #     return (1 / self.m) * self.X.dot(
    #         (self._exp_X_T_W_j_plus_bias(W, b, j) /
    #          (np.sum([self._exp_X_T_W_j_plus_bias(W, b, k) for k in range(self.n_labels)]))) - c_j.T)  # n x 1
