from typing import List, Callable, Tuple

import numpy as np

from NNHiddenLayer import NNHiddenLayer
from NNLayer import NNLayer
from NNOutputLayer import NNOutputLayer
from utils import relu, softmax_vector, cross_entropy_loss_with_grad


class NN:
    layers = List[NNLayer]  # the layers of the network
    act_func: Callable[[np.ndarray], np.ndarray]  # activation function
    output_func: Callable[[np.ndarray], np.ndarray]  # output function
    loss_func_with_grad: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]  # the loss function (for a
    # single sample) with it's gradient
    output_layer: NNOutputLayer

    def __init__(self, neurons_counts: List[int], act_func: Callable[[np.ndarray], np.ndarray] = relu,
                 output_func: Callable[[np.ndarray], np.ndarray] = softmax_vector,
                 loss_func_with_grad: Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]] = cross_entropy_loss_with_grad):
        self.layers: List[NNLayer] = []

        # hidden layers
        for i in range(len(neurons_counts) - 2):
            k1 = neurons_counts[i]
            k2 = neurons_counts[i + 1]
            self.layers.append(NNHiddenLayer(i + 1, k1, k2, act_func))

        # output layer
        i = len(neurons_counts) - 2
        k1 = neurons_counts[i]
        k2 = neurons_counts[i + 1]
        self.output_layer = NNOutputLayer(i + 1, k1, k2, output_func, loss_func_with_grad)
        self.layers.append(self.output_layer)

        self.act_func = act_func
        self.output_func = output_func
        self.loss_func = loss_func_with_grad

    def feed_forward(self, x_input: np.ndarray) -> np.ndarray:
        """
        Gets an input x, and returns the networks output
        """
        x = x_input
        for layer in self.layers:
            x = layer.output(x)
        return x

    def objective_loss(self, X_input: np.ndarray, Y_input: np.ndarray) -> float:
        """
        Gets a sample set {(x_i,y_i)} and returns the loss the network gets on it.
        """
        m = X_input.shape[1]
        loss = 0
        # grad = ?  TODO Continue here, what is the gradient?
        for i in range(m):
            _ = self.feed_forward(X_input[:, i])
            y = Y_input[:, i]
            loss += self.output_layer.calc_loss_and_grad(y)
        return loss / m

    # def objective_loss(self):
    #     # """
    #     # :param W: n x n_labels
    #     # :param b: n_labels x 1 (but 1-D array)
    #     # """
    #     # objective_loss = 0
    #     # for i in range(self.m):
    #     #     label = self.Y[:, i].to_list().index(1)
    #     #     objective_loss += softmax_cross_entropy_loss(self.X[:, i].dot(W[:, label]) + b[label], label)
    #     pass
    #
    # def output(self):
    #     pass
    #
    # def loss(self, x_inputs: np.ndarray, y: List[int]) -> float:
    #     m = len(y)
    #     res = 0
    #     for i, x_input in enumerate(x_inputs):
    #         x_output = self.feed_forward(x_input)
    #         res += self.loss_func(x_output, y[i])
    #     return res / m


#### FROM SOFTMAX LAYER WITH LOVE

# class NNOutputLayer(NNLayer):
#     n_labels: int  # number of labels (should equal the output dimension)
#     m: int  # sample size
#     X: np.ndarray  # n x m
#     Y: np.ndarray  # n_labels x m , each column looks like [0, ..., 1, ..., 0]^T
#
#     def __init__(self, X: np.ndarray, Y: np.ndarray):
#         self.n_labels = Y.shape[0]
#         self.m = self.Y.shape[1]
#         self.X = X
#         self.Y = Y

# def loss(self, W: np.ndarray, b: np.ndarray) -> float:
#     """
#     :param W: n x n_labels
#     :param b: n_labels x 1 (but 1-D array)
#     """
#     objective_loss = 0
#     for i in range(self.m):
#         label = self.Y[:, i].to_list().index(1)
#         objective_loss += softmax_cross_entropy_loss(self.X[:, i].dot(W[:, label]) + b[label], label)
#     return objective_loss / self.m

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