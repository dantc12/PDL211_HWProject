from typing import Callable

from data import test_data
import numpy as np


class NNLayer:
    k1: int
    k2: int
    l: int
    W: np.ndarray  # k1 x k2
    b: np.ndarray  # k2 x 1
    act_func: Callable[[float], float]

    def __init__(self, l: int, W: np.ndarray, b: np.ndarray, act_func: Callable[[float], float]):
        if not NNLayer._check_valid(W, b):
            raise Exception('Tried to created layer with bad weights and/or bias dimensions')
        self.k1, self.k2 = W.shape
        self.l = l

        self.W = W
        self.b = b
        self.act_func = act_func

    @staticmethod
    def _check_valid(W: np.ndarray, b: np.ndarray) -> bool:
        try:
            return W.shape[1] == b.shape[0]
        except IndexError:
            return False

    def x_output(self, x_input: np.ndarray):
        return np.array([x_input.dot(self.W[:, j]) + self.b[j] for j in range(self.k2)])

#
# # for loss we will be using mean square error(MSE)
# def loss(out, Y):
#     s = (np.square(out - Y))
#     s = np.sum(s) / len(y)
#     return (s)
#
#
# # Back propagation of error
# def back_prop(x, y, w1, w2, alpha):
#     # hiden layer
#     z1 = x.dot(w1)  # input from layer 1
#     a1 = sigmoid(z1)  # output of layer 2
#
#     # Output layer
#     z2 = a1.dot(w2)  # input of out layer
#     a2 = sigmoid(z2)  # output of out layer
#     # error in output layer
#     d2 = (a2 - y)
#     d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),
#                      (np.multiply(a1, 1 - a1)))
#
#     # Gradient for w1 and w2
#     w1_adj = x.transpose().dot(d1)
#     w2_adj = a1.transpose().dot(d2)
#
#     # Updating parameters
#     w1 = w1 - (alpha * (w1_adj))
#     w2 = w2 - (alpha * (w2_adj))
#
#     return (w1, w2)
#
#
# def train(x, Y, w1, w2, alpha=0.01, epoch=10):
#     acc = []
#     losss = []
#     for j in range(epoch):
#         l = []
#         for i in range(len(x)):
#             out = f_forward(x[i], w1, w2)
#             l.append((loss(out, Y[i])))
#             w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
#         print("epochs:", j + 1, "======== acc:", (1 - (sum(l) / len(x))) * 100)
#         acc.append((1 - (sum(l) / len(x))) * 100)
#         losss.append(sum(l) / len(x))
#     return (acc, losss, w1, w2)
#
#
# def predict(x, w1, w2):
#     Out = f_forward(x, w1, w2)
#     maxm = 0
#     k = 0
#     for i in range(len(Out[0])):
#         if (maxm < Out[0][i]):
#             maxm = Out[0][i]
#             k = i
#     if (k == 0):
#         print("Image is of letter A.")
#     elif (k == 1):
#         print("Image is of letter B.")
#     else:
#         print("Image is of letter C.")
#     plt.imshow(x.reshape(5, 6))
#     plt.show()


if __name__ == '__main__':
    W1 = generate_wt(30, 5)
    w2 = generate_wt(5, 3)
    # print(w1, "\n\n", w2)
    print('W1 shape: {}'.format(str(W1.shape)))
    print('input x shape: {}'.format(str(test_data.X.shape)))
    xx = f_forward(test_data.X, W1)
    print('feed forward x: {}'.format(str(xx)))
    print('feed forward x shape: {}'.format(str(xx.shape)))
    xx_softmax = normalize_vector(xx)
    print('normalized: {}'.format(str(xx_softmax)))
