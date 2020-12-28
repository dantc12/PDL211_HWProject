import numpy as np


def cross_entropy(y, softmax_probs):
    return - np.sum([c_k[i] * np.log2(softmax_probs[i]) for i in range(len(c_k))])


def loss_function_for_single_sample(l, softmax_probs):
    res = []
    for k in range(l):
        c_k = np.zeros(l)
        c_k[k] = 1
        res.append(cross_entropy(c_k, softmax_probs))
    return res


class SoftMaxLayer:
    def __init__(self, l, X, W):
        """

        :param l: Number of possible labels
        :param X: Matrix of all the samples given. Each sample is column. Dimension: d (size of prev. layer) cross m
        (size of sample set)
        :param W: Matrix of all weights from prev. layer to new (softmax) layer. Dimension: d (size of prev. layer)
        cross l (number of possible labels)
        """
        self.l = l
        self.X = X
        self.W = W

        self.XT_W = self.calc_XT_W()


        self.P = np.array([self.calc_P(j) for j in range(l)])

    def get_softmax_layer(self):
        return self.P

    def calc_XT_W(self):
        # TODO switch to tensors
        return self.X.T * self.W

    def calc_P(self, i: int, j: int):
        """
        Compute softmax values for each sets of scores in x.

        x: Length of n, equal to length of neurons of last layer.
        w: Matrix of such that row j is the weights going into "label neuron" j.
           length is n.

        x_i = self.X[:, i]
        w_j = self.W[:, j]
        """
        # xT_W = np.array([self.x.dot(self.W[i]) for i in range(len(self.W))])
        x_iT_W = self.XT_W[i][:]
        eta = np.max(x_iT_W)
        normalizer = np.sum([np.exp(x_iT_W[k] - eta) for k in range(len(self.W))])

        top = np.exp(x_iT_W[j] - eta)
        return top / normalizer

    def loss_function(self):
        summ = np.sum([np])
        inside_log = [np.exp(self.XT_W[:, k]) for k in range(l)]


