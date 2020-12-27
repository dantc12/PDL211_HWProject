from typing import List, Callable

import numpy as np

from NNLayer import NNLayer
from NNRandomLayer import NNRandomLayer


class NN:
    layers = List[NNLayer]
    act_func: Callable[[float], float]
    loss_func: Callable[[np.ndarray, int], float]

    def __init__(self, neurons_counts: List[int], act_func: Callable[[float], float],
                 loss_func: Callable[[np.ndarray, int], float]):
        self.layers = []
        self.act_func = act_func
        self.loss_func = loss_func

        self._initialize_network(neurons_counts, act_func)

    def _initialize_network(self, neurons_counts: List[int], act_func: Callable[[float], float]):
        for i in range(len(neurons_counts) - 1):
            k1 = neurons_counts[i]
            k2 = neurons_counts[i + 1]
            self.layers.append(NNRandomLayer(i + 1, k1, k2, act_func))

    def feed_forward(self, x_input: np.ndarray) -> np.ndarray:
        x = x_input
        for layer in self.layers:
            x = layer.x_output(x)
        return x

    def output(self):
        pass

    def loss(self, x_inputs: np.ndarray, y: List[int]) -> float:
        m = len(y)
        res = 0
        for i, x_input in enumerate(x_inputs):
            x_output = self.feed_forward(x_input)
            res += self.loss_func(x_output, y[i])
        return res / m

