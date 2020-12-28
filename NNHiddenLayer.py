from typing import Callable

import numpy as np

from NNLayer import NNLayer


class NNHiddenLayer(NNLayer):
    act_func: Callable[[np.ndarray], np.ndarray]  # activation function

    def __init__(self, l: int, k1: int, k2: int, act_func: Callable[[np.ndarray], np.ndarray]):
        self.act_func = act_func
        super().__init__(l, k1, k2)

    def output(self, x_input: np.ndarray) -> np.ndarray:
        self.layer_output = self.act_func(x_input.dot(self.W) + self.b)
        # return np.array([x_input.dot(self.W[:, j]) + self.b[j] for j in range(self.k2)])
        return self.layer_output
