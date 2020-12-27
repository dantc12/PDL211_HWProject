from typing import Callable

from NNLayer import NNLayer
import numpy as np


class NNRandomLayer(NNLayer):
    def __init__(self, l: int, k1: int, k2: int, act_func: Callable[[float], float]):
        W = np.random.randn(k1, k2)
        b = np.random.randn(k2, 1)
        super().__init__(l, W, b, act_func)
