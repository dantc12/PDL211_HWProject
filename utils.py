from typing import Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def d_relu(x: np.ndarray) -> np.ndarray:
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# def normalize_vector(x: np.ndarray) -> np.ndarray:
#     """
#     Normalize function that converts current layer neurons such they all sum to 1.
#     :param x: Current layer neurons
#     :return:
#     """
#     return np.array([v / sum(x) for v in x])


# def softmax_p_y(output_vector: np.ndarray, y_label: int) -> float:
#     return (np.exp(output_vector[y_label])) / (np.sum(np.exp(output_vector)))
#
#
# def softmax_p_y_safely(output_vector: np.ndarray, y_label: int) -> float:
#     max_o = np.max(output_vector)
#     return softmax_p_y(np.array([o - max_o for o in output_vector]), y_label)


def softmax_vector(x: np.ndarray) -> np.ndarray:
    safe_x = x - np.max(x)
    return np.exp(safe_x) / np.exp(safe_x).sum()


# def softmax_cross_entropy_loss(output_vector: np.ndarray, y_label: int) -> float:
#     return - np.log(softmax_p_y_safely(output_vector, y_label))


def cross_entropy_loss_with_grad(prediction_vector: np.ndarray, true_y_vector: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    The cross entropy loss on a single sample.
    The grad got by reading about gradient of cross entropy loss with softmax.
    :param prediction_vector: Assumed to be the softmax vector
    :param true_y_vector: Assumed to be a classification vector of the type: [0, ..., 1, ..., 0]^T
    :return: the value of the loss and the gradient w.r.t the prediction vector
    """
    true_label = true_y_vector.tolist().index(1)
    return -np.log(prediction_vector[true_label]), prediction_vector - true_y_vector


# def loss_on_data_samples(S: SampleSet, output_vector: np.ndarray) -> float:
#     pass
    # return (np.sum([ for i in range(S.m)])) / S.m
