import numpy as np

from SampleSet import SampleSet


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def normalize_vector(x: np.ndarray) -> np.ndarray:
    """
    Normalize function that converts current layer neurons such they all sum to 1.
    :param x: Current layer neurons
    :return:
    """
    return np.array([v / sum(x) for v in x])

def softmax_p_y(y_label: int, output_vector: np.ndarray) -> float:
    return (np.exp(output_vector[y_label])) / (np.sum(np.exp(output_vector)))

def softmax_p_y_safely(y_label: int, output_vector: np.ndarray) -> float:
    max_o = np.max(output_vector)
    return softmax_p_y(y_label, np.array([o - max_o for o in output_vector]))

def softmax_vector(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()

def softmax_cross_entropy_loss(y_label: int, output_vector: np.ndarray) -> float:
    return - np.log(softmax_p_y_safely(y_label, output_vector))

# def loss_on_data_samples(S: SampleSet, output_vector: np.ndarray) -> float:
#     return (np.sum([ for i in range(S.m)])) / S.m