import numpy as np



# def sigmoid(x: np.ndarray) -> np.ndarray:
#     return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


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


def cross_entropy_loss(prediction_vector: np.ndarray, true_y_vector: np.ndarray) -> float:
    true_label = true_y_vector.tolist().index(1)
    return -np.log(prediction_vector[true_label])


# def loss_on_data_samples(S: SampleSet, output_vector: np.ndarray) -> float:
#     pass
    # return (np.sum([ for i in range(S.m)])) / S.m
