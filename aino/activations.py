import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Calculates the Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    """Calculates the Hyperbolic Tangent activation function."""
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """Calculates the ReLU activation function."""
    return np.maximum(0.0, x)


def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    """Calculates Sigmoid derivative given the output y."""
    return y * (1.0 - y)


def tanh_derivative(y: np.ndarray) -> np.ndarray:
    """Calculates Tanh derivative given the output y."""
    return 1.0 - (y ** 2)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Calculates ReLU derivative given the logit z."""
    return np.where(z > 0.0, 1.0, 0.0)


def get_derivative(activation_type: str, z: np.ndarray) -> np.ndarray:
    """Routes the derivative calculation based on activation type."""
    if activation_type == 'sigmoid':
        y = sigmoid(z)
        return sigmoid_derivative(y)
    elif activation_type == 'tanh':
        y = tanh(z)
        return tanh_derivative(y)
    elif activation_type == 'relu':
        return relu_derivative(z)

    return np.ones_like(z)