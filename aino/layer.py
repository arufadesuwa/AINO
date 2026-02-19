import numpy as np
from typing import Optional
from . import activations


class Layer:
    """
    Represents a single layer in a multi-layer perceptron (MLP) using Vectorization.

    Computes transformations using a weight matrix and a bias vector
    for highly optimized parallel computation.
    """

    def __init__(self, input_dim: int, amount: Optional[int] = None, activation_type: str = 'relu'):
        self.activation_type: str = activation_type if activation_type is not None else 'relu'
        self.amount: int = amount if amount is not None else 10
        limit = np.sqrt(6 / (input_dim + self.amount))
        self.w = np.random.uniform(-limit, limit, size=(input_dim, self.amount)).astype(np.float32)
        self.b = np.zeros((1, self.amount), dtype=np.float32)
        self.last_x: Optional[np.ndarray] = None
        self.last_z: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        self.last_z = np.dot(x, self.w) + self.b

        if self.activation_type == 'sigmoid':
            return activations.sigmoid(self.last_z)
        elif self.activation_type == 'tanh':
            return activations.tanh(self.last_z)
        else:
            return activations.relu(self.last_z)

    def backward(self, error: np.ndarray) -> np.ndarray:
        m = self.last_x.shape[0]

        dZ = error * activations.get_derivative(self.activation_type, self.last_z)
        error_for_back = np.dot(dZ, self.w.T)

        self.dW = np.dot(self.last_x.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        return error_for_back

    def update(self, n: float) -> None:
        """Updates weights and biases using the calculated gradients."""
        self.w -= (n * self.dW)
        self.b -= (n * self.db)