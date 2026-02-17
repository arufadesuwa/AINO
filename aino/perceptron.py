import numpy as np
from typing import Union, List, Optional

e = 2.718

class Perceptron:
    """
    A simple Perceptron model representing a single neuron in a neural network.
    """

    def __init__(self, dim: int, b: Optional[float] = None, activation_type: str = 'relu'):
        """
        Initializes the Perceptron.

        Args:
            dim (int): The number of input features (dimensionality).
            b (float, optional): Initial bias. Defaults to a random value.
            activation_type (str): The activation function ('relu', 'sigmoid', or 'tanh').
        """
        self.last_z: Optional[float] = None
        self.last_x: Optional[np.ndarray] = None
        self.w: np.ndarray = np.random.uniform(-1, 1, size=dim)
        self.b: float = b if b is not None else np.random.uniform(-1, 1)
        self.activation_type: str = activation_type

    @staticmethod
    def sigmoid(x: float) -> float:
        """Calculates the Sigmoid activation function."""
        return 1 / (1 + e ** -x)

    @staticmethod
    def tanh(x: float) -> float:
        """Calculates the Hyperbolic Tangent activation function."""
        return (e ** x - e ** -x) / (e ** x + e ** -x)

    @staticmethod
    def relu(x: float) -> float:
        """Calculates the ReLU activation function."""
        return max(0.0, x)

    @staticmethod
    def sigmoid_derivative(y: float) -> float:
        """Calculates Sigmoid derivative given the output y."""
        return y * (1.0 - y)

    @staticmethod
    def tanh_derivative(y: float) -> float:
        """Calculates Tanh derivative given the output y."""
        return 1.0 - (y ** 2)

    @staticmethod
    def relu_derivative(z: float) -> float:
        """Calculates ReLU derivative given the logit z."""
        return 1.0 if z > 0 else 0.0

    def get_derivative(self) -> float:
        """
        Calculates the local gradient of the activation function.

        Returns:
            float: The derivative value at self.last_z.
        """
        if self.activation_type == 'sigmoid':
            y = self.sigmoid(self.last_z)
            return self.sigmoid_derivative(y)
        elif self.activation_type == 'tanh':
            y = self.tanh(self.last_z)
            return self.tanh_derivative(y)
        elif self.activation_type == 'relu':
            return self.relu_derivative(self.last_z)
        return 1.0

    def forward(self, x: Union[np.ndarray, List[float]]) -> float:
        """
        Performs a forward pass.

        Args:
            x (Union[np.ndarray, List[float]]): Input feature vector.

        Returns:
            float: The activated output.
        """
        self.last_x = np.array(x)
        y = np.dot(self.w, self.last_x) + self.b
        self.last_z = float(y)

        if self.activation_type == 'sigmoid':
            return self.sigmoid(self.last_z)
        elif self.activation_type == 'tanh':
            return self.tanh(self.last_z)
        else:
            return self.relu(self.last_z)

    def update_weight(self, n: float, D: float) -> np.ndarray:
        """
        Updates weights and bias using the delta rule.

        Args:
            n (float): Learning rate (often denoted as $\eta$).
            D (float): The error gradient/delta value.

        Returns:
            np.ndarray: The weighted error to be passed to the previous layer.
        """
        self.w += n * D * self.last_x
        self.b += n * D
        return D * self.w

if __name__ == '__main__':
    # Perceptron.get
    pass