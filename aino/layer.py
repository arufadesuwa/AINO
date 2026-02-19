from .backend import xp
from typing import Optional
from . import activations


class Layer:
    """
    Represents a single layer in a multi-layer perceptron (MLP) using Vectorization.

    Computes transformations using a weight matrix and a bias vector
    for highly optimized parallel computation across CPU or GPU.
    """

    def __init__(self, input_dim: int, amount: Optional[int] = None, activation_type: str = 'relu'):
        """
        Initializes the neural network layer with weights and biases.

        Uses Glorot/Xavier uniform initialization for the weights to ensure
        stable gradients during training. Biases are initialized to zero.

        Args:
            input_dim (int): The number of features or neurons from the previous layer.
            amount (Optional[int]): The number of neurons in this layer. Defaults to 10.
            activation_type (str): The activation function to use ('relu', 'sigmoid', 'tanh').
                                   Defaults to 'relu'.
        """
        self.activation_type: str = activation_type if activation_type is not None else 'relu'
        self.amount: int = amount if amount is not None else 10

        limit = xp.sqrt(6 / (input_dim + self.amount))

        self.w: xp.ndarray = xp.random.uniform(-limit, limit, size=(input_dim, self.amount)).astype(xp.float32)
        self.b: xp.ndarray = xp.zeros((1, self.amount), dtype=xp.float32)

        # Cache for backpropagation
        self.last_x: Optional[xp.ndarray] = None
        self.last_z: Optional[xp.ndarray] = None
        self.dW: Optional[xp.ndarray] = None
        self.db: Optional[xp.ndarray] = None

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        """
        Performs the forward pass for this layer.

        Calculates the linear transformation (Z = XW + b) and applies the
        chosen non-linear activation function.

        Args:
            x (xp.ndarray): The input data matrix of shape (batch_size, input_dim).

        Returns:
            xp.ndarray: The activated output matrix of shape (batch_size, amount).
        """
        self.last_x = x
        self.last_z = xp.dot(x, self.w) + self.b

        if self.activation_type == 'sigmoid':
            return activations.sigmoid(self.last_z)
        elif self.activation_type == 'tanh':
            return activations.tanh(self.last_z)
        else:
            return activations.relu(self.last_z)

    def backward(self, error: xp.ndarray) -> xp.ndarray:
        """
        Performs the backward pass (backpropagation) for this layer.

        Calculates the gradient of the loss with respect to the weights (dW)
        and biases (db), and computes the error to be passed to the previous layer.

        Args:
            error (xp.ndarray): The gradient of the loss from the subsequent layer.

        Returns:
            xp.ndarray: The gradient of the loss with respect to this layer's input,
                        to be passed down to the previous layer.
        """
        m = self.last_x.shape[0]

        dZ = error * activations.get_derivative(self.activation_type, self.last_z)

        error_for_back = xp.dot(dZ, self.w.T)

        self.dW = xp.dot(self.last_x.T, dZ) / m
        self.db = xp.sum(dZ, axis=0, keepdims=True) / m

        return error_for_back

    def update(self, n: float) -> None:
        """
        Updates the weights and biases using the computed gradients.

        Applies standard Gradient Descent optimization.

        Args:
            n (float): The learning rate determining the step size of the update.
        """
        self.w -= (n * self.dW)
        self.b -= (n * self.db)