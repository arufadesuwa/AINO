import numpy as np
from main import Perceptron
from typing import List, Union, Optional

class Layer:
    """
    Represents a single layer in a multi-layer perceptron (MLP).

    A layer consists of a collection of Perceptron objects that all receive
    the same input vector and produce a combined output vector.
    """

    def __init__(self, input_dim: int, amount: Optional[int] = None, activation_type: str = 'relu'):
        """
        Initializes the Layer and populates it with Perceptrons.

        Args:
            input_dim (int): The number of inputs each perceptron receives.
            amount (int, optional): Number of neurons in this layer. Defaults to 10.
            activation_type (str): The activation function to use for all neurons in the layer.
        """
        self.activation_type: str = activation_type if activation_type is not None else 'relu'
        self.amount: int = amount if amount is not None else 10
        self.layer: List[Perceptron] = []
        self.input_dim: int = input_dim

        # Initialize the neurons
        self.add_perceptron(self.amount)

    def add_perceptron(self, amount: int) -> None:
        """
        Instantiates and adds Perceptron units to the layer.

        Args:
            amount (int): The number of Perceptrons to create and add.
        """
        for _ in range(amount):
            p = Perceptron(dim=self.input_dim, activation_type=self.activation_type)
            self.layer.append(p)

    def forward(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Passes the input through every perceptron in the layer.

        Args:
            x (Union[np.ndarray, List[float]]): The input vector from the previous layer or data.

        Returns:
            np.ndarray: A vector of activated outputs, one from each neuron.
        """
        Y = []
        for perceptron in self.layer:
            Y.append(perceptron.forward(x))
        return np.array(Y)

    def backward(self, error: np.ndarray, n: float) -> np.ndarray:
        """
        Performs backpropagation for the layer.

        Calculates the error gradient for each neuron, updates their weights,
        and aggregates the error to be passed to the preceding layer.

        Args:
            error (np.ndarray): The error vector from the subsequent layer (gradient of loss).
            n (float): The learning rate.

        Returns:
            np.ndarray: The accumulated error vector for the previous layer.
        """
        error_for_back = np.zeros(self.input_dim)

        for i in range(len(self.layer)):
            perceptron = self.layer[i]
            perceptron_error = error[i]

            D = perceptron_error * perceptron.get_derivative()

            error_for_back += D * perceptron.w

            perceptron.update_weight(n, D)

        return error_for_back