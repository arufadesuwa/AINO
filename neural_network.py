from typing import List, Union, Optional, Any
import numpy as np
from model import Layer

class NeuralNetwork:
    """
    A Multi-Layer Perceptron (MLP) manager that chains Layer objects.

    This class orchestrates the training (fit), inference (predict),
    and persistence (save/load) of the neural network weights.
    """

    def __init__(self, layer_config: List[int], activation_type: str = 'relu'):
        """
        Initializes the network architecture.

        Args:
            layer_config (List[int]): List defining neurons per layer (e.g., [3, 10, 2]).
            activation_type (str): Nonlinearity to use throughout the network.
        """
        self.config: List[int] = layer_config
        self.activation_type: str = activation_type if activation_type is not None else 'relu'
        self.layers: List[Layer] = []

        for i in range(len(layer_config) - 1):
            i_dim = layer_config[i]
            neuron_count = layer_config[i + 1]

            new_layer = Layer(i_dim, neuron_count, activation_type=self.activation_type)
            self.layers.append(new_layer)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """
        Internal method to propagate input through all layers.

        Args:
            x (np.ndarray): Input vector.

        Returns:
            np.ndarray: The final output vector from the last layer.
        """
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def _backprop(self, pred: np.ndarray, target: np.ndarray, n: float = 0.1) -> None:
        """
        Internal method to calculate gradients and update weights.

        Args:
            pred (np.ndarray): The output from the forward pass.
            target (np.ndarray): The expected ground-truth values.
            n (float): Learning rate.
        """
        # Initial error derivative (dLoss/dOutput) for Mean Squared Error
        error = target - pred
        for layer in reversed(self.layers):
            error = layer.backward(error, n)

    def predict(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Generates predictions for a batch of inputs.

        Args:
            X (Union[np.ndarray, List[List[float]]]): A 2D array of input samples.

        Returns:
            np.ndarray: A 2D array of predictions.
        """
        X = np.array(X)
        results = []

        for i in range(len(X)):
            output = self._forward(X[i])
            results.append(output)

        return np.array(results)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, n: float = 0.1, verbose: bool = False) -> None:
        """
        Trains the model using Stochastic Gradient Descent.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            epochs (int): Number of iterations over the dataset.
            n (float): Learning rate.
            verbose (bool): If True, prints loss progress every 10%.
        """
        X = np.array(X)
        y = np.array(y)

        print(f"Start training AINO model over {epochs} epoch...")

        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                pred = self._forward(X[i])
                total_error += np.mean((y[i] - pred) ** 2)
                self._backprop(pred, y[i], n)

            if verbose and (epoch % (max(1, epochs // 10)) == 0):
                avg_loss = total_error / len(X)
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        print("Training is done!")

    def save(self, filename: str = "model.dit") -> None:
        """
        Serializes model weights, biases, and config to a .dit file (NPZ format).

        Args:
            filename (str): Name of the file to save.
        """
        if not filename.endswith(".dit"):
            filename += ".dit"

        data = {
            'config': np.array(self.config),
            'activation': np.array(self.activation_type)
        }

        for i, layer in enumerate(self.layers):
            w_layer = np.array([p.w for p in layer.layer])
            b_layer = np.array([p.b for p in layer.layer])
            data[f"layer_{i}_w"] = w_layer
            data[f"layer_{i}_b"] = b_layer

        with open(filename, 'wb') as f:
            np.savez(f, **data)

        print(f"succesfully save model to '{filename}'")

    @staticmethod
    def load(filename: str) -> Optional['NeuralNetwork']:
        """
        Reconstructs a NeuralNetwork from a saved .dit file.

        Args:
            filename (str): Path to the .dit file.

        Returns:
            Optional[NeuralNetwork]: The restored model, or None if loading fails.
        """
        if not filename.endswith(".dit"):
            print(f"Warning: '{filename}' isnt a .dit model...")

        try:
            with open(filename, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                config = data['config'].tolist()
                activation = str(data['activation'])

                print(f"Architecture: {config}, Non-linearity: {activation}")
                model_baru = NeuralNetwork(config, activation_type=activation)

                for i, layer in enumerate(model_baru.layers):
                    w_layer = data[f"layer_{i}_w"]
                    b_layer = data[f"layer_{i}_b"]
                    for j, perceptron in enumerate(layer.layer):
                        perceptron.w = w_layer[j]
                        perceptron.b = b_layer[j]

                print("AINO model ready")
                return model_baru

        except Exception as e:
            print(f"Error when try to load model: {e}")
            return None