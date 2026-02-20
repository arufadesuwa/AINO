from .backend import xp
from typing import List, Optional
from .layer import Layer


class NeuralNetwork:
    """
    A Multi-Layer Perceptron (MLP) Neural Network capable of running on CPU (NumPy)
    or GPU (CuPy) dynamically.

    Supports mini-batch gradient descent and universal model saving/loading.
    """

    def __init__(self, layer_config: List[int], activation_type: str = 'relu', output_activation: str = 'softmax'):
        """
        Initializes the neural network architecture.

        Args:
            layer_config (List[int]): A list defining the number of neurons in each layer.
                                      For example, [784, 128, 10] creates a network with
                                      784 inputs, 128 hidden neurons, and 10 outputs.
            activation_type (str): The activation function to use for the hidden layers
                                   ('relu', 'sigmoid', 'tanh'). Defaults to 'relu'.
        """
        self.config: List[int] = layer_config
        self.activation_type: str = activation_type if activation_type is not None else 'relu'
        self.output_activation: str = output_activation
        self.layers: List[Layer] = []

        for i in range(len(layer_config) - 1):
            i_dim = layer_config[i]
            neuron_count = layer_config[i + 1]

            if i == len(layer_config) - 2:
                new_layer = Layer(i_dim, neuron_count, activation_type=self.output_activation)
            else:
                new_layer = Layer(i_dim, neuron_count, activation_type=self.activation_type)

            self.layers.append(new_layer)

    def _forward(self, x: xp.ndarray) -> xp.ndarray:
        """
        Performs a forward pass through all layers of the network.

        Args:
            x (xp.ndarray): The input data matrix.

        Returns:
            xp.ndarray: The output predictions from the final layer.
        """
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def _backprop(self, error_gradient: xp.ndarray, n: float = 0.1) -> None:
        """
        Performs backpropagation using a pre-calculated error gradient.

        Args:
            error_gradient (xp.ndarray): The initial gradient of the loss with respect to the output.
            n (float): The learning rate. Defaults to 0.1.
        """
        error = error_gradient

        for layer in reversed(self.layers):
            error = layer.backward(error)
            layer.update(n)

    def predict(self, X: xp.ndarray) -> xp.ndarray:
        """
        Predicts the output for the given input data.

        Args:
            X (xp.ndarray): The input data matrix.

        Returns:
            xp.ndarray: The predicted outputs.
        """
        X = xp.atleast_2d(X)
        return self._forward(X)

    def fit(self, X: xp.ndarray, y: xp.ndarray, epochs: int = 1000, n: float = 0.1,
            batch_size: int = 32, verbose: bool = False, loss_type: str = 'cross_entropy') -> None:
        """
        Trains the neural network using Mini-Batch Gradient Descent.
        """
        if hasattr(xp, 'cuda'):
            xp.cuda.Device(0).use()

        X = xp.ascontiguousarray(xp.asarray(X, dtype=xp.float32))
        y = xp.ascontiguousarray(xp.asarray(y, dtype=xp.float32))

        n_samples = len(X)
        n_outputs = y.shape[1]

        num_batches = int(xp.ceil(n_samples / batch_size))

        print(f"Start training AINO model (Mini-Batch) over {epochs} epochs...")
        print(f"Batch Size: {batch_size} | Samples: {n_samples} | Loss Function: {loss_type.upper()}")

        indices = xp.arange(n_samples)

        for epoch in range(epochs):
            xp.random.shuffle(indices)

            total_error = 0.0

            for i in range(0, n_samples, batch_size):
                idx_batch = indices[i: i + batch_size]
                X_batch = X[idx_batch]
                y_batch = y[idx_batch]

                pred = self._forward(X_batch)

                if loss_type == 'mse':
                    batch_loss = xp.sum((y_batch - pred) ** 2) / batch_size
                    error_grad = pred - y_batch

                elif loss_type == 'cross_entropy':
                    epsilon = xp.float32(1e-7)
                    batch_loss = -xp.sum(y_batch * xp.log(pred + epsilon)) / batch_size
                    error_grad = pred - y_batch

                else:
                    raise ValueError(f"Unknown loss_type: '{loss_type}'.")

                self._backprop(error_grad, n)

                if hasattr(batch_loss, 'get'):
                    total_error += float(batch_loss.get())
                else:
                    total_error += float(batch_loss)

            if verbose and (epoch % (max(1, epochs // 10)) == 0):
                avg_loss = total_error / num_batches
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        print("Training is done!")

    def save(self, filename: str = "model.dit") -> None:
        """
        Saves the network's architecture and trained weights to a file.

        Safely pulls data from GPU to CPU before saving to ensure the resulting
        file is universally loadable regardless of the target machine's hardware.

        Args:
            filename (str): The name of the file to save. Defaults to "model.dit".
        """
        import numpy as np
        from .backend import to_cpu

        if not filename.endswith(".dit"):
            filename += ".dit"

        data = {
            'config': np.array(self.config),
            'activation': np.array(self.activation_type),
            'output_activation': np.array(self.output_activation)  # Tambahan baru
        }

        # Safely extract weights and biases back to CPU memory
        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_w"] = to_cpu(layer.w)
            data[f"layer_{i}_b"] = to_cpu(layer.b)

        with open(filename, 'wb') as f:
            np.savez(f, **data)

        print(f"Successfully saved universal model to '{filename}'")

    @staticmethod
    def load(filename: str) -> Optional['NeuralNetwork']:
        """
        Loads a trained neural network model from a .dit file.

        Args:
            filename (str): The path to the saved model file.

        Returns:
            Optional[NeuralNetwork]: The instantiated NeuralNetwork with loaded weights,
                                     or None if the loading process fails.
        """
        if not filename.endswith(".dit"):
            print(f"Warning: '{filename}' isn't a .dit model...")

        try:
            with open(filename, 'rb') as f:
                data = xp.load(f, allow_pickle=True)
                config = data['config'].tolist()
                activation = str(data['activation'])

                out_activation = str(data['output_activation']) if 'output_activation' in data else 'softmax'

                print(f"Architecture: {config}, Hidden: {activation}, Output: {out_activation}")

                new_model = NeuralNetwork(config, activation_type=activation, output_activation=out_activation)

                for i, layer in enumerate(new_model.layers):
                    layer.w = data[f"layer_{i}_w"]
                    layer.b = data[f"layer_{i}_b"]

                print("AINO model ready")
                return new_model

        except Exception as e:
            print(f"Error when trying to load model: {e}")
            return None